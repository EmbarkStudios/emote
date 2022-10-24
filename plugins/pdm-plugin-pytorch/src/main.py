from __future__ import annotations

from typing import Iterable

import tomlkit

from pdm import termui
from pdm._types import Source
from pdm.cli.commands.base import BaseCommand
from pdm.cli.utils import fetch_hashes, format_lockfile, format_resolution_impossible
from pdm.core import Core
from pdm.models.candidates import Candidate
from pdm.models.repositories import BaseRepository, LockedRepository
from pdm.models.requirements import Requirement, parse_requirement
from pdm.project import Project
from pdm.project.config import ConfigItem
from pdm.resolver import resolve
from pdm.resolver.providers import BaseProvider
from pdm.utils import atomic_open_for_write
from resolvelib.reporters import BaseReporter
from resolvelib.resolvers import ResolutionImpossible, ResolutionTooDeep, Resolver


def sources(project: Project, sources: list) -> list[Source]:
    if all(source.get("name") != "pypi" for source in sources):
        sources.insert(0, project.default_source)
    expanded_sources: list[Source] = [
        Source(
            url=s["url"],
            verify_ssl=s.get("verify_ssl", True),
            name=s.get("name"),
            type=s.get("type", "index"),
        )
        for s in sources
    ]
    return expanded_sources


def get_repository(
    project: Project,
    raw_sources: list,
    cls: type[BaseRepository] | None = None,
    for_install: bool = False,
    lockfile: dict = None,
) -> BaseRepository:
    """Get the repository object"""
    if cls is None:
        cls = project.core.repository_class

    fixed_sources = sources(project, raw_sources)
    if for_install:
        return LockedRepository(lockfile, fixed_sources, project.environment)

    return cls(
        fixed_sources,
        project.environment,
    )


def get_provider(
    project: Project,
    raw_sources: list,
    strategy: str = "all",
    for_install: bool = False,
    lockfile: dict = None,
) -> BaseProvider:
    """Build a provider class for resolver.
    :param strategy: the resolve strategy
    :param tracked_names: the names of packages that needs to update
    :param for_install: if the provider is for install
    :returns: The provider object
    """

    from pdm.resolver.providers import BaseProvider

    repository = get_repository(
        project, raw_sources, for_install=for_install, lockfile=lockfile
    )
    allow_prereleases = False

    return BaseProvider(repository, allow_prereleases, [])


def do_lock(
    project: Project,
    raw_sources: list,
    strategy: str = "all",
    requirements: list[Requirement] | None = None,
) -> dict[str, Candidate]:
    """Performs the locking process and update lockfile."""

    provider = get_provider(project, raw_sources, strategy)
    resolve_max_rounds = int(project.config["strategy.resolve_max_rounds"])
    ui = project.core.ui
    with ui.logging("lock"):
        # The context managers are nested to ensure the spinner is stopped before
        # any message is thrown to the output.
        try:
            with ui.open_spinner(title="Resolving dependencies") as spin:
                reporter = project.get_reporter(requirements, None, spin)
                resolver: Resolver = project.core.resolver_class(provider, reporter)

                mapping, dependencies = resolve(
                    resolver,
                    requirements,
                    project.environment.python_requires,
                    resolve_max_rounds,
                )
                fetch_hashes(provider.repository, mapping)

        except ResolutionTooDeep:
            ui.echo(f"{termui.Emoji.LOCK} Lock failed", err=True)
            ui.echo(
                "The dependency resolution exceeds the maximum loop depth of "
                f"{resolve_max_rounds}, there may be some circular dependencies "
                "in your project. Try to solve them or increase the "
                f"[green]`strategy.resolve_max_rounds`[/] config.",
                err=True,
            )
            raise
        except ResolutionImpossible as err:
            ui.echo(f"{termui.Emoji.LOCK} Lock failed", err=True)
            ui.echo(format_resolution_impossible(err), err=True)
            raise ResolutionImpossible("Unable to find a resolution") from None
        else:
            data = format_lockfile(project, mapping, dependencies)
            ui.echo(f"{termui.Emoji.LOCK} Lock successful")
            return data


def write_lockfile(
    project: Project, lock_name: str, toml_data: dict, show_message: bool = True
) -> None:
    toml_data["metadata"] = project.get_lock_metadata()
    lockfile_file = project.root / lock_name

    with atomic_open_for_write(lockfile_file) as fp:
        tomlkit.dump(toml_data, fp)  # type: ignore
    if show_message:
        project.core.ui.echo(f"Torch locks are written to [success]{lockfile_file}[/].")


class LockCommand(BaseCommand):
    """Generate a lockfile for torch specifically."""

    def handle(self, project, options):
        plugin_config = project.pyproject["tool"]["pdm"]["plugins"]["torch"]
        torch_version_spec = plugin_config["torch-version"]
        resolves = {
            cuda: (f"https://download.pytorch.org/whl/{cuda}/", f"+{cuda}")
            for cuda in plugin_config["cuda-versions"]
        }

        if plugin_config.get("enable-rocm", False):
            for rocm_version in plugin_config.get("rocm-versions", ["4.2"]):
                resolves[f"rocm{rocm_version}"] = (
                    "https://download.pytorch.org/whl/",
                    f"+rocm{rocm_version}",
                )

        if plugin_config.get("enable-cpu", False):
            resolves["cpu"] = ("https://download.pytorch.org/whl/cpu", "")

        results = {}
        for (api, (url, local_version)) in resolves.items():
            local_req = f"{torch_version_spec}{local_version}"

            req = parse_requirement(local_req, False)

            results[local_version] = do_lock(
                project,
                [
                    {
                        "name": "torch",
                        "url": url,
                        "type": "index",
                    }
                ],
                requirements=[req],
            )

        write_lockfile(project, plugin_config["lockfile"], results)


def resolve_candidates_from_lockfile(
    project: Project,
    requirements: Iterable[Requirement],
    raw_sources,
    lockfile: dict,
) -> dict[str, Candidate]:
    ui = project.core.ui
    resolve_max_rounds = int(project.config["strategy.resolve_max_rounds"])
    reqs = [
        req
        for req in requirements
        if not req.marker or req.marker.evaluate(project.environment.marker_environment)
    ]
    with ui.logging("install-resolve"):
        with ui.open_spinner("Resolving packages from lockfile..."):
            reporter = BaseReporter()
            provider = get_provider(
                project, raw_sources, for_install=True, lockfile=lockfile
            )
            resolver: Resolver = project.core.resolver_class(provider, reporter)
            mapping, *_ = resolve(
                resolver,
                reqs,
                project.environment.python_requires,
                resolve_max_rounds,
            )
            fetch_hashes(provider.repository, mapping)

    return mapping


def do_sync(
    project: Project,
    *,
    raw_sources: list,
    requirements: list[Requirement] | None = None,
    lockfile: dict,
) -> None:
    """Synchronize project"""

    candidates = resolve_candidates_from_lockfile(
        project, requirements, raw_sources, lockfile
    )

    handler = project.core.synchronizer_class(
        candidates,
        project.environment,
        False,
        False,
        no_editable=True,
        install_self=False,
        use_install_cache=project.config["install.cache"],
        reinstall=True,
        only_keep=False,
    )

    handler.synchronize()


def read_lockfile(project: Project, lock_name: str) -> None:
    lockfile_file = project.root / lock_name

    data = tomlkit.parse(lockfile_file.read_text("utf-8"))
    return data


class InstallCommand(BaseCommand):
    """Generate a lockfile for torch specifically."""

    def add_arguments(self, parser):
        parser.add_argument(
            "--api", help="the api to use, e.g. cuda version or rocm", required=True
        )

    def handle(self, project, options):
        plugin_config = project.pyproject["tool"]["pdm"]["plugins"]["torch"]
        torch_version_spec = plugin_config["torch-version"]
        resolves = {
            cuda: f"https://download.pytorch.org/whl/{cuda}/"
            for cuda in plugin_config["cuda-versions"]
        }

        if plugin_config.get("enable-rocm", False):
            for rocm_version in plugin_config.get("rocm-versions", ["4.2"]):
                resolves[f"rocm{rocm_version}"] = "https://download.pytorch.org/whl/"

        if plugin_config.get("enable-cpu", False):
            resolves["cpu"] = "https://download.pytorch.org/whl/cpu"

        if options.api not in resolves:
            raise ValueError(
                f"unknown API {options.api}, expected one of {[v for v in resolves]}"
            )

        lockfile = read_lockfile(project, plugin_config["lockfile"])
        spec_for_version = lockfile[options.api]
        source = resolves[options.api]
        local_req = f"{torch_version_spec}+{options.api}"
        req = parse_requirement(local_req, False)
        do_sync(
            project,
            raw_sources=[
                {
                    "name": "torch",
                    "url": source,
                    "type": "index",
                }
            ],
            requirements=[req],
            lockfile=spec_for_version,
        )


def torch_plugin(core: Core):
    core.register_command(LockCommand, "torch-lock")
    core.register_command(InstallCommand, "torch-install")
    core.add_config("hello.name", ConfigItem("The person's name", "John"))
