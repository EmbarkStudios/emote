from __future__ import annotations

from pdm.cli.actions import resolve_candidates_from_lockfile
from pdm.cli.commands.base import BaseCommand
from pdm.cli.filters import GroupSelection
from pdm.cli.options import (
    clean_group,
    dry_run_option,
    groups_group,
    install_group,
    lockfile_option,
    skip_option,
    venv_option,
)
from pdm.core import Core
from pdm.models.caches import SafeFileCache
from pdm.models.candidates import _find_best_match_link


class FetcherCommand(BaseCommand):
    """Generate a lockfile for torch specifically."""

    arguments = (
        *BaseCommand.arguments,
        groups_group,
        dry_run_option,
        lockfile_option,
        skip_option,
        clean_group,
        install_group,
        venv_option,
    )

    def handle(self, project, options):
        selection = GroupSelection.from_options(project, options)
        requirements = []
        for group in selection:
            requirements.extend(project.get_dependencies(group).values())
        candidates = resolve_candidates_from_lockfile(project, requirements)

        with project.core.ui.open_spinner("Preparing candidates..."):
            prepared = [
                candidate.prepare(project.environment)
                for candidate in candidates.values()
            ]

        sfc = SafeFileCache(project.cache("http"))

        links = []
        with project.core.ui.open_spinner("Resolving links..."):
            with project.environment.get_finder() as finder:
                for c in prepared:
                    link = _find_best_match_link(
                        finder,
                        c.req.as_pinned_version(c.candidate.version),
                        c.candidate.hashes,
                        ignore_compatibility=False,
                    ).url

                    links.append(
                        {
                            "link": link,
                            "path": sfc._get_cache_path(link),
                        }
                    )

        import json

        print(json.dumps({"links": links}, indent=4))


def fetcher_plugin(core: Core):
    core.register_command(FetcherCommand, "fetch")
