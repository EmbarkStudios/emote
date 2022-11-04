from __future__ import annotations

import sys

from typing import Iterable

import tomlkit

from pdm import termui
from pdm._types import Source
from pdm.cli.actions import resolve_candidates_from_lockfile
from pdm.cli.commands.base import BaseCommand
from pdm.cli.utils import (
    fetch_hashes,
    format_lockfile,
    format_resolution_impossible,
    translate_groups,
)
from pdm.core import Core
from pdm.models.caches import SafeFileCache
from pdm.models.candidates import Candidate, _find_best_match_link
from pdm.models.repositories import BaseRepository, LockedRepository
from pdm.models.requirements import Requirement, parse_requirement
from pdm.models.specifiers import get_specifier
from pdm.project import Project
from pdm.project.config import ConfigItem
from pdm.resolver import resolve
from pdm.resolver.providers import BaseProvider
from pdm.termui import Verbosity
from pdm.utils import atomic_open_for_write
from resolvelib.reporters import BaseReporter
from resolvelib.resolvers import ResolutionImpossible, ResolutionTooDeep, Resolver


class FetcherCommand(BaseCommand):
    """Generate a lockfile for torch specifically."""

    def add_arguments(self, parser):
        parser.add_argument(
            "-G",
            "--group",
            dest="groups",
            metavar="GROUP",
            action="append",
            help="Select group of optional-dependencies "
            "or dev-dependencies(with -d). Can be supplied multiple times, "
            'use ":all" to include all groups under the same species.',
            default=[],
        )

    def handle(self, project, options):
        groups = translate_groups(project, True, True, options.groups)

        requirements = []
        for group in groups:
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
