# 1. "Nightly" continuous releases

Date: 2022-10-21

## Status

Accepted

## Context

It would be useful for CI purposes, testing, and local development to be able to install wheels that have gone through
CI; rather than pulling the whole git repository and installing. This somewhat aligns with the
`git+ssh://.../owner/repo#egg=...` syntax, but that is still a repo pull and not easily distributable.

## Decision

Each night there'll be a nightly build done on the latest main; IFF there have been commits in the last 24 hours. This
will be tagged as `latest` and relased as `pre-release` on GitHub.

## Consequences

We'll need to maintain somewhat reasonable stability and testing on average builds to support nightly builds. Nightly
builds don't need to be as thoroughly tested.
