__defaults__(
    {
        python_source: dict(resolve=parametrize("cpu", "gpu", "base")),
        python_sources: dict(resolve=parametrize("cpu", "gpu", "base")),
        pex_binary: dict(resolve=parametrize("cpu", "gpu", "base")),
    }
)

python_requirements(
    name="reqs",
    source="pyproject.toml",
)
