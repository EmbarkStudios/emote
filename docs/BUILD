python_sources()

resources(name = "index", sources = ["index.rst"])

DOC_DEPS = [
    ":index",
    "//:adr",
    "//docs",
    "//emote",
    "//emote/algorithms",
    "//emote/algorithms/genrl",
    "//emote/callbacks",
    "//emote/env",
    "//emote/env/box2d",
    "//emote/extra",
    "//emote/memory",
    "//emote/mixins",
    "//emote/models",
    "//emote/nn",
    "//emote/utils",
    "//experiments/gym",
]

SPHINX_DEPS = [
    "//:reqs-dev#myst-parser",
    "//:reqs-dev#Sphinx",
    "//:reqs-dev#sphinx-autodoc-typehints",
    "//:reqs-dev#sphinx-rtd-theme",
    "//:reqs-dev#sphinxcontrib-apidoc",
]

pex_binary(
    name="sphinx-apidoc",
    script="sphinx-apidoc",
    dependencies=SPHINX_DEPS,
)

adhoc_tool(
    name="build-apidoc",
    runnable=":sphinx-apidoc",
    args=[
        "--force",
        "--module-first",
        "--output-dir",
        "docs",
        "./../",
    ],
    execution_dependencies = DOC_DEPS,
    output_directories=["./"],
    root_output_directory="./",
    log_output=True,
)

pex_binary(
    name="sphinx-build",
    script="sphinx-build",
    dependencies=SPHINX_DEPS + DOC_DEPS,
)

adhoc_tool(
    name="build-html",
    runnable=":sphinx-build",
    args=["-M", "html", ".", "_build"],
    execution_dependencies=[":build-apidoc"] + DOC_DEPS,
    output_directories=["_build/html"],
    root_output_directory="./_build/html",
    log_output=True,
)

archive(name="docs-out", files=[":build-html"], format="zip")
