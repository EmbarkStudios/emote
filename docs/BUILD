python_sources()

resources(
    name="index",
    sources=["index.rst"],
)

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
    "//:emote-base#myst-parser",
    "//:emote-base#Sphinx",
    "//:emote-base#sphinx-autodoc-typehints",
    "//:emote-base#sphinx-rtd-theme",
    "//:emote-base#sphinxcontrib-apidoc",
]

pex_binary(
    name="sphinx-apidoc",
    dependencies=SPHINX_DEPS,
    script="sphinx-apidoc",
)

adhoc_tool(
    name="build-apidoc",
    args=[
        "--force",
        "--module-first",
        "--output-dir",
        "docs",
        "./../",
    ],
    execution_dependencies=DOC_DEPS,
    log_output=True,
    output_directories=["./"],
    root_output_directory="./",
    runnable=":sphinx-apidoc",
)

pex_binary(
    name="sphinx-build",
    dependencies=SPHINX_DEPS + DOC_DEPS,
    script="sphinx-build",
)

adhoc_tool(
    name="build-html",
    args=[
        "-M",
        "html",
        ".",
        "_build",
    ],
    execution_dependencies=[":build-apidoc"] + DOC_DEPS,
    log_output=True,
    output_directories=["_build/html"],
    root_output_directory="./_build/html",
    runnable=":sphinx-build",
)

experimental_wrap_as_resources(
    name="html-files",
    inputs=[":build-html"],
)

archive(
    name="docs-out",
    files=[":build-html"],
    format="zip",
)
