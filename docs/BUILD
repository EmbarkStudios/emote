python_sources(name="py")

DOC_DEPS = [
    emote_dependency_path(":adr"),
    emote_dependency_path("/emote@resolve=base"),
    emote_dependency_path("/emote/algorithms@resolve=base"),
    emote_dependency_path("/emote/algorithms/genrl@resolve=base"),
    emote_dependency_path("/emote/callbacks@resolve=base"),
    emote_dependency_path("/emote/env@resolve=base"),
    emote_dependency_path("/emote/env/box2d@resolve=base"),
    emote_dependency_path("/emote/extra@resolve=base"),
    emote_dependency_path("/emote/memory@resolve=base"),
    emote_dependency_path("/emote/mixins@resolve=base"),
    emote_dependency_path("/emote/models@resolve=base"),
    emote_dependency_path("/emote/nn@resolve=base"),
    emote_dependency_path("/emote/utils@resolve=base"),
    emote_dependency_path("/experiments/gym@resolve=base"),
]

pex_binary(
    name="adr",
    entry_point="adr.py",
    resolve="base",
)

resource(
    name="template",
    source="./SUMMARY.tmpl",
)

pex_binary(
    name="apibook",
    entry_point="apibook.cli:main",
    dependencies=["//:apibook#apibook"],
    resolve="apibook",
)

adhoc_tool(
    name="build-apibook",
    runnable=":apibook",
    args=["../emote", "src", "--summary-template-file", "SUMMARY.tmpl"],
    execution_dependencies=DOC_DEPS
    + [
        ":template",
    ],
    log_output=True,
    output_directories=["src"],
    root_output_directory="/",
)

system_binary(
    name="dot",
    binary_name="dot",
)

pex_binary(
    name="graphviz",
    entry_point="graphviz.py",
    resolve="base",
)

resources(
    name="dot-files",
    sources=["raw/*.dot"],
)

adhoc_tool(
    name="build-dot",
    runnable=":graphviz",
    args=["./raw", "./src"],
    execution_dependencies=[":dot-files"],
    runnable_dependencies=[":dot"],
    log_output=True,
    output_directories=["src"],
    root_output_directory="/",
)

md_book(
    name="book",
    sources=["book.toml", "src/*.md", "adr.py"],
    dependencies=DOC_DEPS
    + [
        ":build-apibook",
        ":build-dot",
    ],
)

# Hack to somewhat easily serve the generated doc book
pex_binary(
    name="serve",
    entry_point="http.server",
    resolve="base",
    args=["-d", "dist/docs/book"],
    dependencies=[":book"],
)
