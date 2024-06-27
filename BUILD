TORCH_VERSION = "1.12.0"
CUDA_VERSION = "cu116"

TORCH_VARIANTS = {
    "base": f"=={TORCH_VERSION},!={TORCH_VERSION}+cpu,!={TORCH_VERSION}+{CUDA_VERSION}",
    "cpu": f"=={TORCH_VERSION}+cpu,!={TORCH_VERSION}+{CUDA_VERSION}",
    "gpu": f"=={TORCH_VERSION}+{CUDA_VERSION},!={TORCH_VERSION}+cpu",
}


if is_standalone():
    resolves = ["cpu", "gpu", "base"]

    __defaults__(
        {
            python_source: dict(resolve=parametrize(*resolves)),
            python_sources: dict(resolve=parametrize(*resolves)),
        }
    )

    TOOLS = {
        "pytest": [
            "pytest-cov!=2.12.1,<3.1,>=2.12",
            "pytest-xdist<3,>=2.5",
            "pytest~=8.0",
            "pytest-platform-markers",
            "pytest-rerunfailures",
            "pytest-benchmark==4.0.0",
        ],
        "black": ["black>=22.6.0,<24"],
        "ipython": ["ipython>=7.27.0,<8"],
        "isort": ["isort[pyproject,colors]>=5.9.3,<6.0"],
        "docformatter": ["docformatter[tomli]"],
        "apibook": ["apibook~=0.1.0"],
    }

    for tool, reqs in TOOLS.items():
        python_requirement(
            name=tool,
            requirements=reqs,
            resolve=tool,
        )

    for resolve in resolves:
        python_requirements(
            name=resolve,
            source="pyproject.toml",
            resolve=resolve,
            module_mapping={
                "protobuf": ["google.protobuf"],
                "opencv-python": ["cv2"],
            },
            overrides={
                "gymnasium": {"dependencies": [f":{resolve}#box2d-py", f":{resolve}#pygame"]}
            },
        )

        python_requirement(
            name=f"pytest-{resolve}",
            requirements=TOOLS["pytest"],
            resolve=resolve,
        )

resources(
    name="adr",
    sources=["adr/*.md"],
)

resources(name="package_data", sources=["pyproject.toml", "README.md"])

python_distribution(
    name="package",
    dependencies=[
        ":package_data",
        emote_dependency_path("/emote:emote@resolve=base"),
        emote_dependency_path("/emote/algorithms@resolve=base"),
        emote_dependency_path("/emote/algorithms/genrl@resolve=base"),
        emote_dependency_path("/emote/memory@resolve=base"),
        emote_dependency_path("/emote/nn@resolve=base"),
    ],
    provides=python_artifact(
        name="emote",
        version="0.1.0",
        long_description_content_type="markdown",
    ),
    long_description_path="./README.md",
    interpreter_constraints=[">=3.10,<3.11"],
)

pex_binary(
    name="tensorboard",
    entry_point="tensorboard.main:run_main",
    dependencies=["//:cpu#tensorboard"],
    resolve="cpu",
)
