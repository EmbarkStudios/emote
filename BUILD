python_requirements(
    name="root",
    source="pyproject.toml",
)

# FIXME: Not consistent with below
python_requirements(
    name = "reqs-dev",
    source = "requirements/dev.txt",
)

MODULE_MAPPINGS = {
    "torch": ["torch==1.12.0"],
    "atari": [
        "gymnasium>=0.27.1",
        "box2d-py>=2.3.5",
        "pygame>=2.1.0",
    ],
    "wandb": ["wandb>=0.14.0"],
    "ci": ["gsutil>=4.66", "emote[atari,wandb]"],

}

# FIXME: Not consistent with above
python_requirement(
    name = "reqs-opt",
    requirements = [
        "torch==1.12.0",
        "gymnasium>=0.27.1",
        "box2d-py>=2.3.5",
        "pygame>=2.1.0",
        "wandb>=0.14.0",
        "gsutil>=4.66",
    ],
    modules = MODULE_MAPPINGS,
)

resources(name = "adr", sources = ["adr/*.md"])

TOOLS = {
    "pytest": [
        "pytest-cov!=2.12.1,<3.1,>=2.12",
        "pytest-xdist<3,>=2.5",
        "pytest==7.0.*",
        "pytest-platform-markers",
        "pytest-rerunfailures",
        "pytest-benchmark==4.0.0",
    ],
    "black": ["black>=22.6.0,<24"],
    "ipython": ["ipython>=7.27.0,<8"],
    "isort": ["isort[pyproject,colors]>=5.9.3,<6.0"],
    "docformatter": ["docformatter[tomli]"],
}

for tool, reqs in TOOLS.items():
    python_requirement(
        name=tool,
        requirements=reqs,
        resolve=tool,
    )
