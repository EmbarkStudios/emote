python_requirements(
    name="emote-base",
    source="pyproject.toml",
)

resources(
    name="adr",
    sources=["adr/*.md"],
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
}

for tool, reqs in TOOLS.items():
    python_requirement(
        name=tool,
        requirements=reqs,
        resolve=tool,
    )
