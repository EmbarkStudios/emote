python_requirements(
    name="root",
    source="pyproject.toml",
)

# python_requirements(
#     name = "reqs",
#     source = "requirements/common.txt",
# )

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
    resolve = "opt",
)

resources(name = "adr", sources = ["adr/*.md"])
