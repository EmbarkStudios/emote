python_test_utils(
    name="test_utils",
    dependencies=[
        "//:root#pytest",
        "//:root#gymnasium",
        "//:root#pygame",
        "//:root#box2d-py",
    ],
)

python_tests(
    overrides={
        "test_memory_exports.py": {
            "dependencies": [
                ":memory-artifact",
            ]
        },
    },
)


resource(
    name="memory-artifact",
    source="data/export-v1.zip",
)
