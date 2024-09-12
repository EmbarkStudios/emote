python_test_utils(
    name="test_utils",
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
