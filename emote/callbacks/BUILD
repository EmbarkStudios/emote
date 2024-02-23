for resolve in ["base", "cpu", "gpu"]:
    python_sources(
        name = resolve,
        overrides = {"logging.py":  {"dependencies": [f"//:{resolve}#tensorboard"]}},
        resolve = resolve,
    )
