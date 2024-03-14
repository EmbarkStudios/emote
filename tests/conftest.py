""""""

import builtins

import pytest


@pytest.fixture
def hide_pkg(monkeypatch):
    import_orig = builtins.__import__

    def _install(hide):
        def mocked_import(name, *args, **kwargs):
            if name == hide:
                raise ImportError(f"No module named '{name}'")
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    return _install
