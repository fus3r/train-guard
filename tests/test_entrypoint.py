from __future__ import annotations

import runpy

import pytest

from trainguard import cli


def test_python_module_entrypoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "main", lambda: 7)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("trainguard.__main__", run_name="__main__")

    assert exit_info.value.code == 7
