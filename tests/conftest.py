import pytest

from trainguard import cli


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    """Keep every test away from the user's real ~/.train-guard state."""
    home = tmp_path / "home"
    tg_home = home / ".train-guard"
    monkeypatch.setattr(cli, "HOME", home)
    monkeypatch.setattr(cli, "TG_HOME", tg_home)
    monkeypatch.setattr(cli, "RUNDIR", tg_home / "run")
    monkeypatch.setattr(cli, "LOGDIR", tg_home / "logs")
    monkeypatch.setattr(cli, "PERSIST", tg_home / "persist")
    monkeypatch.setattr(cli, "CONFIGF", tg_home / "config.json")
    cli._Cool.on = False
    cli._LINUX_AFFINITY.clear()
    yield
    cli._Cool.on = False
    cli._LINUX_AFFINITY.clear()
