import pytest

from trainguard import cli
from trainguard.state import AppPaths


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    """Keep every test away from the user's real ~/.train-guard state."""
    home = tmp_path / "home"
    tg_home = home / ".train-guard"
    monkeypatch.setenv("TRAIN_GUARD_HOME", str(tg_home))
    # Tests run inside Claude Code or Codex would otherwise record that session.
    for variable in cli._AGENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(cli, "HOME", home)
    cli._sync_path_aliases(AppPaths.from_environment())
    yield


@pytest.fixture
def app_paths():
    paths = AppPaths.from_environment()
    paths.ensure()
    return paths
