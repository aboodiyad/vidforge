import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own VidForge data folder."""
    monkeypatch.setenv("VIDFORGE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"
