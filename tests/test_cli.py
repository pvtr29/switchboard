import subprocess
import sys

import pytest

from switchboard import __version__
from switchboard.client.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"sb {__version__}"


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: sb" in capsys.readouterr().out


def test_installed_entry_point() -> None:
    # Runs the same module the `sb` script points at, in a fresh interpreter.
    result = subprocess.run(
        [sys.executable, "-c", "from switchboard.client.cli import main; main(['--version'])"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"sb {__version__}"
