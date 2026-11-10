from pathlib import Path

import pytest

from switchboard.client.cli import build_parser, find_api_key, main, read_env_file


def test_read_env_file(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("# comment\n\nGEMINI_API_KEY='abc'\nexport OTHER=\"x y\"\nBROKEN\n  SPACED = v \n")
    assert read_env_file(f) == {"GEMINI_API_KEY": "abc", "OTHER": "x y", "SPACED": "v"}
    assert read_env_file(tmp_path / "missing") == {}


def test_environment_wins_over_the_file_and_gemini_over_google(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("GOOGLE_API_KEY=from-file-google\nGEMINI_API_KEY=from-file\n")
    assert find_api_key({}, f) == "from-file"
    assert find_api_key({"GOOGLE_API_KEY": "env-google"}, f) == "env-google"
    assert find_api_key({"GEMINI_API_KEY": "e1", "GOOGLE_API_KEY": "e2"}, f) == "e1"
    assert find_api_key({"GEMINI_API_KEY": ""}, tmp_path / "none") is None


def test_submit_parses_gemini_options() -> None:
    args = build_parser().parse_args(
        ["submit", "--agent", "gemini", "--task", "roman", "--model", "a", "--model", "b"]
    )
    assert (args.agent, args.task, args.models) == ("gemini", "roman", ["a", "b"])


def test_tasks_lists_the_suite(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tasks"]) == 0
    names = [line.split()[0] for line in capsys.readouterr().out.splitlines()]
    assert {"roman", "intervals", "lru", "slugify"} <= set(names)
