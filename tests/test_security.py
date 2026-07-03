"""Security posture tests."""

from __future__ import annotations

from pathlib import Path

from src.security import scan_file_for_secrets


def test_gitignore_contains_env():
    content = Path(".gitignore").read_text(encoding="utf-8")
    assert ".env" in content.splitlines()


def test_env_example_contains_placeholders_only():
    content = Path(".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=your_openai_api_key_here" in content
    assert "LANGCHAIN_API_KEY=your_langsmith_api_key_here" in content


def test_source_files_do_not_contain_common_secret_patterns():
    roots = [Path("src"), Path("api"), Path("dashboard"), Path("tests"), Path("prompts"), Path("docs")]
    scanned = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".txt", ".md"}:
                scanned.append(path)
                assert not scan_file_for_secrets(path), f"Secret-like pattern found in {path}"
    assert scanned
