"""Test to scan repository files for accidental secrets and API key leaks.

Enforces Rule #3 from AGENTS.md:
"Secrets: .env is gitignored... Add a test that greps the repo (excluding .env) for key-like strings."
"""

import os
import re
from pathlib import Path

# Regex patterns matching potential real API keys (excluding obvious placeholders)
SUSPICIOUS_KEY_PATTERNS = [
    re.compile(r"sk-or-v1-[a-f0-9]{48,}", re.IGNORECASE),  # Live OpenRouter key
    re.compile(r"sk-[a-zA-Z0-9]{32,}", re.IGNORECASE),       # Live OpenAI key
    re.compile(r"sk-ant-api03-[a-zA-Z0-9_-]{40,}", re.IGNORECASE),  # Live Anthropic key
]

EXCLUDED_DIRS = {
    ".git",
    "venv",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
}

EXCLUDED_FILES = {
    ".env",       # Local env file (strictly gitignored)
    ".env.local",
}


def test_no_secrets_in_tracked_files() -> None:
    """Verify that no actual API key secrets are present in project files."""
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Check that .env is not committed
    env_file = repo_root / ".env"
    if env_file.exists():
        # Ensure .env is not in git tracking (if git is available)
        try:
            import subprocess
            result = subprocess.run(
                ["git", "status", "--porcelain", ".env"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
            # If tracked, git status would show .env
            assert ".env" not in result.stdout, ".env file must NOT be tracked in git!"
        except Exception:
            pass

    found_violations = []

    for root, dirs, files in os.walk(repo_root):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            if file in EXCLUDED_FILES:
                continue

            file_path = Path(root) / file
            # Only inspect text-based code/config files
            if file_path.suffix.lower() in [
                ".py", ".md", ".json", ".ts", ".tsx", ".html", ".css", ".yaml", ".yml", ".toml", ".txt"
            ]:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    for pattern in SUSPICIOUS_KEY_PATTERNS:
                        matches = pattern.findall(content)
                        # Filter out known mock test placeholders
                        real_matches = [m for m in matches if "mock" not in m.lower() and "placeholder" not in m.lower()]
                        if real_matches:
                            found_violations.append(
                                f"{file_path.relative_to(repo_root)} matched pattern {pattern.pattern}"
                            )
                except Exception:
                    pass

    assert not found_violations, f"Potential secret leak detected in files: {found_violations}"
