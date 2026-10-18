"""Keep dependency-heavy CI for executable changes; never hide unknown paths."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT_DOCS = {
    "README.md", "README.rst", "CHANGELOG.md", "CHANGELOG.rst",
    "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "RELEASING.md",
    "ARCHITECTURE.md", "INSTALL.md", "USAGE.md", "LICENSE", "LICENSE.md",
}


def documentation_only(path):
    parts = PurePosixPath(path).parts
    if any(part.lower() in {"test", "tests", "fixture", "fixtures", "config", "configs", ".github"} for part in parts):
        return False
    if PurePosixPath(path).name.lower().startswith(("requirements", "constraints")):
        return False
    return path in ROOT_DOCS or (
        len(parts) > 1 and parts[0] == "docs"
        and PurePosixPath(path).suffix in {".md", ".rst"}
    )


def needs_code_checks(event, event_name):
    # Tags, manual invocations and unfamiliar events retain full qualification.
    if event_name == "pull_request":
        base, head = event["pull_request"]["base"]["sha"], event["pull_request"]["head"]["sha"]
        revision = "..."
    elif event_name == "push" and not event.get("ref", "").startswith("refs/tags/"):
        base, head = event.get("before", ""), event.get("after", "")
        revision = ".."
        if not base or set(base) == {"0"}:
            return True
    else:
        return True
    if not all(re.fullmatch(r"[0-9a-f]{40,64}", value) for value in (base, head)):
        raise ValueError("Expected immutable Git commit identities")
    # No rename detection: a source-to-doc rename must retain its source deletion.
    raw = subprocess.check_output([
        "git", "diff", "--no-renames", "--name-only", "-z", f"{base}{revision}{head}", "--",
    ])
    paths = [path.decode("utf-8", errors="surrogateescape") for path in raw.split(b"\0") if path]
    return not paths or not all(documentation_only(path) for path in paths)


if __name__ == "__main__":
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    required = needs_code_checks(event, os.environ["GITHUB_EVENT_NAME"])
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"run_code={str(required).lower()}\n")
    print("Full code checks required" if required else "Documentation-only change; dependency-heavy jobs skipped")
