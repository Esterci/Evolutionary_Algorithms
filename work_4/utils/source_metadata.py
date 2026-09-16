"""Optional Git provenance for environments without Git or repository metadata."""

import subprocess


def git_revision(directory):
    """Return the commit and an explicit reason when it cannot be obtained."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        return None, f"{type(error).__name__}: {error}"
    if result.returncode != 0:
        return None, result.stderr.strip() or f"git exited with status {result.returncode}"
    return (result.stdout.strip(), None) if result.stdout.strip() else (None, "git returned an empty revision")
