"""Silent macOS notifications (banner, no sound) via osascript."""
import subprocess


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notify(title: str, text: str):
    script = f"display notification {_quote(text[:200])} with title {_quote('Class · ' + title[:60])}"
    try:
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass  # notifications are a nice-to-have; never break the class for them
