"""Native OS folder-picker dialog, run off the Streamlit UI thread.

Lives outside ``app/`` so the module split's "no threading in app/" rule
(only ``IngestionManager`` may own a thread within the app package) holds
literally, while the dialog itself — a real, kept feature per the spec's
2026-07-19 revision of the local-folder design — still needs a background
thread so a blocking native subprocess never freezes the UI.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# A bare `choose folder` run from osascript belongs to a faceless
# background process: the panel opens without keyboard focus, never
# raises above the browser, and with a fullscreen browser it lands on a
# different Space entirely — the click looks like a no-op. Routing it
# through System Events and activating first makes the dialog take
# focus and switch to the user's Space. First use prompts once for
# Automation permission (osascript -> System Events).
_MACOS_CHOOSE_FOLDER_SCRIPT = (
    'tell application "System Events"\n'
    "activate\n"
    'return POSIX path of (choose folder with prompt "Select a codebase folder")\n'
    "end tell"
)


@dataclass
class FolderPickResult:
    path: str | None = None
    error: str | None = None


def _normalize_dialog_path(raw: str) -> str:
    """Trim a trailing separator without collapsing a root selection.

    A bare ``rstrip("/\\\\")`` turns macOS's ``/`` into an empty string
    (mistaken for a cancel) and Windows's ``C:\\`` into the drive-relative
    ``C:`` (resolves to the process's cwd instead of the drive root).
    """
    path = raw.strip()
    stripped = path.rstrip("/\\")
    if not stripped:
        return path[:1]  # "/" (or "\") selected: keep the root itself.
    if len(stripped) == 2 and stripped[1] == ":":
        return stripped + "\\"  # Windows drive root, e.g. "C:\\".
    return stripped


class FolderPicker:
    """Runs a single native folder-picker dialog per (per-session) request token.

    Each request gets its own token instead of a shared module-level dict,
    so results from one browser session/tab can never be read by another.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._token: object | None = None
        self._result: FolderPickResult | None = None

    def open(self) -> object | None:
        """Start the dialog in a background thread; returns a request token.

        Returns None if a dialog is already open (no-op instead of
        stacking a second native dialog on top of it).
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return None
            token = object()
            self._token = token
            self._result = None

            def _run() -> None:
                path, error = _pick_folder_path()
                with self._lock:
                    if self._token is token:
                        self._result = FolderPickResult(path=path, error=error)

            thread = threading.Thread(target=_run, daemon=True)
            self._thread = thread

        thread.start()
        return token

    def is_open(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def poll(self, token: object) -> FolderPickResult | None:
        """Return the result for ``token`` once ready, else None.

        A stale token (from a previous dialog) never returns a result, so
        a session that missed its own dialog can't accidentally pick up
        a later one's answer.
        """
        with self._lock:
            if self._token is not token or self._result is None:
                return None
            return self._result


def _pick_folder_path() -> tuple[str | None, str | None]:
    """Show the native folder picker and return ``(path, error)``.

    ``(path, None)`` on selection, ``(None, None)`` on cancel/timeout,
    and ``(None, message)`` for real failures the user should see.
    """
    try:
        if sys.platform == "darwin":
            result = subprocess.run(  # noqa: S603
                ["osascript", "-e", _MACOS_CHOOSE_FOLDER_SCRIPT],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        elif sys.platform == "win32":
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$d.Description = 'Select a codebase folder'; "
                "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath } else { '' }"
            )
            result = subprocess.run(  # noqa: S603
                ["powershell", "-NoProfile", "-Command", ps_script],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        elif shutil.which("zenity"):
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title=Select a codebase folder"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        elif shutil.which("kdialog"):
            result = subprocess.run(
                ["kdialog", "--getexistingdirectory", "."],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        else:
            logger.warning("No folder dialog tool available (install zenity or kdialog)")
            return None, "No folder dialog tool available — install zenity or kdialog, or type a path instead."
        path = _normalize_dialog_path(result.stdout)
        if path:
            return path, None
        stderr = result.stderr.strip()
        if result.returncode == 0 or not stderr or "cancel" in stderr.lower():
            return None, None
        logger.warning("Folder dialog failed: %s", stderr)
        if "-1743" in stderr or "Not authorized" in stderr:
            return None, (
                "The folder dialog needs Automation permission: allow your "
                "terminal to control System Events under System Settings -> "
                "Privacy & Security -> Automation, then try again."
            )
        return None, f"Folder dialog failed: {stderr}"
    except subprocess.TimeoutExpired:
        logger.warning("Folder dialog timed out waiting for a selection")
        return None, None
    except OSError as exc:
        logger.warning("Folder dialog failed: %s", exc)
        return None, f"Folder dialog failed: {exc}"
