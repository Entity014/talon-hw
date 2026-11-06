"""Guard against commanding the MCU while Unitree's sport controller is also commanding it."""

import subprocess


def controller_running():
    """A free UDP 8080 proves nothing here: A1_sport_* talks to the MCU from its own ports."""
    return subprocess.call(["pgrep", "-f", "bin/A1_sport_"], stdout=subprocess.DEVNULL) == 0


def require_stopped():
    if controller_running():
        raise SystemExit("A1_sport_* is running. Stop it first with: sudo scripts/sport_mode.sh stop")
