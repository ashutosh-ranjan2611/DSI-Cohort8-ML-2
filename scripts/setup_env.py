#!/usr/bin/env python3
"""Cross-platform project setup using uv.

Detects the operating system, installs uv if missing, creates a Python 3.12
virtual environment at .venv, and syncs all dependencies from pyproject.toml.

Usage:
    python scripts/setup_env.py              # interactive setup
    python scripts/setup_env.py --yes        # non-interactive (auto-approve)
    python scripts/setup_env.py --dev        # include optional [dev] extras
    python scripts/setup_env.py --yes --dev  # non-interactive with dev extras
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys

# ── Constants ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_DIR = os.path.join(ROOT, ".venv")
PYTHON_VERSION = "3.12"

log = logging.getLogger("setup")

DIVIDER = "=" * 62


# ── Logging ──────────────────────────────────────────────────────────────────
def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, log the command, and return the result."""
    log.debug("  $ %s", " ".join(cmd))
    return subprocess.run(cmd, check=False, **kwargs)


def _banner(text: str) -> None:
    log.info("")
    log.info(DIVIDER)
    log.info("  %s", text)
    log.info(DIVIDER)


def _step(number: int, total: int, text: str) -> None:
    log.info("")
    log.info("--- Step %d/%d: %s ---", number, total, text)


# ── Platform detection ───────────────────────────────────────────────────────
def detect_platform() -> str:
    """Return platform.system() ('Windows' | 'Darwin' | 'Linux') and log info."""
    os_name = platform.system()
    arch = platform.machine()
    py_ver = platform.python_version()
    log.info("  Platform : %s %s", os_name, arch)
    log.info("  Python   : %s (%s)", py_ver, sys.executable)
    log.info("  Project  : %s", ROOT)
    return os_name


# ── Step 1 — Ensure uv is installed ─────────────────────────────────────────
def _find_uv() -> str | None:
    """Look for uv on PATH and in common install locations."""
    uv = shutil.which("uv")
    if uv:
        return uv
    # Common install locations (official installer puts uv here)
    if sys.platform == "win32":
        candidates = [
            os.path.expanduser(r"~\.local\bin\uv.exe"),
            os.path.expanduser(r"~\.cargo\bin\uv.exe"),
        ]
    else:
        candidates = [
            os.path.expanduser("~/.local/bin/uv"),
            os.path.expanduser("~/.cargo/bin/uv"),
        ]
    for c in candidates:
        if os.path.isfile(c):
            parent = os.path.dirname(c)
            os.environ["PATH"] = parent + os.pathsep + os.environ.get("PATH", "")
            log.debug("  Found uv at %s — added %s to PATH", c, parent)
            return c
    return None


def ensure_uv(os_name: str, noninteractive: bool) -> str:
    """Ensure uv is available. Install it if missing. Returns path to uv."""
    uv = _find_uv()
    if uv:
        ver = subprocess.run(
            [uv, "--version"], capture_output=True, text=True, check=False,
        )
        ver_str = ver.stdout.strip() if ver.returncode == 0 else "unknown"
        log.info("uv found   : %s  (%s)", uv, ver_str)
        return uv

    log.warning("uv is NOT installed on this machine.")
    if not noninteractive:
        ans = input("  Install uv now? [Y/n]: ").strip().lower() or "y"
        if ans not in ("y", "yes"):
            log.error("uv is required for this project. Exiting.")
            sys.exit(1)

    # --- platform-specific official installer ---
    if os_name == "Windows":
        log.info("Installing uv via official PowerShell installer ...")
        r = _run([
            "powershell", "-ExecutionPolicy", "ByPass", "-NoProfile", "-Command",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ])
    else:
        log.info("Installing uv via official shell installer ...")
        curl = shutil.which("curl")
        if not curl:
            log.error("curl not found — cannot run installer. Install uv manually.")
            sys.exit(1)
        r = _run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])

    if r.returncode != 0:
        log.warning(
            "Official installer exited with %d. Trying 'pip install uv' fallback ...",
            r.returncode,
        )
        r = _run([sys.executable, "-m", "pip", "install", "uv"])
        if r.returncode != 0:
            log.error(
                "Could not install uv. Install manually: "
                "https://docs.astral.sh/uv/getting-started/installation/"
            )
            sys.exit(1)

    uv = _find_uv()
    if not uv:
        log.error(
            "uv was installed but is not on PATH. "
            "Restart your terminal (or add ~/.local/bin to PATH) and retry."
        )
        sys.exit(1)

    log.info("uv installed successfully: %s", uv)
    return uv


# ── Step 2 — Platform-specific checks ───────────────────────────────────────
def ensure_libomp_mac(noninteractive: bool) -> None:
    """On macOS, ensure libomp is installed (required by XGBoost)."""
    brew = shutil.which("brew")
    if not brew:
        log.warning(
            "Homebrew not found. If XGBoost import fails later, "
            "install libomp manually or use conda-forge."
        )
        return

    r = _run(
        [brew, "list", "--versions", "libomp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if r.returncode == 0:
        log.info("libomp already installed via Homebrew.")
        return

    if not noninteractive:
        ans = input("  Install libomp via Homebrew? [Y/n]: ").strip().lower() or "y"
        if ans not in ("y", "yes"):
            log.info("Skipping libomp. Install manually if XGBoost import fails.")
            return

    log.info("Installing libomp via Homebrew ...")
    r = _run([brew, "install", "libomp"])
    if r.returncode == 0:
        log.info("libomp installed.")
    else:
        log.error("brew install libomp failed. Run 'brew install libomp' manually.")


def windows_checks() -> None:
    """Log helpful Windows-specific information."""
    cl = shutil.which("cl")
    if cl:
        log.info("MSVC toolchain detected: %s", cl)
    else:
        log.info("MSVC (cl.exe) not on PATH — not needed for pre-built wheels.")


# ── Step 3 — Create venv & sync dependencies ────────────────────────────────
def setup_environment(uv: str, dev: bool) -> None:
    """Create .venv via uv and sync all deps from pyproject.toml."""
    # 3a — virtual environment
    if os.path.isdir(VENV_DIR):
        log.info("Virtual environment already exists at .venv — reusing.")
    else:
        log.info("Creating virtual environment with Python %s ...", PYTHON_VERSION)
        r = _run([uv, "venv", VENV_DIR, "--python", PYTHON_VERSION])
        if r.returncode != 0:
            log.error(
                "Failed to create virtual environment (exit %d). Exiting.",
                r.returncode,
            )
            sys.exit(1)
        log.info("Virtual environment created at .venv")

    # 3b — sync dependencies from pyproject.toml
    sync_cmd = [uv, "sync"]
    if dev:
        sync_cmd.append("--all-extras")
        log.info("Syncing all dependencies + dev extras from pyproject.toml ...")
    else:
        log.info("Syncing dependencies from pyproject.toml ...")

    r = _run(sync_cmd, cwd=ROOT)
    if r.returncode != 0:
        log.error(
            "uv sync failed (exit %d). Check the output above for details.",
            r.returncode,
        )
        sys.exit(1)

    log.info("All dependencies synced successfully.")


# ── Finish ───────────────────────────────────────────────────────────────────
def print_next_steps(os_name: str) -> None:
    _banner("SETUP COMPLETE")
    log.info("")
    if os_name == "Windows":
        log.info("  Activate (PowerShell)  :  .venv\\Scripts\\Activate.ps1")
        log.info("  Activate (cmd.exe)     :  .venv\\Scripts\\activate.bat")
    else:
        log.info("  Activate               :  source .venv/bin/activate")
    log.info("")
    log.info("  Run pipeline           :  python scripts/run_pipeline.py")
    log.info("  Run dashboard          :  streamlit run app/main.py")
    log.info("  Run tests              :  pytest")
    log.info("")
    log.info(DIVIDER)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-platform project setup using uv",
    )
    ap.add_argument("--yes", action="store_true", help="Skip prompts (non-interactive)")
    ap.add_argument(
        "--dev",
        action="store_true",
        help="Include [dev] extras (pytest, ruff, jupyter, etc.)",
    )
    args = ap.parse_args()

    _configure_logging()
    _banner("ENVIRONMENT SETUP")

    # ── Detect platform ──────────────────────────────────────────────────
    os_name = detect_platform()

    total_steps = 3

    # ── Step 1: Ensure uv ────────────────────────────────────────────────
    _step(1, total_steps, "Checking uv installation")
    uv = ensure_uv(os_name, args.yes)

    # ── Step 2: Platform-specific pre-checks ─────────────────────────────
    _step(2, total_steps, "Platform-specific checks")
    if os_name == "Darwin":
        log.info("macOS detected — checking libomp (required by XGBoost) ...")
        ensure_libomp_mac(args.yes)
    elif os_name == "Windows":
        windows_checks()
    else:
        log.info("Linux detected — no additional setup required.")

    # ── Step 3: venv + sync ──────────────────────────────────────────────
    _step(3, total_steps, "Creating venv & syncing dependencies")
    setup_environment(uv, args.dev)

    # ── Done ─────────────────────────────────────────────────────────────
    print_next_steps(os_name)


if __name__ == "__main__":
    main()
