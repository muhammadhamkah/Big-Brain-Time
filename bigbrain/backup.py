"""Backups of the brain: consistent snapshots, local rotation, and a copy on GitHub.

The brain is one SQLite file that changes every candle, so it is never
committed to the main branch. Instead ``snapshot`` uses SQLite's online
backup API to take a consistent copy while the trader keeps writing, gzips
it into a dated file, and keeps the last N locally. ``push`` puts the latest
snapshot on a dedicated ``brain-backup`` branch as a single commit, force
pushed, so the repository only ever holds one copy and stays small.
``restore`` brings a snapshot back.
"""

from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BRANCH = "brain-backup"
SNAPSHOT_NAME = "brain.db.gz"
GITHUB_FILE_LIMIT = 95 * 1024 * 1024  # GitHub rejects files over 100 MB


def snapshot(db_path: str | Path, out_dir: str | Path, keep: int = 14) -> Path:
    """Consistent gzipped copy of the database, dated; prunes old copies beyond ``keep``."""
    db_path, out_dir = Path(db_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    target = out_dir / f"brain-{stamp}.db.gz"
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir=out_dir) as tmp:
        tmp_path = Path(tmp.name)
    try:
        src = sqlite3.connect(str(db_path), timeout=30)
        dst = sqlite3.connect(str(tmp_path))
        with dst:
            src.backup(dst)  # online backup: safe while the trader is writing
        src.close(); dst.close()
        with open(tmp_path, "rb") as fin, gzip.open(target, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout)
    finally:
        tmp_path.unlink(missing_ok=True)
    snaps = sorted(out_dir.glob("brain-*.db.gz"))
    for old in snaps[:-keep] if keep > 0 else []:
        old.unlink()
    return target


def latest(out_dir: str | Path) -> Path | None:
    snaps = sorted(Path(out_dir).glob("brain-*.db.gz"))
    return snaps[-1] if snaps else None


def restore(snapshot_path: str | Path, db_path: str | Path) -> Path:
    """Replace the database with a snapshot. The current file is kept as brain.db.before-restore."""
    snapshot_path, db_path = Path(snapshot_path), Path(db_path)
    if db_path.exists():
        shutil.copy2(db_path, db_path.with_name(db_path.name + ".before-restore"))
    for suffix in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            side.unlink()
    with gzip.open(snapshot_path, "rb") as fin, open(db_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return db_path


def _git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def push(snapshot_path: str | Path, repo_dir: str | Path, remote: str = "origin", branch: str = BRANCH, note: str = "") -> str:
    """Publish one snapshot as the only commit on ``branch`` (force pushed). Returns the commit hash."""
    snapshot_path, repo_dir = Path(snapshot_path), Path(repo_dir)
    size = snapshot_path.stat().st_size
    if size > GITHUB_FILE_LIMIT:
        raise RuntimeError(f"snapshot is {size / 1e6:.0f} MB, over GitHub's file limit; keep local backups instead")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        _git("init", "-q", str(work), cwd=repo_dir)
        url = _git("remote", "get-url", remote, cwd=repo_dir).stdout.strip()
        _git("remote", "add", "origin", url, cwd=work)
        _git("checkout", "-q", "--orphan", branch, cwd=work)
        shutil.copy2(snapshot_path, work / SNAPSHOT_NAME)
        (work / "README.md").write_text(
            f"# Brain backup\n\nLatest snapshot of `.brain/brain.db` for Big Brain Time, taken {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC "
            f"from `{snapshot_path.name}` ({size / 1e6:.1f} MB gzipped).{(' ' + note) if note else ''}\n\n"
            "Restore with `bigbrain restore --from-github`. This branch always holds exactly one commit.\n"
        )
        # commit identity: reuse the main repo's, or a neutral one
        name = _git("config", "user.name", cwd=repo_dir, check=False).stdout.strip() or "bigbrain"
        email = _git("config", "user.email", cwd=repo_dir, check=False).stdout.strip() or "bigbrain@localhost"
        _git("-c", f"user.name={name}", "-c", f"user.email={email}", "add", "-A", cwd=work)
        _git("-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "-m", f"Brain snapshot {snapshot_path.name}", cwd=work)
        _git("push", "-q", "--force", "origin", branch, cwd=work)
        return _git("rev-parse", "HEAD", cwd=work).stdout.strip()


def fetch(repo_dir: str | Path, out_dir: str | Path, remote: str = "origin", branch: str = BRANCH) -> Path:
    """Download the snapshot on the backup branch into ``out_dir``."""
    repo_dir, out_dir = Path(repo_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _git("fetch", "-q", remote, branch, cwd=repo_dir)
    blob = subprocess.run(["git", "show", f"{remote}/{branch}:{SNAPSHOT_NAME}"], cwd=repo_dir, capture_output=True, check=True).stdout
    target = out_dir / f"brain-github-{datetime.now(timezone.utc):%Y%m%d-%H%M}.db.gz"
    target.write_bytes(blob)
    return target


def run_periodic(db_path: str | Path, out_dir: str | Path, repo_dir: str | Path, every_hours: float, keep: int, do_push: bool, log=print, sleep=time.sleep, once: bool = False) -> None:
    while True:
        try:
            snap = snapshot(db_path, out_dir, keep)
            line = f"[{datetime.now(timezone.utc):%H:%M:%S}] snapshot {snap.name} ({snap.stat().st_size / 1e6:.1f} MB)"
            if do_push:
                sha = push(snap, repo_dir)
                line += f", pushed to {BRANCH} ({sha[:7]})"
            log(line)
        except Exception as exc:
            log(f"backup error: {exc}")
        if once:
            return
        sleep(every_hours * 3600)
