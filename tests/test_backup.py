import gzip
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from bigbrain import Brain, backup


class BackupTests(unittest.TestCase):
    def test_snapshot_restore_roundtrip_while_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "brain.db"
            brain = Brain(db)
            brain.learn("note", "Keep me", "The brain remembers between runs and across backups.")
            brain.set_state("trader:main", {"cash": 123.0})
            snap = backup.snapshot(db, Path(tmp) / "backups", keep=2)
            self.assertTrue(snap.exists())
            brain.learn("note", "After snapshot", "This one should vanish on restore.")
            brain.close()
            backup.restore(snap, db)
            restored = Brain(db)
            self.assertEqual(len(restored.find("Keep me")), 1)
            self.assertEqual(restored.find("After snapshot"), [])
            self.assertEqual(restored.get_state("trader:main")["cash"], 123.0)
            self.assertTrue((Path(tmp) / "brain.db.before-restore").exists())
            restored.close()

    def test_rotation_keeps_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "brain.db"
            Brain(db).close()
            out = Path(tmp) / "backups"
            for i in range(4):
                p = backup.snapshot(db, out, keep=2)
                os.rename(p, out / f"brain-2026010{i}-0000.db.gz")  # distinct names within the same minute
            backup.snapshot(db, out, keep=2)
            self.assertEqual(len(list(out.glob("brain-*.db.gz"))), 2)
            self.assertEqual(backup.latest(out), sorted(out.glob("brain-*.db.gz"))[-1])

    def test_push_and_fetch_against_a_local_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote.git"
            subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
            repo = Path(tmp) / "repo"
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(remote)], check=True)
            db = Path(tmp) / "brain.db"
            b = Brain(db); b.learn("note", "Cloud copy", "This brain went to the remote and came back."); b.close()
            snap = backup.snapshot(db, Path(tmp) / "backups")
            sha = backup.push(snap, repo, note="test")
            self.assertEqual(len(sha), 40)
            heads = subprocess.run(["git", "-C", str(remote), "branch", "--list"], capture_output=True, text=True).stdout
            self.assertIn("brain-backup", heads)
            # a second push replaces the single commit rather than stacking
            sha2 = backup.push(snap, repo)
            count = subprocess.run(["git", "-C", str(remote), "rev-list", "--count", "brain-backup"], capture_output=True, text=True).stdout.strip()
            self.assertEqual(count, "1")
            self.assertNotEqual(sha, sha2)
            fetched = backup.fetch(repo, Path(tmp) / "fetched")
            with gzip.open(fetched) as f:
                self.assertTrue(f.read(16).startswith(b"SQLite format 3"))
            backup.restore(fetched, Path(tmp) / "restored.db")
            self.assertEqual(len(Brain(Path(tmp) / "restored.db").find("Cloud copy")), 1)


if __name__ == "__main__":
    unittest.main()
