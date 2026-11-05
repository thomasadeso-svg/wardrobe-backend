"""Crash orphan cleanup and retention tests (no paid providers)."""
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from durable_jobs import DurableJobStore
from test_durable_jobs import FakeClock

class OrphanCleanupTests(unittest.TestCase):
    def test_orphan_upload_garbage_collected_only_after_24h(self):
        with tempfile.TemporaryDirectory() as root:
            store=DurableJobStore(root)
            orphan=Path(root)/('a'*32+'.input')
            orphan.write_bytes(b'fake-photo')
            store.cleanup()
            self.assertTrue(orphan.exists())
            old=time.time()-25*3600
            os.utime(orphan,(old,old))
            store.cleanup()
            self.assertFalse(orphan.exists())

    def test_active_job_not_removed_when_its_input_is_old(self):
        with tempfile.TemporaryDirectory() as root:
            store=DurableJobStore(root)
            job=store.submit('video',b'fake-video','video/mp4')
            f=Path(root)/(job+'.input')
            old=time.time()-25*3600
            os.utime(f,(old,old))
            store.cleanup()
            self.assertTrue(f.exists())

    def test_pseudonymous_rate_logs_removed_on_wake_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            clock=FakeClock()
            store=DurableJobStore(root,clock=clock)
            store.enforce_video_rate('198.51.100.12')
            clock.advance(86401)
            store.cleanup()
            with sqlite3.connect(Path(root)/'jobs.sqlite3') as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM scan_rate_events').fetchone()[0],0)

if __name__=='__main__':unittest.main()
