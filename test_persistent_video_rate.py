"""Persistence tests use no user content or paid API calls."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from durable_jobs import RateLimitExceeded
from durable_jobs import DurableJobStore
from test_durable_jobs import FakeClock

class RateLimitTests(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.clock=FakeClock()
        self.store=DurableJobStore(self.dir.name,clock=self.clock)

    def test_video_cap_survives_cold_start(self):
        for _ in range(3):self.store.enforce_video_rate('198.51.100.4')
        fresh=DurableJobStore(self.dir.name,clock=self.clock)
        for _ in range(2):fresh.enforce_video_rate('198.51.100.4')
        with self.assertRaises(RateLimitExceeded) as e: fresh.enforce_video_rate('198.51.100.4')
        self.assertIn("Too many scans",str(e.exception))

    def test_other_network_gets_independent_budget(self):
        for _ in range(5):self.store.enforce_video_rate('198.51.100.4')
        self.store.enforce_video_rate('203.0.113.7')

    def test_window_expires(self):
        for _ in range(5):self.store.enforce_video_rate('198.51.100.4')
        self.clock.advance(86400+1)
        self.store.enforce_video_rate('198.51.100.4')

    def test_ip_not_stored_in_plaintext(self):
        network='198.51.100.4'
        self.store.enforce_video_rate(network)
        with sqlite3.connect(str(Path(self.dir.name)/'jobs.sqlite3')) as db:
            bucket=db.execute('SELECT bucket FROM scan_rate_events').fetchone()[0]
        self.assertNotIn(network, bucket)
        self.assertEqual(len(bucket),64)
        self.assertEqual((Path(self.dir.name)/'.video_rate_salt').stat().st_mode & 0o777,0o600)

    def test_rate_limit_shared_between_instances(self):
        sibling=DurableJobStore(self.dir.name,clock=self.clock)
        for i in range(5):
            (self.store if i%2 else sibling).enforce_video_rate('198.51.100.4')
        with self.assertRaises(RateLimitExceeded): self.store.enforce_video_rate('198.51.100.4')

if __name__=='__main__':unittest.main()
