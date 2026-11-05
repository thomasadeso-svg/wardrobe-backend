"""Real OS process termination and on-disk recovery, no paid providers.

This tests durability of the SQLite/file store, not Railway's real sleep behavior.
"""
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from durable_jobs import DurableJobStore


class ProcessKillTests(unittest.TestCase):
    def test_sigkill_claimed_job_never_autoretries_or_retains_input(self):
        child_code = r'''
import sys,time
from durable_jobs import DurableJobStore
store = DurableJobStore(sys.argv[1])
job = store.submit('photo',b'synthetic test bytes','image/jpeg')
store.claim(job,lease_seconds=30)
print(job,flush=True)
time.sleep(120)
'''
        with tempfile.TemporaryDirectory() as work:
            p = subprocess.Popen([sys.executable, '-u', '-c', child_code, work],
                                 cwd=str(Path(__file__).parent),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                job_id = p.stdout.readline().strip()
                self.assertEqual(len(job_id), 32)
                self.assertEqual(DurableJobStore(work).status(job_id)['status'], 'processing')
                # The real child process is terminated without any cleanup handler.
                p.kill()
                p.wait(timeout=5)
                # Move only this synthetic test lease back in time.
                with sqlite3.connect(str(Path(work) / 'jobs.sqlite3')) as conn:
                    conn.execute('UPDATE jobs SET lease_until=? WHERE job_id=?',
                                 (time.time() - 1, job_id))
                cold = DurableJobStore(work)
                self.assertEqual(cold.mark_interrupted(), 1)
                self.assertEqual(cold.status(job_id)['error'], 'retry_required')
                self.assertEqual(cold.ready_ids(), [])
                self.assertFalse((Path(work) / (job_id + '.input')).exists())
            finally:
                if p.poll() is None:
                    p.kill()
                p.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
