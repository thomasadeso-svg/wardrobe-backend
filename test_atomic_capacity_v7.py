"""Adversarial v7 tests: no network, paid provider, or live deployment."""
import concurrent.futures
import tempfile
import threading
import multiprocessing
import unittest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from durable_jobs import DurableJobStore, JobCapacityExceeded
from serverless_api_candidate import ServerlessJobApi


def submit_from_process(directory):
    # Real subprocesses have independent SQLite connections and interpreter state.
    store = DurableJobStore(directory)
    try:
        return store.submit('video', b'synthetic-video', 'video/mp4', max_active=4)
    except JobCapacityExceeded:
        return None


class AtomicVideoCapacityTests(unittest.TestCase):
    def test_real_multiple_processes_share_atomic_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=ctx) as pool:
                ids = list(pool.map(submit_from_process, [tmp] * 12))
            self.assertEqual(len([job for job in ids if job]), 4)
            self.assertEqual(DurableJobStore(tmp).count_active('video'), 4)

    def test_concurrent_clients_cannot_exceed_four_active_video_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Independent store objects model separate request handlers/processes.
            clients = [DurableJobStore(tmp) for _ in range(20)]
            barrier = threading.Barrier(len(clients))
            def submit(n):
                barrier.wait(timeout=15)
                try:
                    return clients[n].submit('video', b'video-payload', 'video/mp4', max_active=4)
                except JobCapacityExceeded:
                    return None
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
                ids = list(pool.map(submit, range(len(clients))))
            admitted = [job for job in ids if job is not None]
            self.assertEqual(len(admitted), 4)
            self.assertEqual(clients[0].count_active('video'), 4)
            # Rejected uploads leave neither DB rows nor photo/video files behind.
            self.assertEqual(len(list(Path(tmp).glob('*.input'))), 4)

    def test_completed_slot_allows_one_new_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DurableJobStore(tmp)
            active = [store.submit('video', b'v', 'video/mp4', max_active=4) for _ in range(4)]
            with self.assertRaises(JobCapacityExceeded):
                store.submit('video', b'x', 'video/mp4', max_active=4)
            token, *_ = store.claim(active[0])
            self.assertTrue(store.complete(active[0], token, {'items': []}))
            self.assertTrue(store.submit('video', b'fresh', 'video/mp4', max_active=4))

    def test_429_when_precheck_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DurableJobStore(tmp)
            for _ in range(4):
                store.submit('video', b'x', 'video/mp4', max_active=4)
            async def fake(*_):
                self.fail('A rejected video must never reach the processor')
            api = ServerlessJobApi(store, photo_processor=fake, video_processor=fake)
            # Simulate other concurrent requests passing the optimistic precheck.
            store.count_active = lambda _kind: 0
            app = FastAPI()
            app.include_router(api.router)
            with TestClient(app) as client:
                response = client.post('/scan-video', files={'file': ('pan.mp4', b'v', 'video/mp4')})
                self.assertEqual(response.status_code, 429)
                self.assertIn('busy', response.json()['detail'].lower())

if __name__ == '__main__':
    unittest.main()
