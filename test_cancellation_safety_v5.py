"""Additional offline invariants. No provider SDK import or paid calls."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from fastapi import HTTPException
from fastapi.testclient import TestClient
from durable_jobs import DurableJobStore
from job_runner import run_job
from styligma_durable_integration import install_durable_jobs
from test_real_processing_integration import FakeSources, make_legacy_app
from test_durable_jobs import FakeClock

class CancelledProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_graceful_shutdown_drains_started_job_before_status_changes(self):
        with tempfile.TemporaryDirectory() as path:
            store=DurableJobStore(path)
            job=store.submit('photo',b'fake-photo','image/jpeg')
            started=asyncio.Event()
            release=asyncio.Event()
            async def fake_paid_processing(data, mime, metadata):
                started.set()
                await release.wait()
                return {'analysis': {'category':'top'}}
            task=asyncio.create_task(run_job(store,job,{'photo':fake_paid_processing}, renew_every=.01))
            await asyncio.wait_for(started.wait(),1)
            self.assertFalse((Path(path)/f'{job}.input').exists())
            task.cancel()
            # A cancelled HTTP/background runner MUST NOT expose retry_required
            # while the old potentially paid processor is still executing.
            await asyncio.sleep(.04)
            self.assertFalse(task.done())
            self.assertEqual(store.status(job)['status'],'processing')
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task,1)
            self.assertEqual(store.status(job)['status'],'done')
            self.assertEqual(store.status(job)['result']['analysis']['category'],'top')
            self.assertEqual(store.ready_ids(),[])

class PersistentAdapterTests(unittest.TestCase):
    def test_five_video_scans_max_across_app_restarts(self):
        with tempfile.TemporaryDirectory() as path:
            sources=FakeSources()
            app=make_legacy_app()
            install_durable_jobs(app,sources.backend,sources.video,directory=path,environ={})
            with TestClient(app) as client:
                for i in range(3):
                    response=client.post('/scan-video',files={'file':('v.mp4',b'fake-video-bytes','video/mp4')})
                    self.assertEqual(response.status_code,202)
                    self.assertEqual(client.get('/scan-status/'+response.json()['job_id']).json()['status'],'done')
            new_app=make_legacy_app()
            install_durable_jobs(new_app,sources.backend,sources.video,directory=path,environ={})
            with TestClient(new_app) as client:
                for i in range(2):
                    response=client.post('/scan-video',files={'file':('v.mp4',b'fake-video-bytes','video/mp4')})
                    self.assertEqual(response.status_code,202)
                    self.assertEqual(client.get('/scan-status/'+response.json()['job_id']).json()['status'],'done')
                sixth=client.post('/scan-video',files={'file':('v.mp4',b'fake-video-bytes','video/mp4')})
                self.assertEqual(sixth.status_code,429)
            self.assertEqual(sources.classification_calls.count('video'),5)

if __name__=='__main__':unittest.main()
