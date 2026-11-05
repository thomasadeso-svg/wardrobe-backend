"""Offline seam tests grounded in current GitHub function names/route contracts.

All AI/video methods are deliberately fakes; no network or real models.
This test DOES NOT mean real app code or mobile clients have been verified.
"""
import asyncio
import tempfile
import threading
import types
import unittest
import uuid
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from durable_jobs import DurableJobStore
from styligma_durable_integration import enable_from_env, install_durable_jobs


def make_legacy_app():
    app = FastAPI()

    @app.post('/process-clothing')
    async def old_photo(): return {'old': 'photo'}

    @app.get('/process-clothing-status/{job_id}')
    async def old_status(job_id: str): return {'old': 'photo status'}

    @app.post('/scan-video', status_code=202)
    async def old_video(): return {'old': 'video'}

    @app.get('/scan-status/{job_id}')
    async def old_video_status(job_id: str): return {'old': 'video status'}

    @app.post('/generate-outfit')
    async def original_outfit(): return {'untouched': True}

    return app


class FakeSources:
    def __init__(self):
        self.calls = []
        async def analysis(data, mime):
            self.calls.append('actual_photo_analysis_function')
            return {'category': 'top', 'rejected': False}
        async def remove(data):
            self.calls.append('actual_background_removal_function')
            return 'data:image/png;base64,Zg=='
        self.backend = types.SimpleNamespace(_analyze_clothing_bytes=analysis,
                                             _remove_background_job_bytes=remove)
        self.lock = asyncio.Lock()
        self.classification_calls = []
        self.notifications = []
        def video_processor(path, **kwargs):
            self.classification_calls.append('video')
            assert Path(path).exists()
            assert Path(path).read_bytes() == b'fake-video-bytes'
            return ([{'temp_id': 'scan_0', 'category': 'top', 'image_base64': 'Zg=='}], 2, 1)
        async def notify(push, title, message, payload):
            self.notifications.append((push, title, payload))
        self.video = types.SimpleNamespace(
            _scan_lock=self.lock, _process_video_sync=video_processor,
            _check_rate_limit=lambda addr: None,
            _make_region_detector=lambda: None,
            MULTI_ITEM_LIVE_ENABLED=False,
            _send_push_notification=notify)


class RealProcessorSeamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sources = FakeSources()
        self.app = make_legacy_app()
        self.api = install_durable_jobs(self.app, self.sources.backend, self.sources.video,
                                        directory=self.temp.name, environ={})

    def test_existing_outfit_route_unchanged_and_four_duplicates_removed(self):
        from collections import Counter
        counts = Counter(x.path for x in self.app.router.routes)
        self.assertEqual(counts['/process-clothing'], 1)
        self.assertEqual(counts['/process-clothing-status/{job_id}'], 1)
        self.assertEqual(counts['/scan-video'], 1)
        self.assertEqual(counts['/scan-status/{job_id}'], 1)
        with TestClient(self.app) as client:
            self.assertEqual(client.post('/generate-outfit').json(), {'untouched': True})

    def test_real_photo_functions_are_called_with_original_response_shape(self):
        with TestClient(self.app) as client:
            reply = client.post('/process-clothing', files={'file': ('shirt.jpg', b'photo', 'image/jpeg')})
            self.assertEqual(reply.status_code, 200)
            job = reply.json()['job_id']
            self.assertEqual(len(job), 32)
            done = client.get('/process-clothing-status/' + job).json()
        self.assertEqual(done['status'], 'done')
        self.assertEqual(done['result']['analysis']['category'], 'top')
        self.assertTrue(done['result']['image'].startswith('data:image/png'))
        self.assertCountEqual(self.sources.calls, ['actual_photo_analysis_function',
                                                 'actual_background_removal_function'])

    def test_photo_result_persists_when_app_restarts(self):
        with TestClient(self.app) as client:
            job = client.post('/process-clothing',files={'file':('s.jpg',b'p','image/jpeg')}).json()['job_id']
            self.assertEqual(client.get('/process-clothing-status/'+job).json()['status'],'done')
        new_app = make_legacy_app()
        install_durable_jobs(new_app,self.sources.backend,self.sources.video,directory=self.temp.name,environ={})
        with TestClient(new_app) as client:
            self.assertEqual(client.get('/process-clothing-status/'+job).json()['status'],'done')
        # Restart/poll must not create a new paid analysis call.
        self.assertEqual(self.sources.calls.count('actual_photo_analysis_function'),1)

    def test_interrupted_photo_reports_retry_and_does_not_auto_charge(self):
        from test_durable_jobs import FakeClock
        clock=FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            store=DurableJobStore(tmp,clock=clock)
            jid=store.submit('photo',b'fake-photo','image/jpeg')
            token,*_=store.claim(jid,lease_seconds=30)
            clock.advance(31)
            # Simulate process termination and recreation, not re-running the job.
            reopened=DurableJobStore(tmp,clock=clock)
            reopened.mark_interrupted()
            self.assertEqual(reopened.status(jid)['error'],'retry_required')
            self.assertEqual(reopened.ready_ids(),[])
            self.assertFalse(reopened.complete(jid,token,{'analysis':{}}))
            self.assertEqual(self.sources.calls,[])

    def test_video_uses_actual_video_function_and_preserves_hyphenated_ids(self):
        with TestClient(self.app) as client:
            reply=client.post('/scan-video',files={'file':('v.mp4',b'fake-video-bytes','video/mp4')})
            self.assertEqual(reply.status_code,202)
            jid=reply.json()['job_id']
            self.assertEqual(str(uuid.UUID(jid)),jid)
            result=client.get('/scan-status/'+jid).json()
        self.assertEqual(result['status'],'done')
        self.assertEqual(result['frames_scanned'],2)
        self.assertEqual(result['duplicates_removed'],1)
        self.assertEqual(result['items'][0]['category'],'top')
        self.assertEqual(self.sources.classification_calls,['video'])

    def test_never_started_photo_upload_can_resume_once(self):
        # Accepted but not started before sleep: no paid attempt occurred.
        job=self.api.store.submit('photo',b'queued-photo','image/jpeg')
        self.assertEqual(self.sources.calls,[])
        resumed=asyncio.run(self.api.resume_queued())
        self.assertEqual(resumed,1)
        self.assertEqual(self.api.store.status(job)['status'],'done')
        self.assertEqual(self.sources.calls.count('actual_photo_analysis_function'),1)
        self.assertEqual(asyncio.run(self.api.resume_queued()),0)
        self.assertEqual(self.sources.calls.count('actual_photo_analysis_function'),1)

    def test_pending_photo_polls_dispatch_queued_job(self):
        job=self.api.store.submit('photo',b'queued-photo','image/jpeg')
        with TestClient(self.app) as client:
            reply=client.get('/process-clothing-status/'+job)
            self.assertEqual(reply.status_code,200)
            self.assertEqual(client.get('/process-clothing-status/'+job).json()['status'],'done')
        self.assertEqual(self.sources.calls.count('actual_photo_analysis_function'),1)

    def test_missing_storage_refuses_enablement_on_railway(self):
        app=make_legacy_app()
        with self.assertRaisesRegex(RuntimeError,'attached Railway volume'):
            install_durable_jobs(app,self.sources.backend,self.sources.video,
                                 directory=self.temp.name,
                                 environ={'RAILWAY_PROJECT_ID':'project-id'})
        with self.assertRaisesRegex(RuntimeError,'within Railway volume'):
            install_durable_jobs(app,self.sources.backend,self.sources.video,
                                 directory=self.temp.name,
                                 environ={'RAILWAY_PROJECT_ID':'project-id',
                                          'RAILWAY_VOLUME_MOUNT_PATH':'/srv/not-the-volume'})
        self.assertEqual(len([r for r in app.routes if r.path == '/process-clothing']),1)

    def test_disabled_flag_leaves_original_handlers(self):
        app=make_legacy_app()
        self.assertIsNone(enable_from_env(app,self.sources.backend,self.sources.video,
                                         environ={'STYLIGMA_DURABLE_JOBS_ENABLED':'0'}))
        with TestClient(app) as client:
            self.assertEqual(client.post('/process-clothing').json(),{'old':'photo'})

    def test_changed_api_contract_fails_before_removing_old_routes(self):
        app=FastAPI()
        @app.get('/scan-video')
        async def unusual(): return {}
        with self.assertRaisesRegex(RuntimeError,'endpoints changed'):
            install_durable_jobs(app,self.sources.backend,self.sources.video,directory=self.temp.name,environ={})
        self.assertEqual(len(app.routes),5)  # 4 FastAPI defaults + one fake

if __name__=='__main__': unittest.main()

class InterruptedHttpPhotoTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepted_upload_worker_cancel_drain_preserves_result(self):
        """Actual HTTP path with mock paid work; cancelled runner drains it."""
        import httpx
        from test_durable_jobs import FakeClock
        with tempfile.TemporaryDirectory() as tmp:
            entered=asyncio.Event()
            release=asyncio.Event()
            n_calls=[]
            async def slow_analyst(data,mime):
                n_calls.append('mock-ai-attempt')
                entered.set()
                await release.wait()
                return {'category':'top'}
            async def fake_bg(data):return 'data:fake'
            backend=types.SimpleNamespace(_analyze_clothing_bytes=slow_analyst,
                                          _remove_background_job_bytes=fake_bg)
            sources=FakeSources()
            app=make_legacy_app()
            clk=FakeClock()
            api=install_durable_jobs(app,backend,sources.video,directory=tmp,environ={},clock=clk)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url='http://local.test',timeout=3) as client:
                upload=asyncio.create_task(client.post('/process-clothing',
                    files={'file':('shirt.jpg',b'small-photo','image/jpeg')}))
                await asyncio.wait_for(entered.wait(),timeout=3)
                import sqlite3
                with sqlite3.connect(str(Path(tmp)/'jobs.sqlite3')) as con:
                    job_id=con.execute('SELECT job_id FROM jobs').fetchone()[0]
                self.assertEqual(api.store.status(job_id)['status'],'processing')
                self.assertFalse((Path(tmp)/f'{job_id}.input').exists())
                upload.cancel()
                await asyncio.sleep(.04)
                self.assertFalse(upload.done())
                self.assertEqual(api.store.status(job_id)['status'],'processing')
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(upload,3)
            clk.advance(91)
            cold=make_legacy_app()
            recovered=install_durable_jobs(cold,backend,sources.video,directory=tmp,environ={},clock=clk)
            await recovered.resume_queued()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=cold),
                                         base_url='http://local.test') as client:
                status=(await client.get('/process-clothing-status/'+job_id)).json()
            self.assertEqual(status['status'],'done')
            self.assertEqual(status['result']['analysis']['category'],'top')
            self.assertEqual(n_calls,['mock-ai-attempt'])
