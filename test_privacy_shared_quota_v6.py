"""Offline privacy and combined endpoint quota tests (no Anthropic, no network)."""
import asyncio
import tempfile
import types
import unittest
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from fastapi import FastAPI

from durable_jobs import DurableJobStore
from test_durable_jobs import FakeClock
from test_real_processing_integration import FakeSources, make_legacy_app
from styligma_durable_integration import install_durable_jobs
from serverless_api_candidate import ServerlessJobApi


class PrivacyAfterSleepTests(unittest.TestCase):
    def test_sqlite_metadata_permissions_are_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            DurableJobStore(d)
            self.assertEqual((Path(d)/'jobs.sqlite3').stat().st_mode & 0o777, 0o600)

    def test_crash_orphan_deleted_after_one_hour_of_live_maintenance(self):
        import os
        import time
        with tempfile.TemporaryDirectory() as d:
            store=DurableJobStore(d)
            orphan=Path(d)/('a'*32+'.input')
            orphan.write_bytes(b'synthetic-orphan-photo')
            old=time.time()-3700
            os.utime(orphan, (old,old))
            store.cleanup()
            self.assertFalse(orphan.exists())

    def test_expired_photo_bytes_deleted_when_poller_wakes_service(self):
        clock=FakeClock()
        with tempfile.TemporaryDirectory() as d:
            store=DurableJobStore(d,clock=clock)
            jid=store.submit('photo',b'synthetic-only','image/jpeg')
            token,*_=store.claim(jid)
            store.complete(jid,token,{'image':'synthetic-result'})
            result=Path(d)/f'{jid}.result'
            self.assertTrue(result.exists())
            clock.advance(901)
            # Immediately after logical expiry, a poll must remove bytes.
            self.assertIsNone(store.status(jid))
            self.assertFalse(result.exists())
            self.assertEqual(store.cleanup(),0)

    def test_next_boot_prunes_expired_result_that_was_sleeping(self):
        clock=FakeClock()
        with tempfile.TemporaryDirectory() as d:
            old=DurableJobStore(d,clock=clock)
            jid=old.submit('photo',b'fake-photo','image/jpeg')
            token,*_=old.claim(jid)
            old.complete(jid,token,{'image':'fake-image'})
            file=Path(d)/f'{jid}.result'
            clock.advance(3000)
            # No Python task can delete files while Railway has stopped it.
            self.assertTrue(file.exists())
            source=FakeSources()
            app=make_legacy_app()
            install_durable_jobs(app,source.backend,source.video,directory=d,
                                 environ={},clock=clock)
            self.assertFalse(file.exists())
            with TestClient(app) as client:
                self.assertEqual(client.get('/process-clothing-status/'+jid).status_code,404)

    def test_while_awake_maintenance_prunes_without_http_request(self):
        async def scenario():
            clock=FakeClock()
            with tempfile.TemporaryDirectory() as d:
                store=DurableJobStore(d,clock=clock)
                async def fake(*_args):return {'ok':True}
                api=ServerlessJobApi(store,photo_processor=fake,video_processor=fake)
                jid=store.submit('photo',b'x','image/jpeg')
                token,*_=store.claim(jid)
                store.complete(jid,token,{'ok':True})
                file=Path(d)/f'{jid}.result'
                self.assertTrue(file.exists())
                clock.advance(1000)
                task=asyncio.create_task(api.maintenance_loop(interval_seconds=.01))
                try:
                    for _ in range(100):
                        if not file.exists():break
                        await asyncio.sleep(.01)
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):await task
                self.assertFalse(file.exists())
        asyncio.run(scenario())


class SharedQuotaV6Tests(unittest.TestCase):
    @staticmethod
    def old_app(video_source):
        app=make_legacy_app()
        @app.post('/scan-photo')
        async def scan_photo(request:Request):
            # Models how the actual video_scan.scan_photo handler calls the
            # module-level _check_rate_limit before its paid work.
            video_source._check_rate_limit(request.client.host)
            return {'scanned':'synthetic'}
        return app

    def test_video_photo_share_five_scans_across_restart(self):
        with tempfile.TemporaryDirectory() as d:
            source=FakeSources()
            app=self.old_app(source.video)
            install_durable_jobs(app,source.backend,source.video,directory=d,environ={})
            with TestClient(app) as client:
                for _ in range(2):
                    r=client.post('/scan-photo')
                    self.assertEqual(r.status_code,200)
                for _ in range(2):
                    r=client.post('/scan-video',files={'file':('v.mp4',b'fake-video-bytes','video/mp4')})
                    self.assertEqual(r.status_code,202)
                    self.assertEqual(client.get('/scan-status/'+r.json()['job_id']).json()['status'],'done')
            restarted=self.old_app(source.video)
            install_durable_jobs(restarted,source.backend,source.video,directory=d,environ={})
            with TestClient(restarted) as client:
                self.assertEqual(client.post('/scan-photo').status_code,200)
                self.assertEqual(client.post('/scan-photo').status_code,429)
                self.assertEqual(client.post('/scan-video',files={
                    'file':('v.mp4',b'fake-video-bytes','video/mp4')}).status_code,429)
            self.assertEqual(len(source.classification_calls),2)

    def test_missing_original_scan_photo_limiter_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            s=FakeSources()
            del s.video._check_rate_limit
            app=self.old_app(s.video)
            with self.assertRaisesRegex(RuntimeError,'Missing legacy'):
                install_durable_jobs(app,s.backend,s.video,directory=d,environ={})
            self.assertEqual(len([r for r in app.routes if r.path=='/process-clothing']),1)


if __name__=='__main__':unittest.main()
