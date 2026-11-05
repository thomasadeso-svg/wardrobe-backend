"""Real Styligma FastAPI wiring checks, with no external network or AI requests.

Run only on an isolated test branch. No secrets, no deployment, no rembg models.
"""
import asyncio
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import httpx


BACKEND = Path(__file__).with_name('backend-main.py')


def load_app(*, enabled: bool, directory: str | None = None, railway_env: bool = False):
    env = {
        'STYLIGMA_DURABLE_JOBS_ENABLED': '1' if enabled else '',
        'STYLIGMA_JOB_STORE_DIR': directory or '',
        'RAILWAY_PROJECT_ID': 'test-project' if railway_env else '',
        'RAILWAY_ENVIRONMENT_ID': '',
        'RAILWAY_VOLUME_MOUNT_PATH': '',
        'ANTHROPIC_API_KEY': '',
        'REMOVE_BG_API_KEY': '',
    }
    with patch.dict(os.environ, env):
        spec = importlib.util.spec_from_file_location('styligma_combined_test_' + uuid.uuid4().hex, BACKEND)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class RealCombinedWiringTests(unittest.TestCase):
    def test_off_by_default_legacy_routes_remain(self):
        m = load_app(enabled=False)
        paths = [r.path for r in m.app.router.routes]
        for path in ('/process-clothing', '/process-clothing-status/{job_id}', '/scan-video', '/scan-status/{job_id}'):
            self.assertEqual(paths.count(path), 1)
        self.assertNotIn('/job-by-request/{request_id}', paths)
        self.assertFalse(hasattr(m.app.state, 'styligma_durable_api'))

    def test_opt_in_fails_closed_on_railway_without_persistent_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'Railway volume'):
                load_app(enabled=True, directory=tmp, railway_env=True)

    def test_opt_in_replaces_only_job_routes_and_uses_no_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = load_app(enabled=True, directory=tmp)
            paths = [r.path for r in m.app.router.routes]
            for path in ('/process-clothing', '/process-clothing-status/{job_id}', '/scan-video', '/scan-status/{job_id}'):
                self.assertEqual(paths.count(path), 1)
            self.assertEqual(paths.count('/job-by-request/{request_id}'), 1)
            for path in ('/', '/privacy', '/terms', '/generate-outfit', '/match-item', '/scan-photo'):
                self.assertIn(path, paths)
            import video_scan
            self.assertFalse(video_scan.MULTI_ITEM_LIVE_ENABLED)
            self.assertIsNone(m.client)

    def test_real_asgi_photo_and_video_contracts_without_ai_calls(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as tmp:
                m = load_app(enabled=True, directory=tmp)
                api = m.app.state.styligma_durable_api
                async def fake_photo(data, content_type, metadata):
                    self.assertEqual(data, b'mock-jacket')
                    return {'analysis': {'rejected': False, 'category': 'outerwear'}, 'image': 'data:image/png;base64,c2FtcGxl'}
                async def fake_video(data, content_type, metadata):
                    self.assertEqual(data, b'fake-video')
                    return {'items': [{'temp_id':'synthetic', 'image_base64': 'c2FtcGxl', 'category': 'top'}],
                            'frames_scanned': 1, 'duplicates_removed': 0, 'frame_issues': []}
                api.processors['photo'] = fake_photo
                api.processors['video'] = fake_video
                api.video_rate_check = lambda key: None
                await m.app.router.startup()
                try:
                    transport = httpx.ASGITransport(app=m.app)
                    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                        rid = str(uuid.uuid4())
                        response = await client.post('/process-clothing',
                            files={'file': ('jacket.jpg', b'mock-jacket', 'image/jpeg')},
                            headers={'X-Styligma-Request-ID': rid})
                        self.assertEqual(response.status_code, 200, response.text)
                        body = response.json()
                        self.assertEqual(body['status'], 'processing')
                        jid = body['job_id']
                        status = await client.get('/process-clothing-status/' + jid)
                        self.assertEqual(status.status_code, 200, status.text)
                        self.assertEqual(status.json()['status'], 'done')
                        self.assertEqual(status.json()['result']['analysis']['category'], 'outerwear')
                        recovered = await client.get('/job-by-request/' + rid)
                        self.assertEqual(recovered.status_code, 200)
                        self.assertEqual(recovered.json()['job_id'], jid)
                        duplicate = await client.post('/process-clothing',
                            files={'file': ('jacket.jpg', b'mock-jacket', 'image/jpeg')},
                            headers={'X-Styligma-Request-ID': rid})
                        self.assertEqual(duplicate.status_code, 200, duplicate.text)
                        self.assertEqual(duplicate.json()['job_id'], jid)
                        vid = str(uuid.uuid4())
                        video = await client.post('/scan-video',
                            files={'file':('scan.mp4', b'fake-video', 'video/mp4')},
                            headers={'X-Styligma-Request-ID': vid})
                        self.assertEqual(video.status_code, 202, video.text)
                        video_jid = video.json()['job_id']
                        self.assertIn('-', video_jid)
                        scanned = await client.get('/scan-status/' + video_jid)
                        self.assertEqual(scanned.status_code, 200, scanned.text)
                        self.assertEqual(scanned.json()['status'], 'done')
                        self.assertEqual(len(scanned.json()['items']), 1)
                        self.assertEqual((await client.get('/job-by-request/' + vid)).json()['job_id'], video_jid)
                finally:
                    await m.app.router.shutdown()
        asyncio.run(exercise())


if __name__ == '__main__':
    unittest.main()