"""Contract tests with fake AI/video processor callbacks only."""
import tempfile
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from durable_jobs import DurableJobStore
from serverless_api_candidate import ServerlessJobApi

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store=DurableJobStore(self.tmp.name)
        self.calls=[]
        self.paid_calls=[]
        async def photo(data,mime,meta):
            self.calls.append('photo')
            return {'analysis':{'category':'top'},'image':'data:fake'}
        async def video(data,mime,meta):
            self.calls.append('video')
            return {'items':[{'temp_id':'a','category':'top','image_base64':'ZmFrZQ=='}],
                    'frames_scanned':3,'duplicates_removed':1}
        self.video=video
        self.photo=photo
        self.app=self.newapp()
        self.client=TestClient(self.app)
        self.addCleanup(self.client.close)

    def newapp(self):
        app=FastAPI()
        app.include_router(ServerlessJobApi(self.store,
            photo_processor=self.photo,video_processor=self.video).router)
        return app

    def test_photo_paths_preserve_contract(self):
        res=self.client.post('/process-clothing',files={'file':('shirt.jpg',b'fake','image/jpeg')})
        self.assertEqual(res.status_code,200)
        jid=res.json()['job_id']
        self.assertEqual(res.json()['status'],'processing')
        polled=self.client.get('/process-clothing-status/'+jid)
        self.assertEqual(polled.status_code,200)
        self.assertEqual(polled.json()['result']['analysis']['category'],'top')
        self.assertEqual(self.calls,['photo'])

    def test_video_paths_preserve_contract(self):
        res=self.client.post('/scan-video', files={'file':('pan.mp4',b'fake','video/mp4')})
        self.assertEqual(res.status_code,202)
        jid=res.json()['job_id']
        self.assertEqual(len(jid), 36)  # old Video Pan UUID format with hyphens
        self.assertEqual(jid.count('-'), 4)
        reply=self.client.get('/scan-status/'+jid)
        self.assertEqual(reply.status_code,200)
        self.assertEqual(reply.json()['status'],'done')
        self.assertEqual(reply.json()['frames_scanned'],3)
        self.assertEqual(reply.json()['duplicates_removed'],1)
        self.assertEqual(reply.json()['items'][0]['category'],'top')
        self.assertEqual(self.calls,['video'])

    def test_completed_result_after_app_restart(self):
        r=self.client.post('/process-clothing',files={'file':('s.jpg',b'fake','image/jpeg')})
        jid=r.json()['job_id']
        other=TestClient(self.newapp())
        try:
            self.assertEqual(other.get('/process-clothing-status/'+jid).json()['status'],'done')
        finally:
            other.close()

    def test_interrupted_video_returns_error_without_paid_replay(self):
        import uuid
        from test_durable_jobs import FakeClock
        # Use a clock that can simulate SIGKILL plus expiry without waiting.
        fake=FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            store=DurableJobStore(tmp, clock=fake)
            jid=store.submit('video', b'fake video', 'video/mp4')
            tok,*_=store.claim(jid, lease_seconds=30)
            fake.advance(31)
            # A cold-start adapter must report a retriable error, not process again.
            api=ServerlessJobApi(DurableJobStore(tmp, clock=fake),
                                 photo_processor=self.photo, video_processor=self.video)
            app=FastAPI(); app.include_router(api.router)
            with TestClient(app) as client:
                reply=client.get('/scan-status/'+str(uuid.UUID(hex=jid)))
                self.assertEqual(reply.status_code, 200)
                self.assertEqual(reply.json()['status'], 'error')
                self.assertEqual(reply.json()['error'], 'retry_required')
                self.assertFalse(store.complete(jid,tok,{'items':[]}))
                self.assertEqual(self.calls,[])

    def test_missing_video_results_404(self):
        self.assertEqual(self.client.get('/scan-status/unknown').status_code,404)

    def test_unsupported_video_type_rejected(self):
        res=self.client.post('/scan-video',files={'file':('test.mov',b'x','application/octet-stream')})
        self.assertEqual(res.status_code,400)
        self.assertEqual(self.calls,[])

    def test_photo_oversized_rejected_without_ai(self):
        res=self.client.post('/process-clothing',files={'file':('huge.jpg',b'x'*(20*1024*1024+1),'image/jpeg')})
        self.assertEqual(res.status_code,413)
        self.assertEqual(self.calls,[])

    def test_video_count_limit(self):
        jobs=[self.store.submit('video', b'video', 'video/mp4') for _ in range(4)]
        res=self.client.post('/scan-video',files={'file':('v.mp4',b'video','video/mp4')})
        self.assertEqual(res.status_code,429)
        self.assertEqual(self.calls,[])

if __name__=='__main__':unittest.main()
