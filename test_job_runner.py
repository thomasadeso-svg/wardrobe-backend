import asyncio
import tempfile
import unittest
from durable_jobs import DurableJobStore
from job_runner import run_job, run_ready

class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store=DurableJobStore(self.tmp.name)
    async def test_photo_processor_roundtrip(self):
        async def photo(data,mime,metadata):
            self.assertEqual(data,b'fake-photo')
            return {'analysis':{'category':'top'},'image':'data:fake'}
        job=self.store.submit('photo',b'fake-photo','image/jpeg')
        self.assertTrue(await run_job(self.store,job,{'photo':photo},renew_every=0.01))
        self.assertEqual(self.store.status(job)['result']['image'],'data:fake')
    async def test_video_processor_roundtrip(self):
        def video(data,mime,metadata):return {'items':[],'frames_scanned':2,'duplicates_removed':0}
        job=self.store.submit('video',b'fake-video','video/mp4')
        self.assertEqual(await run_ready(self.store,{'video':video}),1)
        self.assertEqual(self.store.status(job)['result']['frames_scanned'],2)
    async def test_worker_failure_becomes_safe_error(self):
        def failing(data,mime,metadata):raise ValueError('raw secret')
        job=self.store.submit('photo',b'fake-photo','image/jpeg')
        self.assertFalse(await run_job(self.store,job,{'photo':failing}))
        self.assertEqual(self.store.status(job)['error'],'processing_failed')
    async def test_concurrent_runs_only_claim_once(self):
        calls=[]
        async def slow(data,mime,metadata):
            calls.append('called');await asyncio.sleep(0.01);return {'ok':True}
        job=self.store.submit('photo',b'fake-photo','image/jpeg')
        outcome=await asyncio.gather(run_job(self.store,job,{'photo':slow}),run_job(self.store,job,{'photo':slow}))
        self.assertEqual(outcome.count(True),1)
        self.assertEqual(len(calls),1)
    async def test_missing_processor_does_not_stick(self):
        job=self.store.submit('video',b'fake-video','video/mp4')
        self.assertFalse(await run_job(self.store,job,{}))
        self.assertEqual(self.store.status(job)['status'],'error')

if __name__=='__main__':unittest.main()
