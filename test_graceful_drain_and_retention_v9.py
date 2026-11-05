"""New v9 tests: real concurrency, retention and cancellation; never call AI."""
import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from durable_jobs import DurableJobStore
from job_runner import run_job
from test_durable_jobs import FakeClock

class GracefulDrainV9(unittest.IsolatedAsyncioTestCase):
    async def test_cancelling_running_thread_keeps_status_processing_until_thread_exits(self):
        with tempfile.TemporaryDirectory() as d:
            store = DurableJobStore(d)
            jid = store.submit('photo',b'test-shirt','image/jpeg')
            begun, release, finished = threading.Event(), threading.Event(), threading.Event()
            calls = []
            def paid_mock(*args):
                calls.append(1)
                begun.set()
                release.wait(timeout=3)
                finished.set()
                return {'analysis':{'category':'top'},'image':'fake'}
            async def analyst(*args):
                return await asyncio.to_thread(paid_mock, *args)
            task = asyncio.create_task(run_job(store,jid,{'photo':analyst},renew_every=.01))
            self.assertTrue(await asyncio.to_thread(begun.wait,1))
            self.assertFalse((Path(d)/f'{jid}.input').exists())
            task.cancel()
            await asyncio.sleep(.04)
            self.assertEqual(store.status(jid)['status'],'processing')
            self.assertFalse(task.done())
            self.assertFalse(finished.is_set())
            self.assertIsNone(store.claim(jid))
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task,2)
            self.assertEqual(store.status(jid)['status'],'done')
            self.assertEqual(calls,[1])

    async def test_repeated_cancellation_does_not_expose_retry_early(self):
        with tempfile.TemporaryDirectory() as d:
            store=DurableJobStore(d)
            jid=store.submit('photo',b'fake','image/jpeg')
            start, release=asyncio.Event(),asyncio.Event()
            async def mock(*args):
                start.set()
                await release.wait()
                return {'ok':True}
            task=asyncio.create_task(run_job(store,jid,{'photo':mock},renew_every=.01))
            await start.wait()
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(.01)
                self.assertEqual(store.status(jid)['status'],'processing')
            release.set()
            with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,1)
            self.assertEqual(store.status(jid)['status'],'done')

    async def test_handler_throws_after_cancellation_then_user_may_retry(self):
        with tempfile.TemporaryDirectory() as d:
            store=DurableJobStore(d)
            jid=store.submit('photo',b'fake','image/jpeg')
            start, release=asyncio.Event(),asyncio.Event()
            async def failing(*args):
                start.set()
                await release.wait()
                raise ValueError('secret fake provider error')
            task=asyncio.create_task(run_job(store,jid,{'photo':failing},renew_every=.01))
            await start.wait()
            task.cancel()
            await asyncio.sleep(.02)
            self.assertEqual(store.status(jid)['status'],'processing')
            release.set()
            # Current runner treats a processor failure as an error, not a new call.
            self.assertFalse(await asyncio.wait_for(task,1))
            self.assertEqual(store.status(jid)['error'],'processing_failed')

class RetentionV9(unittest.TestCase):
    def test_queued_photo_15m_and_video_30m_not_six_hours(self):
        clk=FakeClock()
        with tempfile.TemporaryDirectory() as d:
            s=DurableJobStore(d, clock=clk)
            p=s.submit('photo',b'photo','image/jpeg')
            v=s.submit('video',b'video','video/mp4')
            clk.advance(15*60+1)
            self.assertIsNone(s.claim(p))
            self.assertFalse(p in s.ready_ids())
            self.assertIsNotNone(s.claim(v))
            self.assertFalse((Path(d)/f'{v}.input').exists())

    def test_queued_video_expires_after_30m(self):
        clk=FakeClock()
        with tempfile.TemporaryDirectory() as d:
            s=DurableJobStore(d, clock=clk)
            j=s.submit('video',b'video','video/mp4')
            clk.advance(1801)
            self.assertIsNone(s.claim(j))
            self.assertEqual(s.cleanup(),1)
            self.assertFalse((Path(d)/f'{j}.input').exists())

    def test_claim_removes_raw_bytes_and_clears_queued_deadline(self):
        clk=FakeClock()
        with tempfile.TemporaryDirectory() as d:
            s=DurableJobStore(d, clock=clk)
            j=s.submit('photo',b'fake-photo','image/jpeg')
            tok,data,*_=s.claim(j,lease_seconds=7200)
            self.assertEqual(data,b'fake-photo')
            self.assertFalse((Path(d)/f'{j}.input').exists())
            clk.advance(3600)
            self.assertEqual(s.status(j)['status'],'processing')
            self.assertTrue(s.complete(j,tok,{'ok':True}))
            self.assertEqual(s.status(j)['status'],'done')

if __name__=='__main__':unittest.main()
