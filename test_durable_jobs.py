"""Offline tests; no Anthropic, image inference, Railway or network calls."""
import os
import tempfile
import unittest
from pathlib import Path
from durable_jobs import DurableJobStore


class FakeClock:
    def __init__(self, start=1000.0):self.now=start
    def __call__(self):return self.now
    def advance(self, secs):self.now+=secs


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock=FakeClock()
        self.a=DurableJobStore(self.tmp.name,clock=self.clock)
    def photo(self):return self.a.submit('photo',b'fake-photo','image/jpeg')
    def test_submit_and_poll(self):
        job=self.photo();self.assertEqual(self.a.status(job)['status'],'processing')
        self.assertEqual((self.a.root/f'{job}.input').read_bytes(),b'fake-photo')
    def test_photo_complete_persists_across_restart(self):
        job=self.photo();token,data,mime,meta=self.a.claim(job)
        self.assertEqual((data,mime),(b'fake-photo','image/jpeg'))
        self.assertTrue(self.a.complete(job,token,{'image':'fake','analysis':{'category':'top'}}))
        self.assertFalse((self.a.root/f'{job}.input').exists())
        b=DurableJobStore(self.tmp.name,clock=self.clock)
        self.assertEqual(b.status(job)['result']['analysis']['category'],'top')
    def test_video_complete_persists_across_restart(self):
        job=self.a.submit('video',b'fake-video','video/mp4')
        token,*_=self.a.claim(job)
        self.assertTrue(self.a.complete(job,token,{'items':[{'category':'top'}]}))
        b=DurableJobStore(self.tmp.name,clock=self.clock)
        self.assertEqual(b.status(job)['result']['items'][0]['category'],'top')
    def test_interrupted_paid_job_not_automatically_replayed(self):
        job=self.photo();old,*_=self.a.claim(job,lease_seconds=60)
        self.assertNotIn(job,self.a.ready_ids())
        self.clock.advance(61)
        b=DurableJobStore(self.tmp.name,clock=self.clock)
        self.assertNotIn(job,b.ready_ids())
        self.assertIsNone(b.claim(job))
        self.assertEqual(b.status(job)['error'],'retry_required')
        self.assertFalse(self.a.complete(job,old,{'stale':True}))

    def test_running_job_not_double_claimed(self):
        job=self.photo();token,*_=self.a.claim(job)
        b=DurableJobStore(self.tmp.name,clock=self.clock)
        self.assertIsNone(b.claim(job))
        self.assertTrue(self.a.renew(job,token))
    def test_failure_redacted(self):
        job=self.photo();token,*_=self.a.claim(job)
        self.assertTrue(self.a.fail(job,token,'Bearer secret-token'))
        self.assertEqual(self.a.status(job)['error'],'processing_failed')
        self.assertFalse((self.a.root/f'{job}.input').exists())
    def test_cleanup_expired_photo(self):
        job=self.photo();token,*_=self.a.claim(job)
        self.a.complete(job,token,{'ok':True})
        self.clock.advance(901)
        self.assertIsNone(self.a.status(job))
        # An expiry poll eagerly purges files, not just hides the result.
        self.assertFalse((self.a.root/f'{job}.result').exists())
        self.assertEqual(self.a.cleanup(),0)
    def test_video_retains_one_hour(self):
        job=self.a.submit('video',b'fake-video','video/mp4')
        token,*_=self.a.claim(job)
        self.a.complete(job,token,{'items':[]})
        self.clock.advance(3599)
        self.assertIsNotNone(self.a.status(job))
        self.clock.advance(2)
        self.assertIsNone(self.a.status(job))
    def test_missing_input_does_not_hang(self):
        job=self.photo();(self.a.root/f'{job}.input').unlink()
        self.assertIsNone(self.a.claim(job))
        self.assertEqual(self.a.status(job)['error'],'missing_input')
    def test_expired_lease_converts_to_retry_required(self):
        job=self.photo();self.a.claim(job,lease_seconds=30)
        self.clock.advance(31)
        self.assertEqual(self.a.mark_interrupted(),1)
        self.assertEqual(self.a.status(job)['error'],'retry_required')
        self.assertFalse((self.a.root/f'{job}.input').exists())
        self.assertEqual(self.a.mark_interrupted(),0)

    def test_count_active(self):
        job=self.photo();self.assertEqual(self.a.count_active('photo'),1)
        token,*_=self.a.claim(job);self.a.complete(job,token,{'ok':True})
        self.assertEqual(self.a.count_active('photo'),0)

    def test_input_size_validation(self):
        with self.assertRaises(ValueError):self.a.submit('photo',b'','image/jpeg')
        with self.assertRaises(ValueError):self.a.submit('unknown',b'x','image/jpeg')
    def test_result_hidden_before_completion(self):
        job=self.photo();self.assertNotIn('result',self.a.status(job))
    def test_stale_job_id_safe(self):
        self.assertIsNone(self.a.status('../oops'))
    def test_no_extra_processes_or_services(self):
        self.assertFalse(hasattr(self.a,'http_client'))
    def test_expired_queued_photo_never_triggers_paid_worker(self):
        job = self.photo()
        self.assertIn(job, self.a.ready_ids())
        self.clock.advance(3601)
        self.assertNotIn(job, self.a.ready_ids())
        self.assertIsNone(self.a.claim(job))
        self.assertEqual(self.a.count_active('photo'), 0)

    def test_expired_video_does_not_consume_active_slot(self):
        job = self.a.submit('video', b'fake-video', 'video/mp4')
        self.assertEqual(self.a.count_active('video'), 1)
        self.clock.advance(6 * 3600 + 1)
        self.assertEqual(self.a.count_active('video'), 0)
        self.assertNotIn(job, self.a.ready_ids())

    def test_queued_input_does_not_persist_indefinitely(self):
        job=self.photo()
        self.clock.advance(3601)
        self.assertEqual(self.a.cleanup(),1)
        self.assertFalse((self.a.root/f'{job}.input').exists())


if __name__=='__main__':unittest.main()
