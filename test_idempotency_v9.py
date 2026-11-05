"""Idempotency across lost HTTP responses, restarts, and competing submitters.

All inputs/outputs synthetic; no Anthropic or network requests.
"""
import asyncio
import concurrent.futures
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from durable_jobs import DurableJobStore, RequestConflict, RequestExpired
from serverless_api_candidate import ServerlessJobApi
from test_durable_jobs import FakeClock


class RequestIdTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store=DurableJobStore(self.tmp.name)
        self.calls=[]
        async def photo(*args):
            self.calls.append('photo')
            return {'analysis':{'category':'top'}}
        async def video(*args):
            self.calls.append('video')
            return {'items':[],'frames_scanned':1,'duplicates_removed':0}
        api=ServerlessJobApi(self.store,photo_processor=photo,
                             video_processor=video, video_rate_check=self.store.enforce_video_rate)
        self.app=FastAPI();self.app.include_router(api.router)
        self.client=TestClient(self.app)
        self.addCleanup(self.client.close)

    def submit_photo(self, req_id, body=b'a synthetic shirt'):
        return self.client.post('/process-clothing',
            files={'file':('shirt.jpg',body,'image/jpeg')},
            headers={'x-styligma-request-id':req_id})

    def submit_video(self, req_id, body=b'a synthetic video'):
        return self.client.post('/scan-video',
            files={'file':('closet.mp4',body,'video/mp4')},
            headers={'x-styligma-request-id':req_id})

    def test_lost_photo_response_recovers_same_job_and_no_paid_replay(self):
        req_id=str(uuid.uuid4())
        first=self.submit_photo(req_id)
        self.assertEqual(first.status_code,200)
        job=first.json()['job_id']
        # Pretend client lost the POST response but persisted its UUID.
        recovered=self.client.get('/job-by-request/'+req_id)
        self.assertEqual(recovered.status_code,200)
        self.assertEqual(recovered.json()['job_id'],job)
        self.assertEqual(recovered.json()['kind'],'photo')
        self.assertNotIn('result',recovered.json())
        self.assertEqual(self.submit_photo(req_id).json()['job_id'],job)
        self.assertEqual(self.calls,['photo'])
        restarted=DurableJobStore(self.tmp.name)
        self.assertEqual(restarted.find_request(req_id)['job_id'],job)

    def test_duplicate_video_does_not_consume_extra_quota(self):
        req_id=str(uuid.uuid4())
        a=self.submit_video(req_id)
        self.assertEqual(a.status_code,202)
        b=self.submit_video(req_id)
        self.assertEqual(b.status_code,202)
        self.assertEqual(a.json()['job_id'],b.json()['job_id'])
        self.assertEqual(self.calls,['video'])
        lookup=self.client.get('/job-by-request/'+req_id)
        self.assertEqual(lookup.json()['job_id'],a.json()['job_id'])
        # If duplicate consumed a slot, only three new IDs would be accepted.
        for _ in range(4):
            self.assertEqual(self.submit_video(str(uuid.uuid4())).status_code,202)
        self.assertEqual(self.submit_video(str(uuid.uuid4())).status_code,429)
        self.assertEqual(self.calls.count('video'),5)
        # Even at the quota limit, repeat retrieval of accepted request works.
        self.assertEqual(self.submit_video(req_id).status_code,202)

    def test_payload_mismatch_conflicts_instead_of_reusing_wrong_job(self):
        req_id=str(uuid.uuid4())
        self.assertEqual(self.submit_photo(req_id).status_code,200)
        self.assertEqual(self.submit_photo(req_id, b'OTHER synthetic shirt').status_code,409)
        self.assertEqual(self.submit_video(req_id).status_code,409)
        self.assertEqual(self.calls,['photo'])

    def test_invalid_id_rejected_without_ai_and_unknown_lookup_404(self):
        self.assertEqual(self.submit_photo('not-uuid').status_code,409)
        self.assertEqual(self.client.get('/job-by-request/not-uuid').status_code,404)
        self.assertEqual(self.calls,[])

    def test_expired_result_never_replays_same_id(self):
        clock=FakeClock()
        with tempfile.TemporaryDirectory() as d:
            s=DurableJobStore(d, clock=clock)
            key=str(uuid.uuid4())
            j=s.submit('photo',b'fake', 'image/jpeg',request_id=key)
            token,*_=s.claim(j)
            s.complete(j,token,{'ok':True})
            clock.advance(901)
            self.assertIsNone(s.status(j))
            self.assertEqual(s.find_request(key)['status'],'expired')
            with self.assertRaises(RequestExpired):
                s.submit('photo',b'fake','image/jpeg',request_id=key)

    def test_competing_submitters_share_job_id_and_only_one_input_blob(self):
        key=str(uuid.uuid4())
        def caller(_):
            s=DurableJobStore(self.tmp.name)
            return s.submit('photo',b'synthetic shirt','image/jpeg',request_id=key)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            jobs=list(pool.map(caller,range(20)))
        self.assertEqual(len(set(jobs)),1)
        self.assertEqual(len(list(Path(self.tmp.name).glob('*.input'))),1)
        with sqlite3.connect(str(Path(self.tmp.name)/'jobs.sqlite3')) as con:
            self.assertEqual(con.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],1)
            self.assertEqual(con.execute('SELECT COUNT(*) FROM request_keys').fetchone()[0],1)

    def test_tombstone_expires_and_removes_metadata_after_48h(self):
        clock=FakeClock()
        with tempfile.TemporaryDirectory() as d:
            store=DurableJobStore(d,clock=clock)
            key=str(uuid.uuid4())
            store.submit('photo',b'fake','image/jpeg',request_id=key)
            clock.advance(48*3600+1)
            self.assertEqual(store.find_request(key)['status'],'expired')
            store.cleanup()
            self.assertIsNone(store.find_request(key))

if __name__ == '__main__':unittest.main()
