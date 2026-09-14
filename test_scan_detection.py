"""Synthetic transport/cap guards; not detection accuracy evidence."""
import json
from contextlib import nullcontext
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import video_scan as scan
import scan_detection as detection


class DetectionTests(unittest.TestCase):
    def client(self,body=None):
        client=Mock();client.with_options.return_value=client
        client.messages.create.return_value=SimpleNamespace(usage=SimpleNamespace(input_tokens=10,output_tokens=10),
            stop_reason='end_turn',content=[SimpleNamespace(type='text',text=json.dumps(body or {'items':[],'complete':True,'unresolved':[]}))])
        return client

    def run_video(self,values,client,candidate_count=0,real_adapter=False,cancel_event=None):
        frames=[np.full((60,60,3),v,dtype='uint8') for v in values];iterator=iter(frames)
        cap=SimpleNamespace(get=lambda p:2 if p==scan.cv2.CAP_PROP_FPS else len(frames),isOpened=lambda:True,
            read=lambda: (True,f) if (f:=next(iterator,None)) is not None else (False,None),release=Mock())
        def adapt(*args):
            return ([dict(temp_id=str(i),category='top',image_base64='synthetic') for i in range(candidate_count)],{'diagnostics':[]})
        detector=detection.RegionDetector(client);issues=[]
        with (patch.object(scan.cv2,'VideoCapture',return_value=cap),patch.object(scan,'_sharpness',return_value=100),patch.object(
                scan,'SourceContinuity'),patch.object(scan,'reconcile_backward'),patch.object(scan,'_classify_with_claude',side_effect=AssertionError('Scalar fallback')),
                patch.object(scan,'_remove_background',side_effect=lambda img:img.convert('RGBA')) as removal,
                (nullcontext() if real_adapter else patch.object(scan,'_process_detected_regions_sync',side_effect=adapt)),patch.object(scan,'_build_garment_tracks',side_effect=lambda items,**kw:items)):
            result=scan._process_video_sync('synthetic',region_detector=detector,frame_issues=issues,cancel_event=cancel_event)
        self.assertTrue(cap.release.called)
        return result,issues,detector,removal.call_count

    def test_one_request_per_frame_not_per_candidate_and_exact_reuse(self):
        client=self.client();result,issues,detector,removals=self.run_video([1,1,2],client,4)
        self.assertEqual(detector.attempts,2);self.assertEqual(removals,2)
        self.assertEqual(len(result[0]),12);self.assertEqual(issues,[])
        self.assertEqual(client.with_options.call_args.kwargs,{'max_retries':0,'timeout':45})

    def test_failed_request_is_not_retried_or_replaced_by_scalar(self):
        client=self.client();client.messages.create.side_effect=TimeoutError()
        result,issues,detector,removals=self.run_video([1,1,1],client)
        self.assertEqual(detector.attempts,1);self.assertEqual(removals,1)
        self.assertEqual(client.messages.create.call_count,1)
        self.assertEqual(result[0],[]);self.assertEqual(issues[0]['code'],'detection_failed')

    def test_request_limit_stops_before_more_removal(self):
        result,issues,detector,removals=self.run_video(list(range(13)),self.client())
        self.assertEqual(detector.attempts,12);self.assertEqual(removals,12)
        self.assertEqual(issues[-1]['code'],'request_limit_reached')

    def test_candidate_bound_does_not_silently_truncate_results(self):
        result,issues,detector,removals=self.run_video(list(range(13)),self.client(),4)
        self.assertEqual(len(result[0]),48);self.assertEqual(detector.attempts,12)
        self.assertEqual(issues[-1]['code'],'processing_limit_reached')

    def test_request_bounds_and_incomplete_response(self):
        client=self.client();detector=detection.RegionDetector(client)
        with self.assertRaises(detection.DetectionLimit):detector.detect(Image.new('RGB',(1025,10)),Image.new('RGBA',(1025,10)))
        self.assertEqual(detector.attempts,0)
        client.messages.create.return_value.stop_reason='max_tokens'
        with self.assertRaises(ValueError):detector.detect(Image.new('RGB',(60,60)),Image.new('RGBA',(60,60)))
        self.assertEqual(detector.attempts,1)

    def test_caches_and_budgets_are_job_local(self):
        client=self.client()
        for _ in range(2):
            _,_,detector,removals=self.run_video([1,1],client)
            self.assertEqual(detector.attempts,1);self.assertEqual(removals,1)
        self.assertEqual(client.messages.create.call_count,2)

    def test_invalid_category_is_rejected_with_reviewable_frame_diagnostic(self):
        client=self.client({'items':[{'id':'invalid','category':'wall','subcategory':'wall','unit':'single',
            'bbox':[0,0,60,60],'component_ids':[1],'confidence':'high'}],'complete':True,'unresolved':[]})
        result,issues,detector,removals=self.run_video([1],client,real_adapter=True)
        self.assertEqual(result[0],[]);self.assertEqual(issues[0]['code'],'rejected_detections')
        self.assertIn('invalid_category',issues[0]['records'][0]['diagnostics'])

    def test_cancel_after_inflight_request_prevents_next_request(self):
        client=self.client();response=client.messages.create.return_value;cancel=threading.Event()
        def respond(**kwargs):cancel.set();return response
        client.messages.create.side_effect=respond
        with self.assertRaises(InterruptedError):self.run_video([1,2],client,cancel_event=cancel)
        self.assertEqual(client.messages.create.call_count,1)


if __name__=='__main__':unittest.main()
