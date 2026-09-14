"""Synthetic adapter guards; genuine-frame replay is separate and labeled."""
import copy
import json
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw
from scan_regions import adapt, parse
import video_scan as scan


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGBA',(100,100),(0,0,0,0))
        ImageDraw.Draw(self.image).rectangle((10,10,40,80),fill=(40,40,40,255))
        self.item=dict(id='a',category='shoes',subcategory='boots',unit='pair',bbox=[10,10,41,81],
                       component_ids=[1],visible_shoes=2,shoe_boxes=[[10,10,25,81],[25,10,41,81]],confidence='high')
    def payload(self,items=None):return dict(items=items or [self.item],complete=True,unresolved=[])
    def convert(self,body):
        with patch.object(scan,'_remove_background',side_effect=AssertionError('No inference')),patch.object(
                scan,'_classify_with_claude',side_effect=AssertionError('No classification')):
            return scan._process_detected_regions_sync(self.image.convert('RGB'),self.image,json.dumps(body),0,0,'job:frame')
    def test_one_fence_is_handled_but_extra_text_is_rejected(self):
        body=json.dumps(self.payload())
        self.assertEqual(parse('```json\n'+body+'\n```'),self.payload())
        for text in ('preamble '+body,'```json\n```json\n'+body+'\n```\n```',body+' suffix'):
            with self.assertRaises(ValueError):parse(text)
    def test_nonfinite_json_is_rejected(self):
        with self.assertRaises(ValueError):parse('{"items":[],"extra":NaN}')
    def test_excess_items_are_an_explicit_frame_failure_not_silent_truncation(self):
        candidates,report=self.convert(self.payload([dict(self.item,id=str(i)) for i in range(5)]))
        self.assertEqual(candidates,[])
        self.assertIn('item_limit_exceeded',report['diagnostics'])
        self.assertEqual(len(report['raw_detection']['items']),5)
    def test_pair_union_does_not_split_disconnected_components(self):
        ImageDraw.Draw(self.image).rectangle((65,10,90,80),fill=(40,40,40,255))
        self.item.update(component_ids=[1,2],bbox=[10,10,91,81],shoe_boxes=[[10,10,41,81],[65,10,91,81]])
        candidates,_=self.convert(self.payload())
        self.assertEqual(len(candidates),1);self.assertFalse(candidates[0]['review_required'])
    def test_optional_incomplete_boxes_preserve_accepted_pair(self):
        self.item['shoe_boxes']=[self.item['bbox']]
        candidates,_=self.convert(self.payload())
        self.assertEqual(len(candidates),1);self.assertFalse(candidates[0]['review_required'])
        self.assertEqual(candidates[0]['image_kind'],'cutout')
        self.assertIn('incomplete_optional_shoe_details',candidates[0]['detection_notes'])

    def test_missing_optional_boxes_do_not_reject_pair(self):
        del self.item['shoe_boxes']
        candidates,_=self.convert(self.payload())
        self.assertFalse(candidates[0]['review_required'])

    def test_missing_count_is_uncertainty_and_single_with_two_shoes_is_contradictory(self):
        self.item.pop('visible_shoes')
        candidates,_=self.convert(self.payload())
        self.assertTrue(candidates[0]['review_required'])
        self.assertIn('missing_visible_shoe_count',candidates[0]['detection_diagnostics'])
        self.assertNotIn('shoe_count_contradiction',candidates[0]['detection_diagnostics'])
        self.item.update(unit='single',visible_shoes=2)
        candidates,_=self.convert(self.payload())
        self.assertIn('shoe_count_contradiction',candidates[0]['detection_diagnostics'])

    def test_real_count_contradictions_require_review(self):
        for count in (1,3,-1,True,'2'):
            self.item['visible_shoes']=count
            candidates,_=self.convert(self.payload())
            self.assertTrue(candidates[0]['review_required'])
            self.assertIn('shoe_count_contradiction',candidates[0]['detection_diagnostics'])
            self.assertTrue(candidates[0]['_identity_unverified'])

    def test_contradictory_pair_cannot_disappear_through_exact_crop_merge(self):
        self.item['visible_shoes']=3
        bad,_=self.convert(self.payload())
        self.item['visible_shoes']=2
        good,_=self.convert(self.payload())
        good[0].update(temp_id='later',_frame_index=1,_timestamp_seconds=.5,_observation_ids={'later'})
        with patch.object(scan,'TRACK_DEBUG',False):result=scan._build_garment_tracks(bad+good)
        self.assertEqual(len(result),2)

    def test_incomplete_optional_boxes_do_not_bypass_geometry(self):
        for box,code in [([-1,0,20,30],'invalid_shoe_coordinates'),([0,0,20,30],'shoe_outside_pair')]:
            self.item['shoe_boxes']=[box]
            candidates,_=self.convert(self.payload())
            self.assertTrue(candidates[0]['review_required'])
            self.assertIn(code,candidates[0]['detection_diagnostics'])
    def test_invalid_coordinates_are_not_clamped_or_discarded(self):
        self.item['bbox']=[-1,0,40,80]
        candidates,report=self.convert(self.payload())
        self.assertEqual(len(candidates),1);self.assertIsNone(candidates[0]['detection_region'])
        self.assertTrue(candidates[0]['_identity_unverified'])
        self.assertEqual(report['raw_detection']['items'][0]['bbox'][0],-1)
    def test_unknown_structured_metadata_is_diagnostic_not_a_crash(self):
        self.item.update(category=[],confidence={},unit=[])
        candidates,_=self.convert(self.payload())
        self.assertIsNone(candidates[0]['category']);self.assertTrue(candidates[0]['review_required'])
    def test_shared_components_are_not_assigned_twice(self):
        second=dict(self.item,id='b')
        result=adapt(json.dumps(self.payload([self.item,second])),self.image)
        self.assertTrue(all(r['mask'] is None for r in result['records']))
        self.assertTrue(all('conflicting_components' in r['diagnostics'] for r in result['records']))
    def test_partial_without_mask_remains_reviewable_source_preview(self):
        self.item.update(unit='single',visible_shoes=1,shoe_boxes=[self.item['bbox']],component_ids=[])
        candidates,_=self.convert(self.payload())
        self.assertEqual(candidates[0]['image_kind'],'unverified_source_region')
        self.assertIn('partial_pair',candidates[0]['detection_diagnostics'])
    def test_covisible_same_subtype_identical_pixels_cannot_merge_or_flag(self):
        a=dict(temp_id='a',category='shoes',subcategory='boots',color='black',image_base64='same',
               _frame_index=0,_timestamp_seconds=0,_crop_sig=scan._crop_signature(self.image),_sharpness=30,_source_exclusions={'b'},_observation_ids={'a'})
        b=dict(a,temp_id='b',_source_exclusions={'a'},_observation_ids={'b'})
        with patch.object(scan,'TRACK_DEBUG',False):result=scan._build_garment_tracks([a,b])
        self.assertEqual(len(result),2);self.assertFalse(any(i.get('possible_duplicate_of') for i in result))
    def test_clearer_representative_does_not_erase_prior_diagnostics(self):
        a=dict(temp_id='a',category='shoes',subcategory='boots',color='black',image_base64='same',confidence='low',
               _frame_index=0,_timestamp_seconds=0,_crop_sig=scan._crop_signature(self.image),_sharpness=10,
               detection_diagnostics=['shoe_count_contradiction'],review_required=True)
        b=dict(a,temp_id='b',confidence='high',_frame_index=1,_timestamp_seconds=.5,_sharpness=40,
               detection_diagnostics=[],review_required=False)
        with patch.object(scan,'TRACK_DEBUG',False):result=scan._build_garment_tracks([a,b])
        self.assertEqual(len(result),1);self.assertTrue(result[0]['review_required'])
        self.assertEqual(result[0]['temp_id'],'b')


if __name__=='__main__':unittest.main()
