"""Synthetic continuity guards, not visual accuracy claims about similar boots."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from scan_continuity import SourceContinuity, working_size


class ContinuityTests(unittest.TestCase):
    def scene(self, seed=1):
        frame = np.random.default_rng(100).integers(0, 256, (240, 320, 3), dtype='uint8')
        frame[40:200, 40:200] = np.random.default_rng(seed).integers(0, 256, (160,160,3), dtype='uint8')
        mask = np.zeros((240,320), dtype='uint8')
        mask[45:195,45:195] = 255
        return frame, mask

    def test_original_frame_motion_keeps_foreground_identity(self):
        frame, mask = self.scene()
        flow = SourceContinuity()
        flow.advance(frame, 0)
        identity, _ = flow.identify(mask, 'boots')
        self.assertIsNotNone(identity)
        for i in range(1, 10):
            transform = np.float32([[1,0,i], [0,1,0]])
            flow.advance(cv2.warpAffine(frame, transform, (320,240)), i/30)
            actual, evidence = flow.identify(cv2.warpAffine(mask, transform, (320,240)), 'boots')
            self.assertEqual(actual, identity)
            self.assertGreaterEqual(evidence['supported_points'], 12)

    def test_partial_pair_mask_keeps_supported_identity_and_other_trajectories(self):
        frame, mask = self.scene()
        partial = mask.copy()
        partial[:,120:] = 0
        flow = SourceContinuity()
        flow.advance(frame, 0)
        identity, _ = flow.identify(mask, 'boots')
        flow.advance(frame, 1/30)
        actual, evidence = flow.identify(partial, 'boots')
        self.assertEqual(actual, identity)
        self.assertLess(evidence['observed_fraction'], .75)
        self.assertGreaterEqual(evidence['foreground_coverage'], .3)
        self.assertGreater(len(flow.tracks[0]['points']), evidence['supported_points'])
        flow.advance(frame, 2/30)
        self.assertEqual(flow.identify(mask, 'boots')[0], identity)

    def test_same_background_different_foreground_does_not_inherit_identity(self):
        frame, mask = self.scene(1)
        other, _ = self.scene(2)
        flow = SourceContinuity()
        flow.advance(frame, 0)
        original, _ = flow.identify(mask, 'boots')
        flow.advance(other, 1/30)
        replacement, _ = flow.identify(mask, 'boots')
        self.assertNotEqual(replacement, original)

    def test_foreground_disappearing_breaks_identity(self):
        frame, mask = self.scene()
        flow = SourceContinuity()
        flow.advance(frame, 0)
        original, _ = flow.identify(mask, 'boots')
        blank = frame.copy()
        blank[40:200,40:200] = 0
        flow.advance(blank, 1/30)
        flow.advance(frame, 2/30)
        self.assertNotEqual(flow.identify(mask, 'boots')[0], original)

    def test_skipped_time_and_new_jobs_do_not_share_identity(self):
        frame, mask = self.scene()
        flow = SourceContinuity()
        flow.advance(frame, 0)
        original, _ = flow.identify(mask, 'boots')
        flow.advance(frame, .5)
        self.assertNotEqual(flow.identify(mask, 'boots')[0], original)
        self.assertEqual(SourceContinuity().tracks, [])

    def test_separate_foreground_regions_and_ambiguous_union(self):
        frame = np.random.default_rng(3).integers(0,256,(240,320,3),dtype='uint8')
        left = np.zeros((240,320),dtype='uint8'); left[30:210,20:140] = 255
        right = np.zeros_like(left); right[30:210,180:300] = 255
        flow = SourceContinuity(); flow.advance(frame,0)
        first, _ = flow.identify(left,'boots')
        second, _ = flow.identify(right,'boots')
        self.assertNotEqual(first,second)
        combined, evidence = flow.identify(cv2.bitwise_or(left,right),'boots')
        self.assertIsNone(combined)
        self.assertEqual(evidence['reason'],'ambiguous_source_tracks')

    def test_background_points_outside_the_mask_cannot_seed_a_track(self):
        frame, mask = self.scene()
        frame[40:200,40:200] = 0
        flow = SourceContinuity(); flow.advance(frame,0)
        identity, evidence = flow.identify(mask,'boots')
        self.assertIsNone(identity)
        self.assertEqual(evidence['reason'],'insufficient_source_points')

    def test_bad_forward_backward_correspondence_is_rejected(self):
        frame, mask = self.scene()
        flow = SourceContinuity(); flow.advance(frame,0)
        flow.identify(mask,'boots')
        pts = flow.tracks[0]['points']
        with patch('scan_continuity.cv2.calcOpticalFlowPyrLK', side_effect=[
                (pts.copy(),np.ones((len(pts),1),dtype='uint8'),np.zeros((len(pts),1))),
                (pts+5,np.ones((len(pts),1),dtype='uint8'),np.zeros((len(pts),1)))]):
            flow.advance(frame,1/30)
        self.assertEqual(flow.tracks,[])

    def test_working_resolution_and_point_count_are_bounded(self):
        self.assertEqual(working_size(1920,1080),(480,270))
        frame,mask = self.scene();flow=SourceContinuity();flow.advance(frame,0)
        flow.identify(mask,'boots')
        self.assertLessEqual(len(flow.tracks[0]['points']),100)

    def test_surviving_concentrated_points_are_evidence_not_an_automatic_merge(self):
        frame, mask = self.scene()
        flow = SourceContinuity(); flow.advance(frame, 0)
        original, _ = flow.identify(mask, 'boots')
        flow.tracks[0]['points'] = np.float32([(x,y) for x in (60,65,70,75) for y in (60,65,70)]).reshape(-1,1,2)
        identity, evidence = flow.identify(mask, 'boots')
        self.assertNotEqual(identity, original)
        partial = evidence['partial_support'][0]
        self.assertEqual((partial['identity'], partial['points']), (original, 12))
        self.assertLess(partial['coverage'], .3)


if __name__ == '__main__':
    unittest.main()
