"""Real PIL/OpenCV pipeline functions with explicitly mocked removal/classification."""
import unittest
from unittest.mock import patch
from PIL import Image
import video_scan as scan


class PixelCacheTests(unittest.TestCase):
    def test_exact_normalized_pixels_reuse_classification_only_within_job(self):
        crop = Image.new('RGBA', (120, 120), (30, 30, 30, 255))
        with patch.object(scan, '_remove_background', return_value=crop), patch.object(
                scan, '_classify_with_claude', return_value={'category': 'shoes', 'subcategory': 'boots'}) as classify:
            cache = {}
            a = scan._process_single_frame_sync(crop, 0, 'a', cache)
            b = scan._process_single_frame_sync(crop, 1, 'b', cache)
            self.assertEqual(classify.call_count, 1)
            self.assertEqual(a['image_base64'], b['image_base64'])
            scan._process_single_frame_sync(crop, 0, 'new-job', {})
            self.assertEqual(classify.call_count, 2)

    def test_similar_shoe_silhouettes_without_distinctive_details_are_reviewable(self):
        a = dict(temp_id='a', category='shoes', subcategory='boots', color='black',
                 _frame_index=0, _timestamp_seconds=0, _sharpness=30, image_base64='a')
        b = dict(a, temp_id='b', _frame_index=1, _timestamp_seconds=.5, image_base64='b')
        crop = Image.new('RGBA', (120, 120), (30, 30, 30, 255))
        for candidate in [a, b]:
            candidate.update(_crop_sig=scan._crop_signature(crop), _palette=scan._foreground_palette(crop),
                             _local_features=scan._local_features(crop))
        with patch.object(scan, 'TRACK_DEBUG', False):
            result = scan._build_garment_tracks([a, b])
        self.assertEqual(len(result), 2)
        self.assertEqual(sum(item.get('confidence') == 'low' for item in result), 1)

    def test_faded_pixels_cannot_be_auto_selected_from_classifier_confidence(self):
        crop = Image.new('RGBA', (120, 120), (30, 30, 30, 80))
        with patch.object(scan, '_remove_background', return_value=crop), patch.object(
                scan, '_classify_with_claude', return_value={'category': 'shoes', 'confidence': 'high'}):
            result = scan._process_single_frame_sync(crop, 0, 'faded', {})
        self.assertEqual(result['confidence'], 'low')

if __name__ == '__main__':
    unittest.main()
