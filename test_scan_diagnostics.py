"""Offline regression tests with inference tripwires and fake decoded frames."""
import ast
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
import uuid


SOURCE = Path(__file__).with_name("video_scan.py").read_text()


class Frame:
    def __init__(self, value, score):
        self.value, self.score = value, score

    def tobytes(self):
        return self.value


def environment(frames, fps=2, count=None):
    iterator = iter(frames)
    cap = SimpleNamespace(
        get=lambda prop: fps if prop == 1 else len(frames) if count is None else count,
        isOpened=lambda: True,
        read=lambda: (True, f) if (f := next(iterator, None)) else (False, None),
        release=lambda: setattr(cap, "released", True))
    calls = []

    def process(img, idx, temp_id):
        calls.append(idx)
        return {"temp_id": temp_id, "category": "shoes", "subcategory": "sneakers",
                "color": "black", "_frame_index": idx, "_crop_sig": 0, "_sharpness": 30}

    env = dict(cv2=SimpleNamespace(VideoCapture=lambda _: cap, CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2, COLOR_BGR2RGB=3, cvtColor=lambda f, _: f),
        Image=SimpleNamespace(fromarray=lambda _: SimpleNamespace(close=lambda: None)),
        hashlib=hashlib, json=json, math=math, uuid=uuid, time=time,
        logger=logging.getLogger("test"), List=list,
        MAX_VIDEO_SECONDS=20, FRAME_SAMPLE_INTERVAL_SEC=0.5, MIN_FRAME_SHARPNESS=22.0,
        MAX_FRAME_EDGE=0, MAX_ITEMS_RETURNED=20, TRACK_MAX_FRAME_GAP=6,
        TRACK_SIMILARITY_THRESHOLD=20, TRACK_DEBUG=False,
        _sharpness=lambda f: f.score, _process_single_frame_sync=process,
        _crop_distance=lambda a, b: abs(a-b), _colors_compatible=lambda a, b: a == b,
        DetectedItem=lambda **kw: kw)
    names = {"_scan_event", "_process_video_sync", "_build_garment_tracks"}
    nodes = [n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<actual-source>", "exec"), env)
    return env, calls, cap


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostic_never_processes_or_tracks(self):
        env, calls, cap = environment([Frame(b"a", 10), Frame(b"b", 22), Frame(b"c", 30)])
        def forbidden(*args, **kwargs):
            self.fail("Diagnostic mode invoked inference or tracking")
        env['_process_single_frame_sync'] = forbidden
        env['_build_garment_tracks'] = forbidden
        report = []
        self.assertEqual(env['_process_video_sync']('unused', True, report), ([], 2, 0))
        self.assertEqual([e['decision'] for e in report if e['stage'] == 'sampling'],
                         ['sharpness_rejected', 'accepted', 'accepted'])
        self.assertEqual(report[-1]['sampling_opportunities'], 3)
        self.assertEqual(report[-1]['sharpness_rejected'], 1)
        self.assertTrue(cap.released)
        self.assertEqual(calls, [])

    def test_normal_mode_reuse_and_timestamp_correlation(self):
        env, calls, _ = environment([Frame(b"a", 30), Frame(b"a", 30), Frame(b"b", 30)])
        report = []
        items, accepted, duplicates = env['_process_video_sync']('unused', report=report, scan_id='job-test')
        self.assertEqual(calls, [0, 2])
        self.assertEqual((len(items), accepted, duplicates), (1, 3, 2))
        events = [e for e in report if e['stage'] == 'classification']
        self.assertEqual([e['source'] for e in events], ['processed', 'exact_reuse', 'processed'])
        self.assertEqual([e['timestamp_seconds'] for e in events], [0, 0.5, 1])
        self.assertTrue(all(e['scan_id'] == 'job-test' for e in report))
        self.assertFalse(any(k.startswith('_') for k in items[0]))

    def test_rejected_frames_do_not_change_legacy_indices(self):
        env, calls, _ = environment([Frame(b"a", 30), Frame(b"b", 0), Frame(b"c", 30)])
        report = []
        env['_process_video_sync']('unused', report=report)
        self.assertEqual(calls, [0, 1])
        self.assertEqual([e['timestamp_seconds'] for e in report if e['stage'] == 'classification'], [0, 1])

    def test_no_category_is_reported(self):
        env, _, _ = environment([Frame(b"a", 30)])
        env['_process_single_frame_sync'] = lambda *a: None
        report = []
        self.assertEqual(env['_process_video_sync']('unused', report=report), ([], 1, 0))
        self.assertEqual(next(e for e in report if e['stage'] == 'classification')['decision'],
                         'no_category_or_invalid_json')

    def test_error_logs_no_exception_message_and_releases_capture(self):
        env, _, cap = environment([Frame(b"a", 30)])
        def fail(*args): raise RuntimeError('PRIVATE_API_RESPONSE')
        env['_process_single_frame_sync'] = fail
        report = []
        with self.assertRaises(RuntimeError): env['_process_video_sync']('unused', report=report)
        self.assertTrue(cap.released)
        self.assertNotIn('PRIVATE_API_RESPONSE', json.dumps(report))
        self.assertEqual(report[-1]['error_type'], 'RuntimeError')

    def test_duration_guard_in_diagnostic_mode(self):
        env, _, cap = environment([], fps=2, count=100)
        with self.assertRaises(ValueError): env['_process_video_sync']('unused', True, [])
        self.assertTrue(cap.released)

    def test_threshold_unchanged_and_track_decision_logged(self):
        env, _, _ = environment([])
        def candidate(idx, sig):
            return dict(_frame_index=idx, _crop_sig=sig, _sharpness=30,
                        category='shoes', color='black', subcategory='sneakers')
        report = []
        tracks = env['_build_garment_tracks']([candidate(0, 0), candidate(1, 25)], report, 'test')
        self.assertEqual(len(tracks), 2)
        self.assertIn('dist=25.0>20', next(e['reason'] for e in report
                      if e['stage'] == 'tracking' and e['frame_index'] == 1))

    def test_shoe_geometry_recovers_hash_split_with_time_and_metadata_gates(self):
        env, _, _ = environment([])
        calls = []
        env['_geometric_duplicate'] = lambda a, b: calls.append((a, b)) or True
        def candidate(idx, sig, timestamp, category='shoes', subtype='sneakers', color='black'):
            return dict(_frame_index=idx, _crop_sig=sig, _sharpness=30,
                        _timestamp_seconds=timestamp, _local_features=idx,
                        category=category, color=color, subcategory=subtype)
        first = candidate(0, 0, 0)
        report = []
        result = env['_build_garment_tracks']([first, candidate(1, 25, .5)], report)
        self.assertEqual(len(result), 1)
        self.assertEqual(report[1]['method'], 'local_features')
        for second in [candidate(1, 25, 2), candidate(1, 25, .5, subtype='sandals'),
                       candidate(1, 25, .5, color='red'), candidate(1, 25, .5, category='top')]:
            self.assertEqual(len(env['_build_garment_tracks']([first, second])), 2)
        env['_geometric_duplicate'] = lambda *args: False
        self.assertEqual(len(env['_build_garment_tracks']([first, candidate(1, 25, .5)])), 2)

    def test_elapsed_time_blocks_merge_even_after_blur_compresses_indices(self):
        env, _, _ = environment([])
        candidates = [dict(_frame_index=i, _timestamp_seconds=t, _crop_sig=0,
                           _sharpness=30, category='shoes', color='black')
                      for i, t in [(0, 0), (1, 4)]]
        self.assertEqual(len(env['_build_garment_tracks'](candidates)), 2)

    def test_nan_score_is_json_safe_and_rejected(self):
        env, _, _ = environment([Frame(b"a", float('nan'))])
        report = []
        env['_process_video_sync']('unused', True, report)
        json.dumps(report, allow_nan=False)
        self.assertEqual(report[-1]['accepted_frames'], 0)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise unittest.SkipTest("Install existing backend dependencies for real geometry tests") from exc
        cls.np, cls.Image = np, Image
        cls.env = {'cv2': cv2, 'Image': Image}
        names = {'_local_features', '_geometric_duplicate'}
        nodes = [n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-geometry>', 'exec'), cls.env)

    def texture(self, seed):
        rgb = self.np.random.default_rng(seed).integers(0, 256, (320, 320, 3), dtype='uint8')
        return self.Image.fromarray(rgb).convert('RGBA')

    def match(self, a, b):
        return self.env['_geometric_duplicate'](self.env['_local_features'](a), self.env['_local_features'](b))

    def test_rotation_of_same_texture(self):
        original = self.texture(1)
        self.assertTrue(self.match(original, original.rotate(12)))

    def test_different_textures_remain_separate(self):
        self.assertFalse(self.match(self.texture(1), self.texture(2)))

    def test_blank_crop_has_no_identity_evidence(self):
        blank = self.Image.new('RGBA', (320, 320), 'white')
        self.assertFalse(self.match(blank, blank))

    def test_small_shared_patch_is_insufficient(self):
        original, other = self.texture(1), self.texture(2)
        other.paste(original.crop((100, 100, 180, 180)), (100, 100))
        self.assertFalse(self.match(original, other))


if __name__ == '__main__':
    unittest.main()
