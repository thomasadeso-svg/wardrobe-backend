"""Lazy startup and model-session reuse without importing inference dependencies."""
import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def fresh_module():
    spec = importlib.util.spec_from_file_location('isolated_removal', Path(__file__).with_name('background_removal.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BackgroundRemovalTests(unittest.TestCase):
    def test_import_does_not_load_inference_or_download_models(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'rembg', 'pymatting', 'numba', 'onnxruntime', 'pooch'}:
                self.fail(f'Startup imported inference dependency: {name}')
            return original(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=guarded):
            module = fresh_module()
        self.assertEqual(module._sessions, {})

    def test_explicit_models_reuse_their_own_session(self):
        module = fresh_module()
        first, second = object(), object()
        fake = SimpleNamespace(new_session=Mock(side_effect=[first, second]), remove=Mock(return_value=b'cutout'))
        with patch.dict(sys.modules, rembg=fake):
            self.assertEqual(module.remove_background_bytes(b'photo'), b'cutout')
            module.remove_background_bytes(b'video', model='u2net')
            module.remove_background_bytes(b'next-photo')
        self.assertEqual(fake.new_session.call_args_list, [unittest.mock.call('u2netp'), unittest.mock.call('u2net')])
        self.assertEqual([call.kwargs['session'] for call in fake.remove.call_args_list], [first, second, first])

    def test_failed_initialization_is_reported_and_not_cached(self):
        module = fresh_module()
        fake = SimpleNamespace(new_session=Mock(side_effect=RuntimeError('model unavailable')), remove=Mock())
        with patch.dict(sys.modules, rembg=fake), self.assertRaisesRegex(RuntimeError, 'model unavailable'):
            module.remove_background_bytes(b'photo')
        self.assertEqual(module._sessions, {})
        fake.remove.assert_not_called()


if __name__ == '__main__':
    unittest.main()
