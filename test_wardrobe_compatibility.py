import ast
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch
from wardrobe_compatibility import assess


def item(category, **extra):
    return {'category': category, 'style': 'casual', 'color': 'black', **extra}


class CompatibilityTests(unittest.TestCase):
    def test_top_requires_bottom_despite_unrelated_dress(self):
        result = assess({'new_item': item('top'), 'wardrobe': [item('dress')]})
        self.assertEqual(result['assessment_state'], 'missing_essential')
        self.assertEqual(result['matching_indices'], [])
        self.assertEqual(result['outfits'], [])
        self.assertEqual(result['missing_categories'], ['bottom'])

    def test_supported_counts_are_distinct_and_backed_by_complete_outfits(self):
        result = assess({'new_item': item('top'), 'wardrobe': [item('bottom'), item('bottom'), item('shoes'), item('bag')]})
        indices = {i for look in result['outfits'] for i in look['wardrobe_indices']}
        self.assertEqual(set(result['matching_indices']), indices)
        self.assertEqual(result['match_count'], len(indices))
        for look in result['outfits']:
            self.assertEqual(len(look['wardrobe_indices']), 3)
        self.assertEqual(result['assessment_state'], 'supported')

    def test_dress_replaces_core_and_optional_missing_categories_are_not_shopping_evidence(self):
        result = assess({'new_item': item('dress'), 'wardrobe': [item('bag')]})
        self.assertEqual(result['missing_categories'], [])
        self.assertEqual(result['match_count'], 1)
        self.assertEqual(result['outfits'][0]['wardrobe_indices'], [0])

    def test_uncertain_metadata_or_clear_conflict_never_claims_no_matches_or_smart_buy(self):
        for bottom in [item('bottom', style='sporty'), {'category': 'bottom'}]:
            result = assess({'new_item': item('top', style='formal'), 'wardrobe': [bottom], 'occasion': 'formal'})
            self.assertEqual(result['assessment_state'], 'uncertain')
            self.assertEqual(result['match_count'], 0)
            self.assertEqual(result['missing_categories'], [])
            self.assertNotIn('SMART BUY', result['verdict'])

    def test_validation_and_legacy_index_contract(self):
        with self.assertRaises(ValueError):
            assess({'new_item': {'category': 'chair'}})
        with self.assertRaises(ValueError):
            assess({'new_item': item('top'), 'wardrobe': [item('bottom', id='x'), item('shoes', id='x')]})
        result = assess({'new_item': item('top'), 'wardrobe': [item('bottom')], 'language': 'de'})
        for field in ['match_count', 'matching_indices', 'outfits', 'verdict', 'color_harmony', 'style_fit']:
            self.assertIn(field, result)
        self.assertIn('erfassten', result['verdict'])

    def test_endpoint_does_not_call_ai_and_technical_failure_propagates(self):
        tree = ast.parse(Path('backend-main.py').read_text(encoding='utf-8'))
        endpoint = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'match_item')
        endpoint.decorator_list = []
        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                self.status_code = status_code
                self.detail = detail
        namespace = {'asyncio': asyncio, 'HTTPException': HTTPException}
        exec(compile(ast.Module(body=[endpoint], type_ignores=[]), '<endpoint>', 'exec'), namespace)
        with patch('wardrobe_compatibility.assess', side_effect=RuntimeError('technical failure')):
            with self.assertRaises(RuntimeError):
                asyncio.run(namespace['match_item']({'new_item': item('top')}))
        result = asyncio.run(namespace['match_item']({'new_item': item('dress'), 'wardrobe': []}))
        self.assertEqual(result['assessment_source'], 'local_metadata')


if __name__ == '__main__':
    unittest.main()
