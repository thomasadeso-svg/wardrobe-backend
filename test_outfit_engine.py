import ast
import asyncio
import json
from pathlib import Path
import re
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from outfit_engine import build_batch, complete, prepare


def wardrobe(tops=3, bags=1):
    return ([{"id": f"t{i}", "category": "top"} for i in range(tops)] +
            [{"id": "b", "category": "bottom"}, {"id": "s", "category": "shoes"}] +
            [{"id": f"a{i}", "category": "bag"} for i in range(bags)])


def ids(outfit):
    return frozenset(item["item_id"] for item in outfit["outfit"])


class OutfitTests(unittest.TestCase):
    def request(self, **kwargs):
        return {"wardrobe": wardrobe(), "batch_size": 5, "weather": "moderate", **kwargs}

    def assert_complete(self, result, request):
        items, anchor = prepare(request)
        combinations = []
        for suggestion in result["outfits"]:
            self.assertTrue(complete([i["item_index"] for i in suggestion["outfit"]], items, request.get("weather"), anchor))
            combinations.append(ids(suggestion))
        self.assertEqual(len(combinations), len(set(combinations)))

    def test_repeated_ai_is_one_call_and_unique_batch(self):
        ai = Mock(return_value={"outfits": [{"item_ids": ["t0", "b", "s", "a0"]}] * 5})
        request = self.request()
        result = build_batch(request, ai)
        self.assertEqual(ai.call_count, 1)
        self.assertEqual(len(result["outfits"]), 3)
        self.assertTrue(result["complete_catalog"])
        self.assert_complete(result, request)

    def test_missing_categories_invalid_ids_and_indices_use_fallback(self):
        for proposal in [{"item_ids": ["t0"]}, {"item_ids": ["invented", "b", "s", "a0"]},
                         {"selected_indices": [-1, 1, 2]}, {"selected_indices": [True, 1, 2]},
                         {"selected_indices": ["0", 1, 2]}, {"item_ids": ["t0", "t0", "b", "s", "a0"]}]:
            with self.subTest(proposal=proposal):
                request = self.request()
                result = build_batch(request, Mock(return_value={"outfits": [proposal]}))
                self.assertTrue(all(o["source"] == "fallback" for o in result["outfits"]))
                self.assert_complete(result, request)

    def test_ai_error_no_retry(self):
        ai = Mock(side_effect=ValueError("malformed JSON"))
        request = self.request()
        result = build_batch(request, ai)
        self.assertEqual(ai.call_count, 1)
        self.assert_complete(result, request)

    def test_core_variety_before_bag_changes(self):
        request = self.request(wardrobe=wardrobe(tops=3, bags=3), previous_outfits=[["t0", "b", "s", "a0"]])
        result = build_batch(request)
        first_two = [ids(o) for o in result["outfits"][:2]]
        self.assertTrue(all("t0" not in combo for combo in first_two))
        self.assertEqual(len({next(i for i in combo if i.startswith('t')) for combo in first_two}), 2)

    def test_small_wardrobe_rotates_before_repeating(self):
        request = self.request(wardrobe=wardrobe(tops=2), batch_size=1)
        history = []
        actual = []
        for _ in range(6):
            request["previous_outfits"] = history
            result = build_batch(request)
            actual.append(ids(result))
            history.append(list(ids(result)))
        self.assertEqual(actual[0], actual[2])
        self.assertEqual(actual[1], actual[3])
        self.assertNotEqual(actual[0], actual[1])

    def test_order_independent_ids_survive_reorder(self):
        previous = ["s", "a0", "b", "t0"]
        result = build_batch(self.request(wardrobe=list(reversed(wardrobe())), previous_outfits=[previous]))
        self.assertNotIn("t0", ids(result))

    def test_dress_replaces_top_and_bottom(self):
        request = self.request(wardrobe=[{"id": "d", "category": "Dress"}, {"id": "s", "category": "shoes"}, {"id": "a", "category": "accessory"}])
        result = build_batch(request)
        self.assertEqual(ids(result), {"d", "s", "a"})
        self.assert_complete(result, request)

    def test_no_invention_when_categories_not_owned(self):
        request = self.request(wardrobe=[{"id": "d", "category": "dress"}])
        self.assertEqual(ids(build_batch(request)), {"d"})
        ai = Mock()
        self.assertEqual(build_batch(self.request(wardrobe=[{"id": "s", "category": "shoes"}]), ai)["outfits"], [])
        ai.assert_not_called()

    def test_weather_outerwear_enforced_for_ai_and_fallback(self):
        items = wardrobe() + [{"id": "coat", "category": "outerwear"}]
        for weather in ['cold', 'moderate', 'hot']:
            request = self.request(wardrobe=items, weather=weather)
            ai = Mock(return_value={"outfits": [{"item_ids": ["t0", "b", "s", "a0", "coat"]}]})
            result = build_batch(request, ai)
            self.assert_complete(result, request)
            if weather == 'cold':
                self.assertTrue(all('coat' in ids(o) for o in result['outfits']))
            if weather == 'hot':
                self.assertTrue(all('coat' not in ids(o) for o in result['outfits']))

    def test_anchor_all_categories(self):
        items = wardrobe() + [{"id": "d", "category": "dress"}, {"id": "coat", "category": "outerwear"}]
        for anchor in ['t0', 'b', 's', 'a0', 'd', 'coat']:
            request = self.request(wardrobe=items, anchor_item_id=anchor)
            result = build_batch(request)
            self.assertTrue(result['outfits'])
            self.assertTrue(all(anchor in ids(o) for o in result['outfits']))
            self.assert_complete(result, request)

    def test_invalid_anchor_and_duplicate_wardrobe_ids(self):
        with self.assertRaises(ValueError):
            build_batch(self.request(anchor_item_id='missing'))
        with self.assertRaises(ValueError):
            build_batch(self.request(wardrobe=wardrobe() * 2))

    def test_legacy_single_outfit_response(self):
        request = {"wardrobe": [{"category": "top"}, {"category": "bottom"}]}
        result = build_batch(request)
        self.assertEqual(len(result['outfits']), 1)
        self.assertEqual([o['item_index'] for o in result['outfit']], [0, 1])

    def test_prompt_contains_preferences_anchor_and_context(self):
        ai = Mock(return_value={})
        build_batch(self.request(wardrobe=wardrobe(bags=2), style_profile={'vibe': 'minimal', 'budget': 'thrifty'}, anchor_item_id='t0', occasion='formal'), ai)
        prompt = ai.call_args.args[0]
        for value in ['minimal', 'thrifty', 'formal', 'anchor_item_id']:
            self.assertIn(value, prompt)

    def test_prefer_suitable_alternatives_and_less_recent_items(self):
        items = wardrobe()
        items[0]['color'] = 'pink'
        items[1]['color'] = 'black'
        items[2]['color'] = 'black'
        request = self.request(wardrobe=items, style_profile={'avoid': ['pink']}, previous_outfits=[['t1', 'b', 's', 'a0']])
        result = build_batch(request)
        self.assertIn('t2', ids(result))
        self.assertNotIn('t0', ids(result))

    def test_only_accessory_variants_rotate_when_core_is_fixed(self):
        request = self.request(wardrobe=wardrobe(tops=1, bags=3), batch_size=1)
        history = []
        for _ in range(3):
            request['previous_outfits'] = history
            result = build_batch(request)
            self.assertNotIn(set(ids(result)), [set(combo) for combo in history])
            history.append(list(ids(result)))

    def test_five_suggestions_when_enough_core_combinations_exist(self):
        request = self.request(wardrobe=wardrobe(tops=8, bags=2))
        result = build_batch(request)
        self.assertEqual(len(result['outfits']), 5)
        self.assertEqual(len({next(i for i in ids(o) if i.startswith('t')) for o in result['outfits']}), 5)
        self.assertFalse(result['complete_catalog'])

    def test_bounded_search_covers_cores_before_accessory_variants(self):
        with patch('outfit_engine.MAX_CANDIDATES', 4):
            result = build_batch(self.request(wardrobe=wardrobe(tops=3, bags=5)))
        self.assertEqual(len({next(i for i in ids(o) if i.startswith('t')) for o in result['outfits']}), 3)
        self.assertFalse(result['complete_catalog'])

    def test_endpoint_single_sdk_call_no_retries_and_timeout(self):
        # Compile only the actual endpoint, avoiding all scan/model initialization.
        tree = ast.parse(Path('backend-main.py').read_text(encoding='utf-8'))
        route = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'generate_outfit')
        route.decorator_list = []
        client = Mock()
        client.with_options.return_value.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(text='{"outfits": []}')])
        namespace = {'asyncio': asyncio, 're': re, 'json': json, 'client': client, 'HTTPException': ValueError}
        exec(compile(ast.Module(body=[route], type_ignores=[]), 'backend-main.py', 'exec'), namespace)
        result = asyncio.run(namespace['generate_outfit'](self.request()))
        client.with_options.assert_called_once_with(max_retries=0, timeout=18.0)
        self.assertEqual(client.with_options.return_value.messages.create.call_count, 1)
        self.assert_complete(result, self.request())

    def test_timing_mocked_first_result(self):
        def ai(_):
            time.sleep(0.02)
            return {"outfits": []}
        result = build_batch(self.request(), ai)
        print(f"Mocked backend first batch: {result['timing_ms']:.2f} ms (20 ms artificial AI delay), AI calls={result['ai_calls']}")
        self.assertGreaterEqual(result['timing_ms'], 20)

    def test_single_or_fully_explored_catalog_never_calls_ai(self):
        ai = Mock(side_effect=AssertionError('No AI call needed'))
        request = self.request(wardrobe=wardrobe(tops=1))
        result = build_batch(request, ai)
        self.assert_complete(result, request)
        self.assertEqual(result['ai_calls'], 0)
        request = self.request(wardrobe=wardrobe(tops=2))
        first = build_batch(request)
        request['previous_outfits'] = [list(ids(o)) for o in first['outfits']]
        result = build_batch(request, ai)
        self.assert_complete(result, request)
        self.assertEqual(len(result['outfits']), 2)
        self.assertEqual(result['ai_calls'], 0)
        ai.assert_not_called()

    def test_selected_outerwear_remains_in_hot_weather(self):
        request = self.request(wardrobe=wardrobe() + [{'id': 'coat', 'category': 'outerwear'}],
                               anchor_item_id='coat', weather='hot')
        result = build_batch(request)
        self.assert_complete(result, request)
        self.assertTrue(all('coat' in ids(o) for o in result['outfits']))

    def test_root_advertises_matching_outfit_api(self):
        tree = ast.parse(Path('backend-main.py').read_text(encoding='utf-8'))
        route = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'root')
        route.decorator_list = []
        namespace = {'REMOVE_BG_API_KEY': None, 'ANTHROPIC_API_KEY': None, 'vacation_cache': {}}
        exec(compile(ast.Module(body=[route], type_ignores=[]), 'backend-main.py', 'exec'), namespace)
        result = asyncio.run(namespace['root']())
        self.assertEqual(result['outfit_api_version'], 2)
        self.assertIn('exact_item_anchor', result['outfit_features'])


if __name__ == '__main__':
    unittest.main()
