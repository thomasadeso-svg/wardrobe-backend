import unittest
from outfit_engine import build_batch, complete, ACCESSORY_CAPABILITY


class ComplementaryTests(unittest.TestCase):
    def request(self, enabled=True):
        items = [dict(id=id, category=cat, subcategory=sub, style='casual', color='black')
                 for id, cat, sub in [('top', 'top', 'shirt'), ('bottom', 'bottom', 'trousers'),
                     ('shoes', 'shoes', 'boots'), ('bag', 'bag', 'tote'), ('belt', 'accessory', 'belt'),
                     ('watch', 'accessory', 'watch'), ('bracelet', 'jewelry', 'bracelet')]]
        return dict(wardrobe=items, occasion='casual', batch_size=5, anchor_item_id='top',
                    client_capabilities=[ACCESSORY_CAPABILITY] if enabled else [])

    def test_legacy_single_slot_and_capable_complementary_selection(self):
        for enabled in [False, True]:
            request = self.request(enabled)
            result = build_batch(request)
            self.assertTrue(result['outfits'])
            for look in result['outfits']:
                indices = [i['item_index'] for i in look['outfit']]
                self.assertTrue(complete(indices, request['wardrobe'], 'moderate', 'top', enabled))
                ids = {i['item_id'] for i in look['outfit']}
                if enabled:
                    self.assertTrue({'bag', 'belt'} <= ids)
                    self.assertEqual(len(ids & {'watch', 'bracelet'}), 1)
                else:
                    self.assertEqual(len(ids & {'bag', 'belt', 'watch', 'bracelet'}), 1)

    def test_ai_extra_roles_and_redundant_roles(self):
        request = self.request()
        items = request['wardrobe']
        self.assertTrue(complete([0, 1, 2, 3, 4, 5], items, 'moderate', 'top', True))
        self.assertFalse(complete([0, 1, 2, 3, 5, 6], items, 'moderate', 'top', True))
        self.assertFalse(complete([0, 1, 2, 3, 4, 5], items, 'moderate', 'top', False))

    def test_unknown_and_unsuitable_extras_are_not_automatically_added(self):
        request = self.request()
        for item in request['wardrobe'][3:]:
            item['style'] = 'formal'
        result = build_batch(request)
        self.assertEqual(len(result['outfit']), 3)
        request['wardrobe'][4]['subcategory'] = None
        request['anchor_item_id'] = 'belt'
        result = build_batch(request)
        self.assertIn('belt', {i['item_id'] for i in result['outfit']})
        self.assertEqual(len(result['outfit']), 4)

    def test_ai_prompt_and_capability_are_explicit(self):
        request = self.request()
        request['wardrobe'].append(dict(request['wardrobe'][0], id='other-top'))
        request.pop('anchor_item_id')
        prompts = []
        def ask(prompt):
            prompts.append(prompt)
            return {'outfits': [{'item_ids': ['top', 'bottom', 'shoes', 'bag', 'belt', 'watch'],
                                'explanation': 'Recorded black accessories.', 'styling_tip': 'Check the belt fit.'}]}
        result = build_batch(request, ask)
        self.assertEqual(len(prompts), 1)
        self.assertIn('no target item count', prompts[0])
        self.assertEqual(result['ai_calls'], 1)
        self.assertEqual(result['source'], 'ai')

    def test_supported_accessory_variety_remains_in_cached_catalogue(self):
        request = self.request()
        request['wardrobe'].append(dict(request['wardrobe'][3], id='bag2'))
        result = build_batch(request)
        self.assertTrue(result['complete_catalog'])
        self.assertEqual(len(result['outfits']), 4)
        self.assertEqual({i['item_id'] for look in result['outfits'] for i in look['outfit'] if i['item_id'].startswith('bag')}, {'bag', 'bag2'})

if __name__ == '__main__':
    unittest.main()
