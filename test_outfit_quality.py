import unittest
from unittest.mock import Mock
from outfit_engine import build_batch


class QualityTests(unittest.TestCase):
    def request(self):
        return {'wardrobe': [{'id': 'formal', 'category': 'top', 'style': 'formal'},
                            {'id': 'sport', 'category': 'top', 'style': 'sporty'},
                            {'id': 'bottom', 'category': 'bottom', 'style': 'formal'}],
                'occasion': 'formal', 'batch_size': 5,
                'previous_outfits': [['formal', 'bottom']]}

    def test_suitable_ai_repeat_beats_unsuitable_novelty(self):
        result = build_batch(self.request(), lambda _: {'outfits': [
            {'item_ids': ['formal', 'bottom'], 'explanation': 'Recorded formal styles suit this occasion.', 'styling_tip': 'Try it on.'}]})
        self.assertEqual(len(result['outfits']), 1)
        self.assertEqual(result['source'], 'ai')
        self.assertEqual(result['styling_tip'], 'Try it on.')
        self.assertNotIn('sport', [i['item_id'] for i in result['outfit']])

    def test_reasons_and_timings_are_sanitised(self):
        for ai, reason in [(None, 'unavailable_ai'), (Mock(side_effect=TimeoutError('SECRET IMAGE')), 'timeout'),
                           (Mock(side_effect=ValueError('PRIVATE')), 'malformed_response'),
                           (lambda _: {'outfits': [{'item_ids': ['unknown']}]}, 'rejected_proposals')]:
            with self.assertLogs('outfit_engine', level='INFO') as logs:
                result = build_batch(self.request(), ai)
            d = result['diagnostics']
            self.assertEqual(d['reason'], reason)
            self.assertAlmostEqual(d['total_ms'], d['local_ms'] + d['ai_ms'], delta=.03)
            self.assertNotIn('SECRET', str(logs.output))
            self.assertNotIn('PRIVATE', str(logs.output))

    def test_fallback_is_bilingual_and_does_not_invent_attributes(self):
        for language in ['en', 'de']:
            result = build_batch({**self.request(), 'language': language})
            self.assertTrue(result['explanation'])
            self.assertNotIn('cotton', result['explanation'])
            self.assertNotIn('A complete combination', result['explanation'])
            self.assertEqual(result['diagnostics']['ai_ms'], 0)

    def test_known_season_conflict_loses_to_repeat(self):
        request = self.request()
        request['occasion'] = 'casual'
        request['weather'] = 'hot'
        for item in request['wardrobe']:
            item.pop('style')
        request['wardrobe'][1]['season'] = ['winter']
        result = build_batch(request)
        self.assertEqual(len(result['outfits']), 1)
