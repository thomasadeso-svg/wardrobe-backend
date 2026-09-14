import unittest
from unittest.mock import Mock
from outfit_engine import build_batch
from outfit_quality import match_info


def item(id, category, **kw):
    return dict(id=id, category=category, style='casual', color='blue', season='all', **kw) if not kw else {
        'id':id,'category':category,'style':'casual','color':'blue','season':'all',**kw}


class MatchStates(unittest.TestCase):
    def run_request(self, clothes=None, ai=None, **kw):
        return build_batch({'wardrobe':clothes or [item('t','top'),item('b','bottom')],
            'occasion':'casual','weather':'moderate','batch_size':5,'client_capabilities':['match_states_v1'],**kw},ai)

    def test_fallback_and_repetition_do_not_downgrade(self):
        ai=Mock()
        r=self.run_request(ai=ai,previous_outfits=[['t','b']])
        self.assertEqual(r['match_info']['state'],'strong')
        self.assertEqual(r['source'],'fallback');ai.assert_not_called()

    def test_owned_suitability_precedes_novelty(self):
        r=self.run_request([item('old','top',style='formal'),item('new','top',style='sporty'),item('b','bottom',style='formal')],
            occasion='formal',previous_outfits=[['old','b']])
        self.assertEqual({x['item_id'] for x in r['outfit']},{'old','b'})
        self.assertEqual(r['match_info']['state'],'strong')

    def test_best_available_supported_soft_difference(self):
        r=self.run_request(style_profile={'vibe':'elegant'})
        self.assertEqual(r['match_info']['state'],'best_available')
        self.assertEqual(r['match_info']['reason_codes'],['style_preference'])
        self.assertTrue(r['match_info']['alternative_categories'])

    def test_no_sensible_does_not_return_unsuitable_outfit_or_call_ai(self):
        ai=Mock()
        r=self.run_request([item('d','dress',style='sporty')],ai,occasion='formal')
        self.assertEqual(r['outfits'],[]);ai.assert_not_called()
        self.assertEqual(r['match_info']['state'],'no_sensible')
        self.assertIn('activity_conflict',r['match_info']['reason_codes'])

    def test_missing_metadata_not_shopping_evidence(self):
        r=self.run_request([item('d','dress',style=None)])
        self.assertEqual(r['match_info']['state'],'uncertain')
        self.assertEqual(r['match_info']['alternative_categories'],[])
        r=self.run_request([item('d','dress',style=None,color='red')],style_profile={'avoid':['red']})
        self.assertEqual(r['outfits'],[])  # missing style cannot bypass known avoid
        self.assertEqual(r['match_info']['alternative_categories'],[])

    def test_service_error_is_separate_from_local_suitability(self):
        ai=Mock(side_effect=TimeoutError())
        r=self.run_request([item('a','top'),item('b','bottom'),item('c','top')],ai)
        self.assertEqual(r['match_info']['state'],'strong')
        self.assertTrue(r['match_info']['generation_error'])
        self.assertEqual(r['match_info']['alternative_categories'],[]);self.assertEqual(ai.call_count,1)

    def test_incomplete_anchor_and_unknown_categories(self):
        r=self.run_request([item('t','top'),item('d','dress')],anchor_item_id='t')
        self.assertEqual(r['outfits'],[])
        self.assertEqual(r['match_info']['alternative_categories'],['bottom'])
        r=self.run_request([item('t','top'),item('u','unknown')])
        self.assertEqual(r['match_info']['state'],'uncertain')
        self.assertEqual(r['match_info']['alternative_categories'],[])

    def test_legacy_contract_and_bounded_search(self):
        self.assertNotIn('match_info',self.run_request(client_capabilities=[]))
        info=match_info([item('d','dress',style='sporty')],{'occasion':'formal'},[],exhaustive=False)
        self.assertEqual(info['alternative_categories'],[])

    def test_unknown_season_and_accessory_colour_require_review(self):
        r=self.run_request([item('d','dress',season=['unknown'])],weather='cold')
        self.assertEqual(r['match_info']['state'],'uncertain')
        r=self.run_request([item('d','dress'),item('bag','bag',color='unknown')],style_profile={'avoid':['red']})
        self.assertEqual(r['match_info']['state'],'uncertain')
        self.assertEqual(r['match_info']['alternative_categories'],[])

    def test_invalid_ai_selection_does_not_unlock_shopping(self):
        ai=Mock(return_value={'outfits':[{'item_ids':['not-owned']}]})
        r=self.run_request([item('t','top'),item('x','top'),item('b','bottom')],ai,style_profile={'vibe':'elegant'})
        self.assertTrue(r['match_info']['generation_error'])
        self.assertEqual(r['match_info']['alternative_categories'],[])

    def test_quiz_colour_family_is_not_evidence_of_absent_clothing(self):
        for family,hue in [('earth','brown'),('pastels','pink'),('bright','red')]:
            r=self.run_request([item('d','dress',color=hue)],style_profile={'colors':[family]})
            self.assertEqual(r['match_info']['state'],'uncertain')
            self.assertEqual(r['match_info']['alternative_categories'],[])

    def test_en_de_reason_and_no_unverified_ai_prose(self):
        ai=Mock(return_value={'outfits':[{'item_ids':['t','b'],'explanation':'Guaranteed perfect fit 99%'}]})
        r=self.run_request([item('t','top'),item('x','top'),item('b','bottom')],ai,language='de')
        self.assertNotIn('99%',r['explanation'])
        self.assertIn('erfassten',r['match_info']['reason'])


if __name__=='__main__':unittest.main()
