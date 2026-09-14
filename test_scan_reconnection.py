"""Synthetic decision guards, not recognition accuracy for similar shoe pairs."""
import unittest
from test_scan_diagnostics import environment


class ReconnectionTests(unittest.TestCase):
    def setUp(self):
        self.env, _, _ = environment([])
        self.env['_geometric_duplicate'] = lambda a, b: a[1] == b[1]
        self.events = []

    def track(self, identity, start, end, partial=(), family='boots', detail='same', sharpness=1):
        return dict(track_id=identity, category='shoes', subcategory=family,
                    first_frame=start, last_frame=end, source_continuity_ids={identity},
                    source_partial_ids=set(partial), source_views=[dict(features=(list(range(12)), detail),
                    palette='same', frame_index=start, temp_id=str(identity))],
                    best=dict(temp_id=str(identity), _sharpness=sharpness))

    def reconnect(self, *tracks):
        return self.env['_reconnect_source_tracks'](list(tracks),
            lambda a, b: max((a, b), key=lambda c: c['_sharpness']), self.events, None)

    def test_surviving_flow_plus_geometry_consolidates_and_keeps_clearest(self):
        result = self.reconnect(self.track(0, 0, 3), self.track(1, 10, 13, partial=(0,), sharpness=10))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['best']['temp_id'], '1')
        merges = [e for e in self.events if e.get('decision') == 'merged']
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0]['method'], 'source_foreground_geometry')

    def test_geometry_without_surviving_trajectories_cannot_reconnect(self):
        self.assertEqual(len(self.reconnect(self.track(0, 0, 3), self.track(1, 10, 13))), 2)

    def test_partial_coverage_and_metadata_without_geometry_cannot_reconnect(self):
        self.assertEqual(len(self.reconnect(self.track(0, 0, 3), self.track(1, 10, 13, partial=(0,), detail='other'))), 2)

    def test_missing_palette_abstains(self):
        b = self.track(1, 10, 13, partial=(0,))
        b['source_views'][0]['palette'] = None
        self.assertEqual(len(self.reconnect(self.track(0, 0, 3), b)), 2)

    def test_covisible_identical_appearance_is_not_reconnected(self):
        self.assertEqual(len(self.reconnect(self.track(0, 0, 10), self.track(1, 10, 13, partial=(0,)))), 2)

    def test_boot_and_trainer_subtype_veto(self):
        self.assertEqual(len(self.reconnect(self.track(0, 0, 3), self.track(1, 10, 13, partial=(0,), family='sneakers'))), 2)

    def test_competing_matches_abstain_without_greedy_or_transitive_merge(self):
        result = self.reconnect(self.track(0, 0, 3), self.track(1, 10, 13, partial=(0,)),
                                self.track(2, 20, 23, partial=(0,)))
        self.assertEqual(len(result), 3)
        self.assertFalse(any(e.get('decision') == 'merged' for e in self.events))

    def test_track_limit_abstains_without_discarding_items(self):
        tracks = [self.track(i, i*2, i*2+1, partial=range(i)) for i in range(41)]
        self.assertEqual(len(self.reconnect(*tracks)), 41)
        self.assertEqual(self.events[0]['reason'], 'track_limit')

    def test_absent_source_appearance_keeps_legacy_result(self):
        a = self.track(0, 0, 3)
        a['source_views'] = []
        self.assertEqual(len(self.reconnect(a, self.track(1, 10, 13, partial=(0,)))), 2)


if __name__ == '__main__':
    unittest.main()
