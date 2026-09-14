"""Job-local foreground point continuity across original decoded video frames.

This is an identity aid, not an object detector. Background matches alone cannot
create an association. Losing the tracked foreground points ends the identity.
"""
import cv2
import numpy as np


def working_size(width, height):
    scale = min(1.0, 480 / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def source_mask(alpha):
    width, height = alpha.size
    return (np.asarray(alpha.resize(working_size(width, height))) >= 64).astype('uint8') * 255


class SourceContinuity:
    MIN_POINTS = 12

    def __init__(self):
        self.gray = None
        self.timestamp = None
        self.tracks = []
        self.next_id = 0

    def advance(self, frame, timestamp):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(frame, working_size(width, height)), cv2.COLOR_BGR2GRAY)
        if (self.gray is None or self.gray.shape != gray.shape or self.timestamp is None
                or not 0 < timestamp - self.timestamp <= .2):
            self.tracks.clear()
        else:
            kept = []
            for track in self.tracks:
                points = track['points']
                moved, status, error = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None,
                    winSize=(21, 21), maxLevel=3)
                if moved is None:
                    continue
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, moved, None,
                    winSize=(21, 21), maxLevel=3)
                if back is None:
                    continue
                valid = ((status.ravel() == 1) & (back_status.ravel() == 1)
                         & (error.ravel() <= 12)
                         & (np.linalg.norm(back - points, axis=2).ravel() <= 1))
                valid &= np.isfinite(moved).all(axis=(1, 2))
                valid &= ((moved[:, 0, 0] >= 0) & (moved[:, 0, 0] < gray.shape[1])
                          & (moved[:, 0, 1] >= 0) & (moved[:, 0, 1] < gray.shape[0]))
                track['points'] = moved[valid]
                if len(track['points']) >= self.MIN_POINTS:
                    kept.append(track)
            self.tracks = kept
        self.gray, self.timestamp = gray, timestamp

    def identify(self, mask, key):
        """Associate only a single supported foreground identity; ambiguity abstains."""
        if self.gray is None or mask is None or mask.shape != self.gray.shape or not key:
            return None, {'reason': 'no_source_mask_or_metadata'}
        seeds = cv2.goodFeaturesToTrack(self.gray, maxCorners=100, qualityLevel=.01,
                                      minDistance=5, mask=mask, blockSize=5)
        seed_count = 0 if seeds is None else len(seeds)
        matches = []
        partial_support = []
        # A small border tolerance addresses subpixel motion/segmentation rounding.
        support = cv2.dilate(mask, np.ones((3, 3), dtype='uint8'))
        for track in self.tracks:
            if track['key'] != key:
                continue
            xy = np.rint(track['points'][:, 0]).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1]-1)
            xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0]-1)
            hits = track['points'][support[xy[:, 1], xy[:, 0]] > 0]
            if len(hits) < self.MIN_POINTS:
                continue
            # Judge coverage of the currently visible foreground. A partial
            # view of a pair need not contain most points on BOTH shoes.
            # Require distributed support across this mask, not a small patch.
            area = cv2.contourArea(cv2.convexHull(hits))
            if area < .3 * max(1, np.count_nonzero(mask)):
                # Evidence for a later appearance check, NEVER an identity merge
                # on its own. These trajectories still survived every frame.
                partial_support.append({'identity': track['id'], 'points': len(hits),
                    'coverage': area / max(1, np.count_nonzero(mask))})
                continue
            matches.append((track, hits))
        if len(matches) == 1:
            track, hits = matches[0]
            # Replenish only inside the supported point hull. Do not recruit a
            # separate object elsewhere in a broad segmentation mask.
            hull_mask = np.zeros_like(mask)
            cv2.fillConvexPoly(hull_mask, cv2.convexHull(hits).astype('int32'), 255)
            refresh = cv2.goodFeaturesToTrack(self.gray, maxCorners=100, qualityLevel=.01,
                minDistance=5, mask=cv2.bitwise_and(mask, hull_mask), blockSize=5)
            # Existing successful trajectories remain evidence even when a
            # corner detector no longer picks them in this particular view.
            observed_fraction = len(hits) / len(track['points'])
            points = track['points']
            if refresh is not None:
                for point in refresh:
                    if len(points) >= 100:
                        break
                    if np.min(np.linalg.norm(points[:, 0] - point[0], axis=1)) >= 5:
                        points = np.concatenate((points, point.reshape(1, 1, 2)))
            track['points'] = points
            return track['id'], {'reason': 'foreground_flow', 'supported_points': len(hits),
                                  'partial_support': partial_support,
                                  'observed_fraction': observed_fraction,
                                  'foreground_coverage': cv2.contourArea(cv2.convexHull(hits)) / max(1, np.count_nonzero(mask)),
                                  'seed_frame_seconds': track['started'], 'observed_until_seconds': self.timestamp}
        if matches:
            return None, {'reason': 'ambiguous_source_tracks', 'matches': len(matches)}
        if seed_count < self.MIN_POINTS:
            return None, {'reason': 'insufficient_source_points', 'points': seed_count}
        identity = self.next_id
        self.next_id += 1
        self.tracks.append({'id': identity, 'key': key, 'points': seeds, 'started': self.timestamp})
        self.tracks = self.tracks[-40:]
        return identity, {'reason': 'new_source_track', 'points': seed_count,
                          'partial_support': partial_support}


def reconcile_backward(video_path, candidates, frame_count, fps):
    """Join forward/reverse paths only where they share an observed candidate.

Decode backwards in bounded 16-frame blocks, keeping at most 16 downsized
frames. No removal/classification, new candidate, or similarity inference.
Both passes terminate identity on loss of foreground trajectories.
"""
    by_frame = {}
    for candidate in candidates:
        if (candidate.get('category') == 'shoes' and candidate.get('_source_mask') is not None
                and not candidate.get('_identity_unverified')):
            by_frame.setdefault(candidate['_decoded_frame_index'], []).append(candidate)
    if not by_frame:
        return
    flow = SourceContinuity()
    parent = {}

    def root(token):
        parent.setdefault(token, token)
        while parent[token] != token:
            parent[token] = parent[parent[token]]
            token = parent[token]
        return token

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError('Could not reopen saved video for continuity')
    try:
        end = frame_count
        while end > 0:
            start = max(0, end-16)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for index in range(start, end):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Incomplete reverse continuity decode')
                height, width = frame.shape[:2]
                frames.append(cv2.resize(frame, working_size(width, height)))
            for offset in range(len(frames)-1, -1, -1):
                index = start + offset
                flow.advance(frames[offset], (frame_count-1-index)/fps)
                for candidate in by_frame.get(index, []):
                    backward, evidence = flow.identify(candidate['_source_mask'], candidate['_source_family'])
                    forward = candidate.get('_source_continuity')
                    paths = [('forward', forward)] if forward is not None else []
                    if backward is not None:
                        paths.append(('backward', backward))
                    candidate['_source_paths'] = paths
                    if paths:
                        for token in paths:
                            parent[root(token)] = root(paths[0])
                    candidate['_source_continuity_evidence'] = {
                        'reason': 'bidirectional_foreground_flow',
                        'forward': candidate.get('_source_continuity_evidence', {}), 'backward': evidence}
                    if evidence.get('observed_fraction', 1) < .75:
                        candidate['_source_clipped'] = True
            end = start
    finally:
        cap.release()
    for candidate in candidates:
        paths = candidate.get('_source_paths', [])
        if paths:
            candidate['_source_continuity'] = root(paths[0])
            evidence = candidate['_source_continuity_evidence']
            evidence['supported_points'] = max(evidence['forward'].get('supported_points', 0),
                                               evidence['backward'].get('supported_points', 0))
            candidate['_source_partial_continuity'] = {
                root((direction, support['identity']))
                for direction in ('forward', 'backward')
                for support in evidence[direction].get('partial_support', [])}
