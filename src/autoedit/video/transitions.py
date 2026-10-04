"""Frame-level visual discontinuities, independent of face identity."""


class TransitionDetector:
    def __init__(self):
        from collections import deque
        self.previous = None
        self.events = []
        self.shot_events = []
        self.previous_histogram = None
        self.previous_gray = None
        self.history = deque()

    def observe(self, time, frame):
        import cv2
        import numpy as np

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
        while self.history and time - self.history[0][0] > .5:
            self.history.popleft()
        if self.previous is not None:
            delta = np.abs(hsv - self.previous)
            delta[..., 0] = np.minimum(delta[..., 0], 180 - delta[..., 0]) * 2
            score = float(delta.mean() / 255)
            # Deliberately conservative: large flashes also make a cut unsuitable.
            motion = score >= .08 and continuous_motion(self.previous_gray, gray)
            if score >= .08 and not motion:
                self.events.append(float(time))
                # A known cut must not poison half-second comparisons after
                # it. Comparing across it used to mark ordinary frames in
                # the following shot as additional transitions.
                self.history.clear()
            # Safety events also include motion/dissolves; only abrupt distribution
            # changes contribute to the independent shot-occurrence count.
            histogram_change = cv2.compareHist(self.previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA)
            if score >= .12 and histogram_change >= .45 and not motion:
                self.shot_events.append(float(time))
        # Dissolves can be almost invisible between adjacent frames. Compare
        # only within this shot, never across an already detected hard cut.
        if self.history and time - self.history[0][0] >= .4:
            old_time, old, old_histogram, old_gray = self.history[0]
            delta = np.abs(hsv - old)
            delta[..., 0] = np.minimum(delta[..., 0], 180 - delta[..., 0]) * 2
            distribution_change = cv2.compareHist(old_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA)
            if (float(delta.mean() / 255) >= .08 and distribution_change >= .3
                    and not continuous_motion(old_gray, gray)):
                self.events.extend((old_time, float(time)))
        self.previous = hsv
        self.previous_histogram = histogram
        self.previous_gray = gray
        self.history.append((float(time), hsv, histogram, gray))


def continuous_motion(previous, current):
    """Verify distributed, reversible feature tracks before suppressing a cut.

    A pan or moving actor can change most pixels without changing the shot.
    Repeated texture alone is insufficient: tracks must agree geometrically,
    span the frame and retain their appearance. Flashes/flat dissolves abstain.
    """
    import cv2
    import numpy as np
    points = cv2.goodFeaturesToTrack(previous, 100, .02, 4)
    if points is None or len(points) < 12:
        return False
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if moved is None:
        return False
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None)
    if back is None:
        return False
    height, width = current.shape
    valid = ((status[:, 0] != 0) & (back_status[:, 0] != 0)
             & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.)
             & (moved[:, 0, 0] >= 1) & (moved[:, 0, 0] < width-1)
             & (moved[:, 0, 1] >= 1) & (moved[:, 0, 1] < height-1))
    if valid.sum() < 12 or valid.mean() < .35:
        return False
    transform, inliers = cv2.estimateAffinePartial2D(points[valid], moved[valid],
        method=cv2.RANSAC, ransacReprojThreshold=2.)
    if transform is None or inliers.mean() < .55:
        return False
    scale = float(np.linalg.norm(transform[:, 0]))
    if not .75 <= scale <= 1.33:
        return False
    original = points[valid, 0][inliers[:, 0] != 0]
    destination = moved[valid, 0][inliers[:, 0] != 0]
    if np.ptp(original[:, 0]) * np.ptp(original[:, 1]) < width*height*.3:
        return False
    # Compare small patches to avoid accepting a dissolve or exposure flash
    # just because a few underlying edges still track.
    differences = [abs(float(cv2.getRectSubPix(previous, (3, 3), tuple(a)).mean())
                       - float(cv2.getRectSubPix(current, (3, 3), tuple(b)).mean()))
                   for a, b in zip(original, destination)]
    return bool(np.median(differences) < 20 and np.quantile(differences, .8) < 40)


def transition_metrics(events, start, end, margin=.12):
    """Protect 3s and all but the final 25% (at most 2s) of longer cuts."""
    from bisect import bisect_left, bisect_right

    first = bisect_left(events, start - margin)
    last = bisect_right(events, end + margin)
    nearby = events[first:last]
    protected = min(end - start, max(3.0, end - start - min(2.0, (end - start) * .25)))
    inside = [t for t in nearby if start < t < end]
    allowed = not any(abs(t - start) <= margin or abs(t - end) <= margin
                      or start < t < start + protected for t in nearby)
    return {"transition_count": len(inside), "stable_opening_sec": protected,
            "transition_safe": allowed and len(inside) <= 1}
