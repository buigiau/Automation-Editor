"""Frame-level visual discontinuities, independent of face identity."""


class TransitionDetector:
    def __init__(self):
        from collections import deque
        self.previous = None
        self.events = []
        self.shot_events = []
        self.previous_histogram = None
        self.history = deque()

    def observe(self, time, frame):
        import cv2
        import numpy as np

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        histogram = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
        while self.history and time - self.history[0][0] > .5:
            self.history.popleft()
        # Dissolves can be almost invisible between adjacent frames. Compare
        # across half a second too, marking both ends of the uncertain interval.
        if self.history and time - self.history[0][0] >= .4:
            old_time, old = self.history[0]
            delta = np.abs(hsv - old)
            delta[..., 0] = np.minimum(delta[..., 0], 180 - delta[..., 0]) * 2
            if float(delta.mean() / 255) >= .08:
                self.events.extend((old_time, float(time)))
        if self.previous is not None:
            delta = np.abs(hsv - self.previous)
            delta[..., 0] = np.minimum(delta[..., 0], 180 - delta[..., 0]) * 2
            score = float(delta.mean() / 255)
            # Deliberately conservative: large flashes also make a cut unsuitable.
            if score >= .08:
                self.events.append(float(time))
            # Safety events also include motion/dissolves; only abrupt distribution
            # changes contribute to the independent shot-occurrence count.
            histogram_change = cv2.compareHist(self.previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA)
            if score >= .12 and histogram_change >= .45:
                self.shot_events.append(float(time))
        self.previous = hsv
        self.previous_histogram = histogram
        self.history.append((float(time), hsv))


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
