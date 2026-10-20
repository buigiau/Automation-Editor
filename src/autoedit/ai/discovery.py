"""Incremental shot discovery and compact candidate classification for early exit."""
from copy import deepcopy
import math
from pathlib import Path
import time

from autoedit.runtime import check_cancelled
from .cache import digest, read, write
from .local import allocate
from .ollama import LocalAIError
from .selection import used_end

VERSION = 'candidate-first-31-source-audio'


def evidence_duration(slot):
    return (min(1., slot['required_duration_sec']) if slot.get('allow_scene_changes') else
            slot.get('visible_duration_sec', slot['required_duration_sec']))


def discovery_settings(ai):
    mode = ai.get('local_analysis_mode', 'until_filled')
    if mode not in ('until_filled', 'full'):
        raise ValueError('ai.local_analysis_mode must be until_filled or full')
    budget = ai.get('local_target_seconds', 300)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('ai.local_target_seconds must be a positive finite number')
    maximum = ai.get('local_max_seconds', 0)
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or maximum < 0:
        raise ValueError('ai.local_max_seconds must be a nonnegative finite number')
    return {'mode': mode, 'target_seconds': float(budget), 'max_seconds': float(maximum),
            'discovery_version': VERSION, 'region_order_version': 'near-and-far-1'}


def candidate_stream(source, fps, longest, cache, deadline, log, minimum=None, source_gap=5., quick=False,
                     tail_fallback_sec=0):
    """Decode forward only as far as requested; no whole-source index prerequisite."""
    import av
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(2)
    # Candidate frames are independent of model and job-output directory.
    folder = Path(cache) / 'local/candidate_frames' / digest([VERSION, source['identity'], fps, quick])
    folder.mkdir(parents=True, exist_ok=True)
    previous, histogram = None, None
    start = source.get('analyzed_start_sec', 0.)
    region_start = start
    limit = source.get('analyzed_end_sec', source['duration_sec'])
    samples, shot, next_sample = [], 0, start
    last_time, last_log = start, time.monotonic()
    next_detection = start
    minimum = longest if minimum is None else minimum
    emitted_starts = []
    rolling_length = max(4., longest+.3)
    next_window_start = start+.08

    def window(a, b, end, complete):
        midpoint = (a+b)/2
        t, image = min(samples, key=lambda item: abs(item[0]-midpoint))
        name = folder / f'shot-{shot:07d}-{round(a*1000):010d}-{round(b*1000):010d}.jpg'
        Image.fromarray(image).save(name, quality=85)
        result = {'source_id': source['id'], 'shot': shot,
            'shot_start_sec': start, 'shot_end_sec': end,
            'start_sec': a, 'end_sec': b, 'image': str(name),
            'image_time_sec': t, 'alternative': bool(emitted_starts),
            'shot_bounds_complete': complete and not quick and (region_start == 0 or start > region_start)}
        emitted_starts.append(a)
        return result

    def finish(end):
        if not samples:
            return []
        found = []
        if end-start >= minimum+.2:
            # Keep the whole observed interval. Capping this to one fill's
            # length discarded valid middle cuts after unsafe edge windows.
            # Rolling windows still allow early exit before the shot ends.
            found.append(window(start+.08, end-.08, end, True))
        tail_end = min(limit, end+tail_fallback_sec)
        if (tail_fallback_sec and end < limit and
                tail_end-start-.16 >= longest+.16-1e-6):
            tail_length = min(tail_end-start-.16, longest+.3)
            a = max(start+.08, tail_end-.08-tail_length)
            extended = window(a, tail_end-.08, end, False)
            extended['tail_candidate'] = True
            found.append(extended)
        return found

    with av.open(source['path']) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if quick:
            # Coarse discovery only. Every allocated cut still undergoes the
            # unchanged native-frame transition and duplicate audits.
            stream.codec_context.skip_frame = 'NONREF'
        stream.thread_type = 'AUTO'
        origin = float((stream.start_time or 0)*stream.time_base)
        if region_start > 0:
            container.seek(int((region_start+origin)/float(stream.time_base)), stream=stream, backward=True)
        native_fps = float(stream.average_rate or 25)
        for index, frame in enumerate(container.decode(stream)):
            check_cancelled()
            if time.monotonic() >= deadline:
                break
            t = float(frame.time)-origin if frame.time is not None else index/native_fps
            if t < region_start:
                continue
            if t >= limit:
                last_time = limit
                break
            last_time = min(limit, t+1/native_fps)
            # Discovery needs cheap coarse evidence, while transition_safe()
            # still audits every native frame of every proposed cut.
            if quick:
                if t+1e-6 < next_detection:
                    continue
                next_detection = t+1/max(6., fps*2)
            small = frame.reformat(width=160, height=90, format='rgb24').to_ndarray()
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
            cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
            cut = previous is not None and float(np.abs(small.astype(np.int16)-previous).mean())/255 >= .12 and cv2.compareHist(histogram, hist, cv2.HISTCMP_BHATTACHARYYA) >= .45
            if cut:
                windows = finish(t)
                # Discard the previous shot's sample arrays before suspending.
                samples = []
                start, shot = t, shot+1
                emitted_starts, next_window_start = [], t+.08
                for candidate in windows:
                    yield candidate
            previous, histogram = small.astype(np.int16), hist
            if t+.000001 >= next_sample:
                width = min(384, frame.width)
                height = max(2, round(frame.height*width/frame.width))
                image = frame.reformat(width=width, height=height, format='rgb24').to_ndarray()
                samples.append((t, image))
                # Bound memory even for a multi-minute continuous shot.
                if len(samples) > 96:
                    samples = samples[::2]
                next_sample = (int((t+.000001)*fps)+1)/fps
            last_time = min(limit, t+1/native_fps)
            # A long continuous take must not force a scan to its very end.
            # Bounds are the observed shot prefix; the stable shot ID still
            # prevents reuse until the explicit same-shot fallback is enabled.
            if samples and t >= next_window_start+rolling_length+.08:
                yield window(next_window_start, next_window_start+rolling_length, t, False)
                next_window_start += rolling_length+source_gap+.2
            if time.monotonic()-last_log > 10:
                log(f"Candidate search {source['id']}: {t:.1f}s; stop as soon as slots are filled")
                last_log = time.monotonic()
        for candidate in finish(last_time):
            if limit < source['duration_sec']:
                candidate['shot_bounds_complete'] = False
            yield candidate


def region_order(count):
    """Visit distant parts early, then progressively fill every remaining gap."""
    bits = max(0, (count-1).bit_length())
    return sorted(range(count), key=lambda i: int(f'{i:0{bits}b}'[::-1], 2)) if bits else [0]


def distributed_candidates(source, fps, longest, cache, deadline, log, minimum=None, source_gap=5., screen=None, quick=False,
                           needed_duration=None, tail_fallback_sec=0, solo_deadline=float('inf'), eager=False, focus_seconds=None):
    """Seek across the source; retain unused windows for exhaustive fallback."""
    start = source.get('analyzed_start_sec', 0.)
    end = source.get('analyzed_end_sec', source['duration_sec'])
    # A shot long enough for a fill can straddle any region boundary. Include
    # a full fill of right-hand context instead of truncating it at that edge.
    span = max(12., longest*3+source_gap)
    overlap = longest+.8
    count = max(1, math.ceil((end-start)/span))
    deferred = []
    last_log = time.monotonic()
    order, visited = region_order(count), 0
    focused_visits = 0
    while order:
        nearby = focus_seconds() if focus_seconds else []
        candidates = [i for i in order if any(abs(start+(i+.5)*span-t) <= max(120., span*3) for t in nearby)]
        if candidates and focused_visits < 3:
            index = min(candidates, key=lambda i: min(abs(start+(i+.5)*span-t) for t in nearby))
            focused_visits += 1
        else:
            # Recent matches guide nearby discovery, but must not starve the
            # distant regions when the same look disappears from this block.
            index = order[0]
            focused_visits = 0
        order.remove(index)
        visited += 1
        if time.monotonic() >= deadline:
            return
        if deferred and time.monotonic() >= solo_deadline:
            pending, deferred = deferred, []
            for window in pending:
                if not needed_duration or window['end_sec']-window['start_sec'] >= needed_duration()+.16-1e-6:
                    yield window
        a, b = start+index*span, min(end, start+(index+1)*span+overlap)
        region = {**source, 'analyzed_start_sec': a, 'analyzed_end_sec': b}
        record = Path(cache)/'local/candidate_regions'/(
            digest([VERSION, region, fps, longest, minimum, source_gap, quick, tail_fallback_sec])+'.json')
        saved = read(record)
        if saved is not None and all(w.get('image') and Path(w['image']).is_file() for w in saved):
            windows = saved
        else:
            windows = list(candidate_stream(region, fps, longest, cache, deadline, log,
                minimum=minimum, source_gap=source_gap,
                **({'tail_fallback_sec': tail_fallback_sec} if tail_fallback_sec else {}),
                **({'quick': True} if quick else {})))
            if time.monotonic() < deadline:
                write(record, windows)
        # Once short fills are satisfied, do not run person/face inference on
        # hundreds of short shots that cannot possibly fill the remaining slots.
        required = needed_duration() if needed_duration else minimum
        if required is not None:
            windows = [w for w in windows if w['end_sec']-w['start_sec'] >= required+.16-1e-6]
        # Prefer long shots so short fills cannot exhaust the long-shot supply.
        if screen is not None:
            for window in windows:
                check_cancelled()
                if time.monotonic() >= deadline:
                    return
                window['subject_preview'] = screen(window['image'])
        if time.monotonic()-last_log >= 15:
            log(f"Candidate regions {source['id']}: {visited}/{count}; "
                f"remaining fills need at least {required or 0:.2f}s")
            last_log = time.monotonic()
        if eager:
            # Preserve scarce long takes before nearby short fills consume the
            # source-gap space needed by them. Colour score is only a ranking
            # aid, never a reason to spend a long take on a short fill first.
            windows.sort(key=lambda w: (w.get('subject_preview', {}).get('priority', 0),
                -(w['end_sec']-w['start_sec']),
                -(w.get('subject_preview', {}).get('reference_appearance_score') or 0), w['start_sec']))
        else:
            windows.sort(key=lambda w: (w.get('subject_preview', {}).get('priority', 0),
                -w.get('subject_preview', {}).get('face_area', 0),
                -(w['end_sec']-w['start_sec']), w['start_sec']))
        scanned_end = max((w['shot_end_sec'] for w in windows), default=a)
        for window in windows:
            # Partial shots across seeks must not be invented as distinct shots.
            # The shared unknown ID stays conservative until same-shot fallback.
            window['shot'] = ('unknown' if window['shot_start_sec'] == a and a > 0
                              else f"at-{round(window['shot_start_sec']*1000)}")
            window['search_region'] = [a, scanned_end]
        if windows:
            immediate = [w for w in windows if w.get('subject_preview', {}).get('priority') == 0] if screen else windows if eager else []
            if immediate:
                for window in immediate:
                    if needed_duration and window['end_sec']-window['start_sec'] < needed_duration()+.16-1e-6:
                        continue
                    yield window
                deferred.extend(w for w in windows if w not in immediate)
            else:
                yield windows[0]
                deferred.extend(windows[1:])
    for window in deferred:
        if time.monotonic() >= deadline:
            return
        if needed_duration and window['end_sec']-window['start_sec'] < needed_duration()+.16-1e-6:
            continue
        yield window


def sheet_for(windows, path):
    from PIL import Image, ImageDraw, ImageOps
    columns = min(2, len(windows))
    canvas = Image.new('RGB', (columns*320, math.ceil(len(windows)/columns)*202), '#151515')
    draw = ImageDraw.Draw(canvas)
    for index, window in enumerate(windows):
        x, y = index%columns*320, index//columns*202
        with Image.open(window['image']) as original:
            image = ImageOps.contain(original.convert('RGB'), (320, 180))
        canvas.paste(image, (x+(320-image.width)//2, y+22))
        draw.text((x+5, y+3), f'#{index}', fill='white')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=88)


def classify(windows, contract, client, cache, registry, deadline, log):
    from .single import classify as classify_single
    evidence = [{**w, 'end_sec': w['end_sec']-w.get('transition_tail_fallback_sec', 0)} for w in windows]
    return classify_single(evidence, contract, client, cache, registry, deadline, log)


class ProgressiveSelector:
    def __init__(self, client, cache, log):
        self.client, self.cache, self.log = client, Path(cache), log
        self.streams = None
        self.windows, self.runs, self.registry = [], [], {}
        self.deadline = None
        self.searched = {}
        self.regions = {}
        self.classified_count = 0
        self.model_candidate_count = 0
        self.best = None
        self.target_logged = False
        self.people = None
        self.group_phase = False
        self.group_windows = []
        self.native_retry = False
        self.vision_windows = []
        self.vision_phase = self.vision_done = False
        self.allocated_run_counts = {}
        self.audio_deferred = {}
        self.audio_reassign = False

    def defer_audio(self, window):
        if getattr(self, 'source_session', None) and not self.audio_reassign and window.get('source_sound'):
            key = (window['source_id'], window['sound_slot_id'], window['source_sound']['start'])
            self.audio_deferred[key] = window

    def needed_duration(self, contract):
        kept = {r['slot_id'] for r in (self.best or {}).get('selections', [])}
        lengths = [evidence_duration(s) for s in contract['slots'] if s['id'] not in kept]
        long_first = contract['references'] and time.monotonic() < getattr(self, 'reference_long_deadline', float('inf'))
        return max(lengths, default=0.) if long_first else min(lengths, default=0.)

    def source_streams(self, contract, quick):
        if getattr(self, 'source_session', None) is not None:
            if getattr(self, '_audio_stream_started', False):
                return []
            self._audio_stream_started = True
            return [iter(self.source_session.windows(contract))]
        return [distributed_candidates(s, contract['video_sampling']['video_fps'], self.longest,
            self.cache, self.deadline, self.log, minimum=self.minimum,
            source_gap=contract['source_gap_sec'], needed_duration=lambda: self.needed_duration(contract),
            tail_fallback_sec=contract['video_sampling'].get('transition_tail_fallback_sec', 0),
            solo_deadline=self.target_deadline, eager=bool(contract['references']), quick=quick,
            focus_seconds=(lambda source_id=s['id']: self.reference_focus(source_id)) if contract['references'] else None,
            **({'screen': self.people.preview} if self.people else {})) for s in contract['sources']]

    def reference_focus(self, source_id):
        matches = [(r['start_sec']+r.get('identity_end_sec', r['end_sec']))/2 for r in self.runs
                   if r['source_id'] == source_id and self.registry[r['character_id']]['reference_ids']][-3:]
        return matches

    def timed_group_fallback(self):
        if (self.group_phase or not self.group_windows or
                time.monotonic() < getattr(self, 'target_deadline', float('inf'))):
            return
        self.group_phase = True
        self.vision_phase = False
        self.log('Solo search reached the time target; checking deferred group scenes for remaining fills.')
        pending = sorted(self.group_windows, key=lambda w: -(w['end_sec']-w['start_sec']))
        self.group_windows = []
        self.streams.insert(0, iter(pending))

    def result(self, contract, previous, errors, allow_same_shot):
        if (self.best is not None and time.monotonic() >= self.deadline and
                (time.monotonic() >= getattr(self, 'allocation_deadline', self.deadline) or
                 self.allocated_run_counts.get(allow_same_shot, -1) >= len(self.runs))):
            result = deepcopy(self.best)
            rejected = {e['slot_id'] for e in errors or []}
            result['selections'] = [r for r in result['selections'] if r['slot_id'] not in rejected]
            result.setdefault('analysis_scope', {}).update(stop_reason='time_budget',
                searched_until_sec=deepcopy(self.searched), candidate_count=self.classified_count,
                search_regions=deepcopy(self.regions))
            if len(result['selections']) < len(contract['slots']):
                result['shortage_reason'] = f"Search target reached with {len(result['selections'])}/{len(contract['slots'])} valid scenes."
            return result
        characters = []
        for cid, char in self.registry.items():
            appearances = [{'source_id': r['source_id'], 'start_sec': r['start_sec'],
                            'end_sec': r.get('identity_end_sec', r['end_sec'])}
                           for r in self.runs if r['character_id'] == cid]
            if appearances:
                characters.append({**char, 'description': cid, 'appearances': appearances})
        eligible_runs = [r for r in self.runs if self.group_phase or r.get('subject_count', 1) == 1]
        if contract['references'] and (self.streams or getattr(self.client, 'reference_appearance', None)):
            # Do not fill requested characters with unrelated actors merely to
            # meet the time target before the available source is exhausted.
            targeted = {c['id'] for c in characters if c['reference_ids']}
            characters = [c for c in characters if c['id'] in targeted]
            eligible_runs = [r for r in eligible_runs if r['character_id'] in targeted]
        prior = deepcopy(previous or self.best)
        if getattr(self, 'source_session', None) and (self.source_session.full or self.audio_reassign):
            prior = None
        if prior and not self.group_phase:
            prior['selections'] = [r for r in prior['selections'] if r.get('subject_count', 1) == 1]
        result = allocate(contract, characters, eligible_runs, self.cache, previous=prior, errors=errors,
                          allow_same_shot=allow_same_shot,
                          deadline=getattr(self, 'allocation_deadline', self.deadline))
        self.allocated_run_counts[allow_same_shot] = len(self.runs)
        result.update(_autoedit_dedup_policy='distinct_segments' if allow_same_shot else 'unique_shots',
            analysis_scope={'mode': 'until_filled', 'searched_until_sec': self.searched,
                'character_ranking_scope': 'classified_candidate_windows',
                'candidate_count': self.classified_count, 'visibility_is_spot_checked': True,
                'model_candidate_count': self.model_candidate_count,
                'search_strategy': 'distributed_regions', 'search_regions': deepcopy(self.regions),
                'target_exceeded': time.monotonic() >= getattr(self, 'target_deadline', float('inf')),
                'hard_max_seconds': contract['local_analysis'].get('max_seconds', 0),
                'shot_bounds_may_be_partial': True,
                'solo_first': not bool(getattr(self.client, 'reference_appearance', None)), 'group_fallback_enabled': self.group_phase,
                'native_discovery_retry': self.native_retry,
                'fast_people': deepcopy(self.people.stats) if self.people else None,
                'reference_match_scope': 'costume_appearance' if getattr(self.client, 'reference_appearance', None) else None,
                'stop_reason': ('enough_slots' if len(result['selections']) == len(contract['slots']) else
                    'time_budget' if time.monotonic() >= self.deadline else
                    'source_exhausted' if not self.streams else 'searching'),
                'target_seconds': contract['local_analysis']['target_seconds']})
        for row in result['selections']:
            row['local_character_verified'] = False
            if getattr(self.client, 'single_image', False):
                row['character_sample_frames'] = row.get('character_sample_frames', 2)
                row['identity_sample_consistent'] = row.get('identity_sample_consistent', True)
                if row['identity_sample_consistent']:
                    row['reason'] += f" Appearance agreed at {row['character_sample_frames']} sampled cut frames; inspect full playback."
            else:
                row['reason'] += ' Character identity inferred from candidate thumbnail; inspect the full cut.'
        def quality(value):
            rows = value['selections']
            return (len(rows), sum(r.get('audio_match', {}).get('word_probability', 0) for r in rows))
        if self.best is None or quality(result) > quality(self.best):
            self.best = deepcopy(result)
        return result

    def select(self, contract, previous=None, errors=None, allow_same_shot=False):
        if not getattr(self.client, 'single_image', False):
            self.client.check_resources()
        if self.streams is None:
            budget = contract['local_analysis']['target_seconds']
            maximum = contract['local_analysis'].get('max_seconds', 0)
            self.deadline = min(time.monotonic()+maximum if maximum else float('inf'),
                getattr(self.client, 'job_deadline', float('inf')))
            if getattr(self, 'source_session', None) is not None:
                self.deadline = min(self.deadline, self.source_session.deadline)
            self.allocation_deadline = min(time.monotonic()+maximum if maximum else float('inf'),
                getattr(self.client, 'finalization_deadline', self.deadline))
            self.target_deadline = getattr(self.client, 'target_deadline', time.monotonic()+budget)
            # Bound the long-only pass so a difficult long fill cannot prevent
            # useful partial results for every short slot at the hard deadline.
            self.reference_long_deadline = min(self.deadline, time.monotonic()+min(75., budget*.25))
            longest = max(evidence_duration(s) for s in contract['slots'])
            minimum = min(evidence_duration(s) for s in contract['slots'])
            self.longest, self.minimum = longest, minimum
            if getattr(self.client, 'single_image', False):
                from .people import FastPeople
                try:
                    self.people = FastPeople(enable_identity=False) if contract['references'] else FastPeople()
                    self.client.people = self.people
                    if contract['references']:
                        from .appearance import ReferenceAppearance
                        appearance = ReferenceAppearance(contract['references'], self.people, self.cache)
                        if appearance.human_references:
                            self.people.reference_appearance = appearance
                            self.client.reference_appearance = appearance
                            self.group_phase = True
                            self.log('Reference appearance: clothing/accessories only; faces masked, no facial identity matching.')
                        else:
                            self.people.close()
                            self.people = self.client.people = None
                    if getattr(self.client, 'reference_appearance', None):
                        self.log('CPU wardrobe screening enabled; local vision compares masked reference appearances, including group scenes.')
                    else:
                        self.log('CPU fast people enabled: count people/animals and verify aligned faces; single-person cuts first.')
                except (FileNotFoundError, ImportError) as exc:
                    self.log(f'CPU people screening unavailable: {exc}; using local vision with subject counts.')
            self.streams = self.source_streams(contract, quick=bool(self.people or contract['references']))
            self.reference_stage_duration = self.needed_duration(contract)
        size = 1 if getattr(self.client, 'single_image', False) else min(6, len(contract['slots']))
        while True:
            check_cancelled()
            if (not self.streams and self.audio_deferred and not self.audio_reassign
                    and time.monotonic() < self.deadline):
                self.audio_reassign = True
                self.streams = [iter(self.audio_deferred.values())]
                self.log('Reassigning deferred source onsets before using neutral fallback; no ASR rescan.')
            self.timed_group_fallback()
            needed = self.needed_duration(contract)
            if contract['references'] and needed < self.reference_stage_duration-1e-6:
                self.reference_stage_duration = needed
                self.streams = self.source_streams(contract, quick=not self.native_retry)
                self.log(f'Reference search: revisiting cached regions for {needed:.2f}s evidence.')
            if not self.streams and self.people and not self.native_retry and time.monotonic() < self.deadline:
                self.native_retry = True
                self.log('Fast search short; retrying native-frame discovery for remaining fills.')
                self.streams = self.source_streams(contract, quick=False)
                continue
            if not self.streams and self.vision_windows and not self.vision_done:
                self.vision_phase = self.vision_done = True
                self.log('Trying CPU-unrecognized candidates with Qwen.')
                self.streams = [iter(self.vision_windows)]
                self.vision_windows = []
                continue
            if not self.streams and not self.group_phase:
                self.vision_phase = False
                self.group_phase = True
                self.log('Single-character candidate search exhausted; enabling group-scene shortage fallback.')
                if self.group_windows:
                    self.streams = [iter(self.group_windows)]
                    self.group_windows = []
            if not self.target_logged and time.monotonic() >= getattr(self, 'target_deadline', float('inf')):
                self.log('Time target passed; continuing to find enough verified, distinct footage. Cancel remains available.')
                self.target_logged = True
            if self.runs:
                result = self.result(contract, previous, errors, allow_same_shot)
                if len(result['selections']) == len(contract['slots']) and not (
                        getattr(self, 'source_session', None) and self.source_session.full):
                    self.log(f"Enough footage: {len(result['selections'])}/{len(contract['slots'])}; stopping source search.")
                    return result
                needed = self.needed_duration(contract)
                if contract['references'] and needed < self.reference_stage_duration-1e-6 and not getattr(self, 'source_session', None):
                    # Revisit cached regions after filling the scarce long cuts;
                    # their skipped short windows remain useful for later slots.
                    self.reference_stage_duration = needed
                    self.streams = self.source_streams(contract, quick=not self.native_retry)
                    self.log(f'Reference search: revisiting cached regions for {needed:.2f}s evidence.')
                self.log(f"Validated footage: {len(result['selections'])}/{len(contract['slots'])}; searching other regions.")
                if contract['video_sampling']['shot_fallback'] and not allow_same_shot:
                    fallback = self.result(contract, previous, errors, True)
                    if len(fallback['selections']) > len(result['selections']):
                        self.log(f"Validated distinct segments with shortage fallback: {len(fallback['selections'])}/{len(contract['slots'])}.")
                    if len(fallback['selections']) == len(contract['slots']) and not (
                            getattr(self, 'source_session', None) and self.source_session.full):
                        self.log('Enough distinct segments with same-shot shortage fallback; stopping source search.')
                        return fallback
            if time.monotonic() >= self.deadline or not self.streams:
                return self.result(contract, previous, errors, allow_same_shot)
            batch = []
            from .local import transition_safe
            from .visual import fingerprints, similar
            kept = ([] if getattr(self, 'source_session', None) and (self.source_session.full or self.audio_reassign)
                    else (self.best or {}).get('selections', []))
            missing = [s for s in sorted(contract['slots'], key=lambda s: (-evidence_duration(s), -s['required_duration_sec']))
                       if s['id'] not in {r['slot_id'] for r in kept}]
            available_slots = list(missing)
            occupied = list(kept)
            sources = {s['id']: s for s in contract['sources']}
            samples = [fingerprints(sources[r['source_id']], r, self.cache) for r in kept]
            while len(batch) < size and self.streams and time.monotonic() < self.deadline:
                self.timed_group_fallback()
                stream = self.streams[0]
                try:
                    window = next(stream)
                except StopIteration:
                    self.streams.pop(0)
                    continue
                if len(self.streams) > 1:
                    self.streams.append(self.streams.pop(0))
                sid = window['source_id']
                self.searched[sid] = max(self.searched.get(sid, 0), window['shot_end_sec'])
                region = window.get('search_region', [window['shot_start_sec'], window['shot_end_sec']])
                if region not in self.regions.setdefault(sid, []):
                    self.regions[sid].append(region)
                preview = window.get('subject_preview', {})
                appearance_score = preview.get('reference_appearance_score')
                if appearance_score is not None and appearance_score < .25:
                    # A cheap wardrobe screen prevents full Qwen requests for
                    # clearly different looks. It does not establish identity.
                    continue
                if not self.group_phase and (preview.get('subject_count') or 0) > 1:
                    self.group_windows.append(window)
                    continue
                if (self.people and self.people.human_mode and not self.vision_phase
                        and not self.group_phase and preview.get('priority') == 2):
                    # Zero human detections say nothing about animated objects.
                    # A single false human positive must not discard a cartoon.
                    self.vision_windows.append(window)
                    continue
                # Reject unusable footage before paying for vision inference.
                slot = next((s for s in available_slots if
                    (not window.get('sound_slot_id') or window['sound_slot_id'] == s['id']) and
                    window['end_sec']-window['start_sec'] >= evidence_duration(s)+.16-.000001), None)
                if slot is None:
                    self.defer_audio(window)
                    continue
                duration = slot['required_duration_sec']
                evidence_length = evidence_duration(slot)
                final_fallback = bool(slot.get('allow_scene_changes'))
                visible_prefix = not final_fallback and evidence_length < duration-.002
                center = (window['start_sec']+window['end_sec']-evidence_length)/2
                lo, hi = window['start_sec']+.08, window['end_sec']-evidence_length-.08
                starts = dict.fromkeys(round(max(lo, min(hi, t)), 9) for t in
                                       (center, center-.3, center+.3, lo, hi))
                if window.get('source_sound'):
                    starts = [window['source_sound']['start']]
                chosen = None
                for a in starts:
                    row = {**window, 'in_sec': a, 'out_sec': a+duration}
                    if visible_prefix:
                        row.update(timeline_visible_prefix=True, identity_end_sec=a+evidence_length)
                    if row['out_sec'] > sources[sid].get('analyzed_end_sec', sources[sid]['duration_sec']):
                        continue
                    if any(r['source_id'] == sid and not (
                        a >= used_end(r)+contract['source_gap_sec']-.000001 or
                        used_end(row) <= r['in_sec']-contract['source_gap_sec']+.000001) for r in occupied):
                        self.defer_audio(window)
                        continue
                    tail = 0
                    if final_fallback or visible_prefix:
                        if not transition_safe(sources[sid], {**row, 'out_sec': a+evidence_length}, self.cache):
                            continue
                    elif not transition_safe(sources[sid], row, self.cache) or row['out_sec'] > row['shot_end_sec']+1e-6:
                        tail = min(contract['video_sampling'].get('transition_tail_fallback_sec', 0), max(0., duration-1.))
                        if not tail or not transition_safe(sources[sid], row, self.cache, tail_tolerance_sec=tail):
                            continue
                    frames = fingerprints(sources[sid], row, self.cache)
                    if any(similar(frames, old) for old in samples):
                        continue
                    chosen = (a, row, tail, frames)
                    break
                if chosen is None:
                    continue
                a, row, tail, frames = chosen
                # Allocation may choose a shorter fill, always inside this audit.
                window = {**window, 'start_sec': a-.08, 'end_sec': a+evidence_length+.08}
                if final_fallback:
                    window.update(final_scene_transition_fallback=True, fallback_slot_id=slot['id'],
                        fallback_duration_sec=duration, selection_out_sec=a+duration, identity_end_sec=a+evidence_length)
                if visible_prefix:
                    window.update(timeline_visible_prefix=True, fallback_slot_id=slot['id'],
                        fallback_duration_sec=duration, selection_out_sec=a+duration, identity_end_sec=a+evidence_length)
                if tail:
                    window.update(transition_tail_fallback_sec=tail,
                        fallback_duration_sec=duration, identity_end_sec=a+duration-tail)
                batch.append(window)
                occupied.append(row)
                samples.append(frames)
                available_slots.remove(slot)
                if not available_slots:
                    break
            if not batch:
                if not self.streams and not self.group_phase:
                    continue
                return self.result(contract, previous, errors, allow_same_shot)
            self.log(f"Classifying {len(batch)} candidate scenes at {batch[0]['start_sec']:.2f}s; searched {self.searched}")
            if getattr(self, 'source_session', None):
                self.source_session.release_for_vision()
            try:
                fast = None
                force_vision = bool(contract['references']) or self.vision_phase or (self.group_phase and
                    batch[0].get('subject_preview', {}).get('human_count') == 0)
                if self.people and not force_vision:
                    window = batch[0]
                    for offset in ((0.,) if window.get('source_sound') else (0., .3, -.3)):
                        shifted = {**window, 'start_sec': window['start_sec']+offset,
                                   'end_sec': window['end_sec']+offset}
                        tail = window.get('transition_tail_fallback_sec', 0)
                        if shifted['start_sec'] < window['shot_start_sec'] or shifted['end_sec'] > window['shot_end_sec']+tail:
                            continue
                        if tail:
                            shifted['identity_end_sec'] = shifted['end_sec']-.08-tail
                        if window.get('final_scene_transition_fallback') or window.get('timeline_visible_prefix'):
                            shifted.update(identity_end_sec=window['identity_end_sec']+offset,
                                selection_out_sec=window['selection_out_sec']+offset)
                        fast = self.people.classify(shifted, sources[window['source_id']], self.cache,
                                                    allow_group=self.group_phase)
                        if fast:
                            if not fast.get('deferred_group'):
                                batch[0] = shifted
                            break
                    if fast and fast.get('deferred_group'):
                        self.group_windows.append(window)
                        continue
                if fast:
                    self.log(f"CPU person check: {fast['id']}, {fast['subject_count']} subject(s), 3 body-count samples, "
                        f"{fast.get('identity_sample_frames', 0)} matching face samples.")
                    observed = {'cast': [fast], 'labels': [[0, fast['mouth']]], 'details': [fast]}
                elif self.people and self.people.human_mode and not force_vision:
                    self.log('CPU person check inconclusive; checking with local vision: '+
                             getattr(self.people, 'last_rejection', 'uncertain face'))
                    self.model_candidate_count += len(batch)
                    observed = classify(batch, contract, self.client, self.cache, self.registry, self.deadline, self.log)
                else:
                    self.model_candidate_count += len(batch)
                    observed = classify(batch, contract, self.client, self.cache, self.registry, self.deadline, self.log)
            except (TimeoutError, LocalAIError):
                if time.monotonic() < self.deadline-2:
                    raise
                return self.result(contract, previous, errors, allow_same_shot)
            self.classified_count += len(batch)
            for sample_index, (window, label) in enumerate(zip(batch, observed['labels'])):
                index, mouth = label
                if index < 0:
                    continue
                char = observed['cast'][index]
                current = self.registry.setdefault(char['id'], deepcopy(char))
                current['reference_ids'] = sorted(set(current['reference_ids']+char['reference_ids']))
                details = observed.get('details', [{} for _ in batch])[sample_index]
                run = {**window, **details, 'character_id': char['id'], 'mouth': mouth,
                       'voice_hint': 'speech' if mouth >= 2 else 'neutral'}
                if window.get('source_sound', {}).get('action'):
                    from .single import confirm_source_event
                    if getattr(self, 'source_session', None):
                        self.source_session.release_for_vision()
                    action = confirm_source_event(window, contract, self.client, self.cache, self.deadline)
                    if not action:
                        continue
                    run['visual_action'] = action
                if run.get('final_scene_transition_fallback') or run.get('timeline_visible_prefix'):
                    run['end_sec'] = run['selection_out_sec']+.08
                self.runs.append(run)
