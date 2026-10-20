"""Shared incremental source-audio search, bounded workers and evidence caches."""
from collections import deque
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time

from autoedit.runtime import check_cancelled, memory_snapshot
from .matching import OpeningSoundIndex, POLICY_VERSION, search_budget


class SourceAudioError(ValueError):
    """ASR failed; this must not be presented as a silent/unmatched source."""


class SpeechWorker:
    def __init__(self, cache):
        self.cache, self.process, self.messages = Path(cache), None, Queue()
        self.error_file = None

    def request(self, arguments, deadline, command='scan'):
        if time.monotonic() >= deadline:
            raise TimeoutError('Source matching search budget exhausted')
        if self.process is None:
            self.cache.mkdir(parents=True, exist_ok=True)
            self.error_file = (self.cache/'speech-worker.log').open('ab')
            env = dict(os.environ)
            root = str(Path(__file__).resolve().parents[2])
            env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
            self.process = subprocess.Popen([sys.executable, '-m', 'autoedit.audio.speech_worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.error_file,
                text=True, encoding='utf-8', env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            process, queue = self.process, self.messages
            def read_responses():
                try:
                    for line in process.stdout:
                        queue.put(line)
                finally:
                    queue.put(None)
            threading.Thread(target=read_responses, daemon=True).start()
        try:
            self.process.stdin.write(json.dumps({'command': command, 'arguments': arguments})+'\n')
            self.process.stdin.flush()
            while True:
                check_cancelled()
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Source matching search budget exhausted during ASR')
                try:
                    line = self.messages.get(timeout=min(.1, remaining))
                except Empty:
                    continue
                if line is None:
                    raise SourceAudioError('Source audio worker exited; inspect speech-worker.log')
                response = json.loads(line)
                if 'error' in response:
                    raise SourceAudioError(response['error'])
                return response['result']
        except BaseException:
            self.close()
            raise

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()
        if self.error_file:
            self.error_file.close()
            self.error_file = None
        self.messages = Queue()


class SourceMatchSession:
    def __init__(self, config, sources, voices, cache, log=None, *, started=None, scan=None):
        from autoedit.ai.cache import digest
        self.config, self.sources, self.voices = config, sources, voices
        self.cache, self.log = Path(cache), log or (lambda _: None)
        self.started = time.monotonic() if started is None else started
        self.index = OpeningSoundIndex(voices)
        self.worker, self.scanner = SpeechWorker(self.cache/'source_audio'), scan
        ai, video = config.get('ai', {}), config.get('video', {})
        self.full = (ai.get('local_analysis_mode') == 'full' if ai.get('enabled') and ai.get('backend') == 'local'
                     else video.get('analysis_mode', 'until_filled') == 'full')
        budget = search_budget(config.get('audio', {}))
        self.final_deadline = float('inf') if self.full else self.started + budget
        maximum = ai.get('local_max_seconds', 0)
        if maximum:
            self.final_deadline = min(self.final_deadline, self.started+max(1., float(maximum)-30.))
        # Reserve a small part of the SAME budget for a complete neutral draft;
        # finalization must not start a fresh unlimited scene search at expiry.
        remaining = max(0., self.final_deadline-time.monotonic())
        self.deadline = (self.final_deadline-min(30., remaining*.1)
                         if math.isfinite(self.final_deadline) else self.final_deadline)
        self.settings = {'model': video.get('speech_model') or 'small',
                         'language': video.get('speech_language', 'en'),
                         'device': video.get('speech_device', 'cpu'),
                         'compute_type': video.get('speech_compute_type', 'auto'),
                         'device_index': video.get('speech_device_index', 0)}
        self.stats = {'policy_version': POLICY_VERSION, 'regions': 0, 'cache_hits': 0,
                      'asr_sec': 0., 'matched_onsets': 0, 'stop_reason': 'searching'}
        self.covered = {}
        self._parts = {}
        self._digest = digest
        for s in sources:
            s.setdefault('source_speech', {'words': [], 'events': [], 'status': 'not-scanned'})

    @property
    def expired(self):
        return time.monotonic() >= self.deadline

    def scan(self, source, lo, hi):
        from .source_speech import ASR_VERSION
        from autoedit.ai.cache import read, write
        check_cancelled()
        if source.get('has_audio') is False:
            source['source_speech'].update(has_audio=False, status='no-audio', policy_version=POLICY_VERSION)
            source['sound_anchors'] = []
            return source['source_speech']
        hi = min(hi, source.get('analyzed_end_sec', source['duration_sec']))
        if hi <= lo:
            return source['source_speech']
        key = self._digest([source['identity'], lo, hi, self.settings, ASR_VERSION])
        if key in self._parts:
            return source['source_speech']
        saved = self.cache/'source_audio'/'regions'/(key+'.json')
        part = read(saved)
        if part is None:
            if self.expired:
                raise TimeoutError('Source matching search budget exhausted')
            self.log(f"Source audio {source['id']}: {lo:.2f}–{hi:.2f}s")
            began = time.monotonic()
            args = dict(path=source['path'], ranges=[(lo, hi)], cache_dir=str(self.cache/'source_audio'), **self.settings)
            part = (self.scanner(**args) if self.scanner else self.worker.request(args, self.deadline))
            self.stats['asr_sec'] += time.monotonic()-began
            if not isinstance(part, dict) or 'words' not in part or 'events' not in part:
                raise SourceAudioError('Invalid source audio evidence; not cached as silence')
            write(saved, part)
        else:
            self.stats['cache_hits'] += 1
        self._parts[key] = part
        self.covered.setdefault(source['id'], []).append((lo, hi))
        speech = source['source_speech']
        for field in ('words', 'events'):
            # Padding is context; the midpoint assigns one owner per core.
            rows = speech[field] + [r for r in part[field] if lo <= (r['start']+r['end'])/2 < hi]
            unique = {(r['start'], r['end'], r.get('word'), r.get('action')): r for r in rows}
            speech[field] = sorted(unique.values(), key=lambda r: (r['start'], r['end']))
        speech.update(model=self.settings['model'], has_audio=part.get('has_audio', False),
                      status=part.get('status', 'transcribed'), policy_version=POLICY_VERSION)
        source['has_audio'] = part.get('has_audio', False)
        source['sound_anchors'] = self.index.anchors(speech)
        self.stats['regions'] += 1
        self.stats['matched_onsets'] = sum(len(s.get('sound_anchors', [])) for s in self.sources)
        return speech

    def regions(self):
        pending = deque((s, s.get('analyzed_start_sec', 0.)) for s in self.sources)
        while pending:
            check_cancelled()
            if self.expired:
                self.stats['stop_reason'] = 'time_budget'
                return
            s, lo = pending.popleft()
            if s.get('has_audio') is False:
                self.scan(s, lo, lo)
                continue
            end = s.get('analyzed_end_sec', s['duration_sec'])
            hi = min(lo+30., end)
            try:
                self.scan(s, lo, hi)
            except TimeoutError:
                self.stats['stop_reason'] = 'time_budget'
                return
            yield s, lo, hi
            if hi < end:
                pending.append((s, hi))
        self.stats['stop_reason'] = 'source_exhausted'

    def windows(self, contract):
        """Only source-supported onsets reach expensive character inference."""
        from autoedit.ai.cache import read, write
        from autoedit.video.reader import sampled_frames
        from autoedit.video.transitions import TransitionDetector, transition_metrics
        seen = set()
        longest = max(s['required_duration_sec'] for s in contract['slots'])
        for source, lo, hi in self.regions():
            anchors = [a for a in source.get('sound_anchors', []) if lo <= (a['start']+a['end'])/2 < hi]
            if not anchors:
                continue
            bounds_key = self._digest(['source-shot-bounds-1', source['identity'], lo, hi, longest])
            path = self.cache/'source_audio'/'shots'/(bounds_key+'.json')
            bounds = read(path)
            edge = min(source.get('analyzed_end_sec', source['duration_sec']), hi+longest+.2)
            if bounds is None:
                detector = TransitionDetector()
                for _ in sampled_frames(source['path'], 1, start_sec=max(0., lo-.5),
                                        max_seconds=edge, width=160, transition_detector=detector):
                    if self.expired:
                        self.stats['stop_reason'] = 'time_budget'
                        return
                bounds = sorted(set(detector.events))
                write(path, bounds)
            for anchor in sorted(anchors, key=lambda a: (a['confidence_tier'], a['start'])):
                t = anchor['start']
                for slot in sorted(contract['slots'], key=lambda s: -s['required_duration_sec']):
                    key = (source['id'], t, slot['id'])
                    if key in seen:
                        continue
                    seen.add(key)
                    duration = slot['required_duration_sec']
                    visible = slot.get('visible_duration_sec', duration)
                    identity_length = min(1., duration) if slot.get('allow_scene_changes') else visible
                    if t+duration > source.get('analyzed_end_sec', source['duration_sec'])+1e-6 or anchor['end'] > t+identity_length:
                        continue
                    safe = transition_metrics(bounds, t, t+identity_length)['transition_safe']
                    tail = 0.
                    if not safe and not slot.get('allow_scene_changes') and visible == duration:
                        tail = min(contract['video_sampling'].get('transition_tail_fallback_sec', 0), max(0., duration-1.))
                        safe = bool(tail and transition_metrics(bounds, t, t+duration-tail)['transition_safe'])
                    if not safe:
                        continue
                    left = max((b for b in bounds if b < t-.12), default=source.get('analyzed_start_sec', 0.))
                    right = min((b for b in bounds if b >= t+.001), default=edge)
                    shot = f'at-{round(left*1000)}' if left >= lo or lo == 0 else 'unknown'
                    window = {'source_id': source['id'], 'shot': shot, 'shot_start_sec': left,
                              'shot_end_sec': right, 'start_sec': t-.08,
                              'end_sec': t+identity_length+.08, 'source_sound': anchor,
                              'sound_slot_id': slot['id'], 'shot_bounds_complete': False}
                    if slot.get('allow_scene_changes') or visible < duration:
                        window.update(fallback_slot_id=slot['id'], fallback_duration_sec=duration,
                                      selection_out_sec=t+duration, identity_end_sec=t+identity_length)
                        window['final_scene_transition_fallback' if slot.get('allow_scene_changes') else 'timeline_visible_prefix'] = True
                    if tail:
                        window.update(transition_tail_fallback_sec=tail, fallback_duration_sec=duration,
                                      identity_end_sec=t+duration-tail, end_sec=t+duration+.08)
                    yield window

    def release_for_vision(self):
        available = memory_snapshot().get('ram_available_bytes')
        if self.worker.process is not None and (self.settings['device'] in {'cuda', 'auto'}
                or (available is not None and available < 7*1024**3)):
            self.worker.close()

    def close(self):
        self.worker.close()

    unload = close
