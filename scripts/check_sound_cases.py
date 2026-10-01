"""Audit actual speech WAVs with production ASR and the user's Voice library.

Input folder: sentences.json rows (filename, voice, sentence) plus WAV files.
Writes a listening/audit report; visual evidence is intentionally synthetic
to isolate recognition, pronunciation matching and onset-based cut selection.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import soundfile as sf

from autoedit.audio.analyzer import analyze_audio_dir
from autoedit.audio.pronunciation import build_sound_anchors
from autoedit.audio.scene_match import choose_audio
from autoedit.audio.source_speech import transcribe_source
from autoedit.config import DEFAULTS
from autoedit.match.matcher import match_slots_to_video


EXPECTED = {
    "Please buy a bag. The weight is too high. You said gee, not pea. That word is damn.": [
        {"bye"}, {"wait"}, {"two"}, {"hi"}, {"you", "iu"}, {"gi", "giii"}, {"pii"}, {"dam", "damm"}],
    "Please say oh, then yeah. My eye hurts. Who won one game? Hey, you can wait here.": [
        {"oh", "ohh", "ohhh"}, {"yeah", "yeahh", "yeah trầm"}, {"my", "mai", "mái"},
        {"ai", "aii", "ái"}, {"who", "whu", "huu"}, {"one"}, {"hey to", "heyy", "heyyy"},
        {"you", "iu"}, {"wait"}],
    "Buy it now. Say bye.": [{"bye"}],
    "We won one game.": [{"one"}],
    "You should wait here.": [{"you", "iu"}, {"wait"}],
    "Oh, my eye hurts.": [{"oh", "ohh", "ohhh"}, {"my", "mai", "mái"}, {"ai", "aii", "ái"}],
    "What did you do? Why?": [{"what", "whatt"}, {"you", "iu"}, {"why"}],
    "Fine. Fight. Fighting.": [{"fight"}],
    "Too high. A pea.": [{"two"}, {"hi"}, {"pii"}],
    "Who? Three, five, ten.": [{"who", "whu", "huu"}, {"three"}, {"five"}, {"ten"}],
    "Cough. Laugh.": [],
    "Hey! Yeah! Okay.": [{"hey to", "heyy", "heyyy"}, {"yeah", "yeahh", "yeah trầm"}, {"ok"}],
}


def audit_case(path, sentence, voice, library, cache, model):
    duration = sf.info(path).duration
    speech = transcribe_source(path, [(0, duration)], cache, model=model, language="en")
    anchors = build_sound_anchors(library, speech)
    selected = []
    # This tests the selector's use of actual ASR timestamps, not face detection.
    source_duration = duration + 2
    samples = [{"time_sec": i/6, "face_visible": 1, "clear_face": 1,
                "closeup_score": 1, "mouth_clarity": 1, "speaking": 1}
               for i in range(int(source_duration*6)+1)]
    for anchor in anchors:
        slot_duration = max(.5, anchor["end"]-anchor["start"]+.1)
        video = {"duration_sec": source_duration, "sample_fps": 6, "samples": samples,
                 "source_speech": speech, "sound_anchors": [anchor]}
        cut = match_slots_to_video([{"id": "probe", "required_duration_sec": slot_duration}], [], video)[0]["video"]
        chosen, evidence = choose_audio(library, cut, video, Counter())
        selected.append({"word": anchor.get("word"), "action": anchor.get("action"),
            "source_start_sec": anchor["start"], "cut_start_sec": cut["in_sec"],
            "sample": Path(chosen["path"]).name, "offset_sec": chosen["placement_offset_sec"],
            "method": evidence["method"],
            "needs_review": evidence["needs_review"], "phoneme_sync_verified": False,
            "onset_pass": abs(anchor["start"]-cut["in_sec"]) < 1e-6 and chosen["placement_offset_sec"] == 0})
    available = {Path(m["path"]).stem for a in anchors for m in a["matches"]}
    missing = [sorted(group) for group in EXPECTED[sentence] if not group & available]
    unexpected_events = speech.get("events", [])
    return {"file": str(path), "voice": voice, "sentence": sentence, "asr_model": model,
            "words": speech["words"], "anchors": anchors, "selected": selected,
            "missing_expected_groups": missing, "unexpected_events": unexpected_events,
            "pass": not missing and not unexpected_events and all(s["onset_pass"] for s in selected),
            "visual_evidence": "synthetic-face-samples", "padded_test_duration_sec": source_duration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--library", default=DEFAULTS["audio"]["directory"])
    parser.add_argument("--model", default="small")
    args = parser.parse_args()
    library = analyze_audio_dir(args.library)
    cases = json.loads((args.cases/"sentences.json").read_text(encoding="utf-8-sig"))
    rows = []
    for i, case in enumerate(cases, 1):
        result = audit_case(args.cases/case["filename"], case["sentence"], case["voice"],
                            library, args.cases/"cache", args.model)
        rows.append(result)
        print(f"{i}/{len(cases)} {'PASS' if result['pass'] else 'REVIEW'} {case['filename']}: "
              + " ".join(w["word"] for w in result["words"]), flush=True)
        (args.cases/"report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    events = []
    for stem in ("ho khụ khụ", "cười", "thở dài"):
        item = next(i for i in library if i["stem"] == stem)
        speech = transcribe_source(item["path"], [(0, item["duration_sec"])], args.cases/"cache",
                                   model=args.model, language="en")
        expected = item["phonetics"]["action"]
        events.append({"file": item["path"], "expected_action": expected, "words": speech["words"],
                       "events": speech.get("events", []),
                       "detected": any(e["action"] == expected for e in speech.get("events", []))})
    (args.cases/"event-report.json").write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"speech_cases": len(rows), "passed": sum(r["pass"] for r in rows),
                      "event_probes": [{"expected": e["expected_action"], "detected": e["detected"]} for e in events]}), flush=True)


if __name__ == "__main__":
    main()
