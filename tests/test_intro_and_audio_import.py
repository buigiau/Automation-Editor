from collections import Counter
from pathlib import Path
import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import soundfile as sf

from autoedit.audio.analyzer import analyze_file
from autoedit.audio.phonetics import filename_phonetics
from autoedit.audio.scene_match import choose_audio
from autoedit.cubase.archive import component, prepare_archive, sampler_tracks, verify_import, unpack_component, sample_field
from autoedit.cubase.archive import _signature
from autoedit.match.matcher import match_slots_to_video
from autoedit.premiere.prproj import intro_slot, inspect_prproj


def test_intro_is_extra_video_and_respects_existing_cuts_and_gap():
    slot = intro_slot([{"index": 1, "items": []}], [{"first_start_sec": 1}])
    video = {"duration_sec": 100,
             "samples": [{"time_sec": i / 2, "face_visible": 1} for i in range(200)]}
    cut = match_slots_to_video([slot], [], video, min_gap_sec=10,
                              require_speaking=False, excluded_ranges=[(40, 60)])[0]["video"]
    assert cut["out_sec"]-cut["in_sec"] == 1
    assert cut["out_sec"] <= 30 or cut["in_sec"] >= 70
    assert slot["audio_track_index"] == -1
    assert intro_slot([], [{"first_start_sec": 0}]) is None
    assert intro_slot([{"index": 0, "items": [{"start_sec": 0, "end_sec": 1}]}],
                      [{"first_start_sec": 1}])["status"] == "occupied"


def test_completed_soda_intro_does_not_become_sampler_01():
    path = Path(r"D:\Editor\Soda Pop\soda pop\Adobe Premiere Pro Auto-Save\Soda pop_1_1--559037c5-69e4-5f0b-8300-3b611d2ce6b7-2026-09-26_09-18-30.prproj")
    if not path.exists():
        pytest.skip("Local reference")
    data = inspect_prproj(path)
    assert data["intro_slot"]["required_duration_sec"] == 1
    assert len(data["slots"]) == 14 and len(data["scene_slots"]) == 16
    assert data["scene_slots"][0]["nested_sequence"] == "1"
    assert data["scene_slots"][0]["first_start_sec"] == 1
    beat = data["template_audio_tracks"][0]["items"][0]
    assert beat["start_sec"] == 0
    assert beat["end_sec"] == 65.8
    assert beat["source_in_sec"] == 1
    assert beat["source_out_sec"] == 66.8


@pytest.mark.parametrize("name,action,visemes", [("haha", "LAUGH", "A"),
    ("cười trẻ con", "LAUGH", "A"), ("ho khụ khụ", "COUGH", "A"),
    ("ba", "SPEECH", "MA"), ("bye", "SPEECH", "MAE"), ("đơm", "SPEECH", "EM")])
def test_phonetic_names_keep_expression_meaning(name, action, visemes):
    result = filename_phonetics(name)
    assert result["action"] == action
    assert result["visemes"] == list(visemes)


def test_filename_is_primary_and_24bit_envelope_trims_silence(tmp_path):
    path = tmp_path / "ba.wav"
    signal = np.r_[np.zeros(1600), .5*np.sin(np.arange(3200)*.2), np.zeros(1600)]
    sf.write(path, signal, 16000, subtype="PCM_24")
    info = analyze_file(path)
    assert info["phonetics"]["visemes"] == ["M", "A"]
    assert info["category"] == "BA"
    assert info["group"] == "NEUTRAL"
    assert info["active_start_sec"] == pytest.approx(.095, abs=.02)
    assert info["active_end_sec"] == pytest.approx(.305, abs=.02)
    assert len(info["envelope"]) == 20


def test_dense_match_excludes_cough_and_moves_voice_onset():
    info = {"samples": [{"time_sec": t, "clear_face": 1, "category": "A", "lip_aperture": .2 if t >= .2 else 0}
                         for t in np.arange(0, 1, .05)]}
    voice = {"path": "a.wav", "name": "a.wav", "category": "A", "duration_sec": .8,
             "phonetics": {"action": "SPEECH", "visemes": ["A"]},
             "envelope": [1]*40, "envelope_step_sec": .02}
    cough = {**voice, "path": "cough.wav", "phonetics": {"action": "COUGH", "visemes": ["A"]}}
    chosen, evidence = choose_audio([cough, voice], {"in_sec": 0, "out_sec": 1, "category": "A"}, info, Counter())
    assert chosen["path"] == "a.wav"
    assert chosen["placement_offset_sec"] == pytest.approx(.2, abs=.026)
    assert not evidence["phoneme_sync_verified"]


def test_archive_preserves_midi_fx_and_detects_tampering(tmp_path):
    template = Path("templates/cubase/soda_pop.xml")
    if not template.exists():
        pytest.skip("Reference track archive not installed")
    source = ET.parse(template).getroot()
    names = list(sampler_tracks(source))
    wav = tmp_path / "cười.wav"
    sf.write(wav, np.zeros(48000), 48000)
    slots = [{"cubase": {"track_name": n, "sample_path": str(wav)}} for n in names]
    archive = prepare_archive(template, slots, tmp_path / "samples.xml")
    report = verify_import(template, archive, slots)
    assert report["verified_tracks"] == 16
    root = ET.parse(archive)
    assert len(root.getroot().find("list[@name='track']")) == 16
    for track in sampler_tracks(root.getroot()).values():
        assert sample_field(unpack_component(component(track).text)[1])[2] == str(wav)
    note = next(n for n in root.iter("obj") if n.get("class") == "MMidiNote")
    note.find("int[@name='Data1']").set("value", "1")
    root.write(archive)
    with pytest.raises(ValueError, match="MIDI changed"):
        verify_import(template, archive, slots)


def test_velocity_mirror_normalization_does_not_hide_velocity_changes():
    note = ET.fromstring('<obj class="MMidiNote"><float name="HrData" value="0.5"/><member name="Additional Attributes"><float name="HRDT" value="0.5"/></member></obj>')
    saved = ET.fromstring('<obj class="MMidiNote"><float name="HrData" value="0.5"/></obj>')
    assert _signature(note) == _signature(saved)
    saved[0].set("value", "0.9")
    assert _signature(note) != _signature(saved)


def test_tempo_alias_normalization_keeps_actual_bpm():
    a = ET.fromstring('<obj class="MTempoTrackEvent"><float name="BPM" value="120"/></obj>')
    b = ET.fromstring('<obj class="MTempoTrackEvent" name="Tempo Track"><float name="BPM" value="120"/></obj>')
    assert _signature(a) == _signature(b)
    b[0].set("value", "130")
    assert _signature(a) != _signature(b)


@pytest.mark.parametrize("change", ["order", "timing", "automation", "routing"])
def test_sample_import_detects_changes_to_locked_project_configuration(tmp_path, change):
    template = Path("templates/cubase/soda_pop.xml")
    source = ET.parse(template).getroot()
    wav = tmp_path / "voice.wav"
    sf.write(wav, np.full(1000, .2), 8000)
    slots = [{"cubase": {"track_name": n, "sample_path": str(wav)}} for n in sampler_tracks(source)]
    path = prepare_archive(template, slots, tmp_path / "after.xml")
    root = ET.parse(path)
    track = next(iter(sampler_tracks(root.getroot()).values()))
    if change == "order":
        tracks = root.getroot().find("list[@name='track']")
        tracks.remove(track)
        tracks.append(track)
    elif change == "timing":
        track.find("float[@name='Start']").set("value", "1")
    elif change == "automation":
        automation = next(n for n in track.iter('obj') if n.get('class') == 'MAutomationNode')
        ET.SubElement(automation, "float", name="UnexpectedEvent", value="1")
    else:
        ET.SubElement(track.find("obj[@name='Track Device']"), "int", name="Output", value="99")
    root.write(path)
    with pytest.raises(ValueError, match="changed"):
        verify_import(template, path, slots)
