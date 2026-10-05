from collections import Counter

import pytest

from autoedit.audio.onsets import articulation_onsets, AudioOnsetError
from autoedit.audio.scene_match import choose_audio
from autoedit.audio.phonetics import filename_phonetics
from autoedit.audio.pronunciation import VoiceIndex
from test_characters import face


def lip_samples(start, end, category='A'):
    """Source-defined motion, independent of the requested cut boundary."""
    return [{**face(), 'time_sec': i/20, 'clear_face': 1, 'speaking': 1,
             'lip_aperture': [.05,.15,.3,.3,.15,.05][i%6], 'lip_width_ratio': .5, 'category': category}
            for i in range(max(0, int((start-.35)*20)), int((end+.35)*20)+1)]


def motion(values, start=0, category='A', track='actor'):
    return [{'time_sec': start+i*.05, 'clear_face': 1, 'category': category,
             'lip_aperture': value, 'lip_width_ratio': .5, 'track_id': track, 'shot_id': 1}
            for i,value in enumerate(values)]


@pytest.mark.parametrize('stem,shape', [('a','A'), ('ê','E'), ('oo','U'), ('oh','O')])
def test_all_vowel_families_accept_opening_and_reject_closing(stem, shape):
    phon = filename_phonetics(stem)
    opened = articulation_onsets(motion([0,0,.15,.25,.2], category=shape), phon['visemes'], 0,.3)
    assert opened[0]['visual_onset_sec'] == pytest.approx(.1)
    assert not articulation_onsets(motion([.3,.2,.1,0,0], category=shape), phon['visemes'], 0,.3)


def test_preparatory_closed_frames_are_not_vowel_onsets():
    assert not articulation_onsets(motion([0,0,0,0]), ['A'], 0,.3)
    assert not articulation_onsets(motion([.2]*5), ['A'], 0,.3)


def test_closed_consonant_starts_at_release_not_earlier_preparation():
    evidence = articulation_onsets(motion([0,0,0,.2,.25]), ['M','A'], 0,.3)
    assert evidence[0]['visual_onset_sec'] == pytest.approx(.1)
    assert evidence[0]['mouth_phase'] == 'closed-consonant-release'


def test_track_change_cannot_manufacture_an_opening():
    samples = motion([0,0,.2,.25])
    samples[2]['track_id'] = 'other'
    assert not articulation_onsets(samples, ['A'], 0,.3)


def test_source_word_and_visual_fallback_both_require_articulation():
    item = {'path':'ai.wav','duration_sec':.3,'category':'AI', 'phonetics':filename_phonetics('ai')}
    cut = {'in_sec':0.,'out_sec':1.}
    info = {'verify_onsets':True, 'samples':motion([.3,.2,.1,0,0]),
            'source_speech':{'words':[{'word':'I','start':0.,'end':.3,'prob':.95}]}}
    with pytest.raises(AudioOnsetError):
        choose_audio([item], cut, info, Counter())
    info['source_speech'] = {}
    with pytest.raises(AudioOnsetError):
        choose_audio([item], cut, info, Counter())


def test_source_word_onset_has_visual_evidence_and_preserves_wav_beginning():
    item = {'path':'ai.wav','duration_sec':.3,'category':'AI','active_start_sec':.02,
            'active_end_sec':.3,'phonetics':filename_phonetics('ai')}
    info = {'verify_onsets':True, 'samples':motion([0,0,.15,.2,.3]),
            'source_speech':{'words':[{'word':'I','start':.08,'end':.3,'prob':.95}]}}
    selected, evidence = choose_audio([item], {'in_sec':0.,'out_sec':1.}, info, Counter())
    assert evidence['onset_verified'] and evidence['suggested_cut_in_sec'] == pytest.approx(.1)
    assert selected['segment_start_sec'] == .02
    assert not evidence['phoneme_sync_verified']


@pytest.mark.parametrize('stem', ['ợ','gru','ho khụ khụ','ha','ting'])
def test_events_never_leak_into_speech_even_with_stale_catalog(stem):
    event = {'path':stem+'.wav','duration_sec':.3,'group':'NEUTRAL','category':'A',
             'phonetics':{'action':'SPEECH','visemes':['A']}}
    voice = {'path':'hi.wav','duration_sec':.3,'category':'HI'}
    selected, _ = choose_audio([event, voice], {'in_sec':0.,'out_sec':1.}, {}, Counter())
    assert selected['path'] == 'hi.wav'
    selected, _ = choose_audio([event, voice], {'in_sec':0.,'out_sec':1.},
                              {'verify_onsets':True,'samples':lip_samples(0,1)}, Counter())
    assert selected['path'] == 'hi.wav'


def test_event_requires_event_evidence_not_transcript_word():
    index = VoiceIndex([{'path':'ha.wav','duration_sec':.3}])
    assert not index.match({'word':'ha'})
    assert index.match({'action':'LAUGH'})
    assert not VoiceIndex([{'path':'ợ.wav','duration_sec':.3}]).match({'word':'err'})


def test_overlapping_context_does_not_mix_another_selected_actor():
    from autoedit.audio.scene_match import _cut_samples
    cut = {'in_sec':1.,'out_sec':2.,'selection_metrics':{'subject_track_id':'target'}}
    correct = motion([0,.1,.2],start=.95,track='target')
    for sample in correct:
        sample['refinement_cut'] = {'in_sec':1.,'out_sec':2.}
    wrong = motion([.3,.2,.1],start=.95,track='other')
    for sample in wrong:
        sample['refinement_cut'] = {'in_sec':.5,'out_sec':1.5}
    selected = _cut_samples({'samples':wrong+correct+correct},cut,.35)
    assert len(selected) == 3 and all(s['track_id']=='target' for s in selected)


def test_saved_speech_anchor_cannot_restore_a_reclassified_vocal_event():
    cut = {'in_sec':0.,'out_sec':1.,'source_sound':{'word':'err','start':0.,'end':.3,'prob':.95,
           'matches':[{'path':'ợ.wav','kind':'source-audio-pronunciation','phones':['ER']}]}}
    candidates = [{'path':'ợ.wav','duration_sec':.3,'phonetics':{'action':'SPEECH'}},
                  {'path':'hi.wav','duration_sec':.3}]
    with pytest.raises(ValueError, match='regenerate'):
        choose_audio(candidates,cut,{},Counter())


def test_earlier_visual_onset_moves_the_cut_without_trimming_first_sound():
    voice = {'path':'a.wav','category':'A','duration_sec':.3,'active_start_sec':.02,'active_end_sec':.3}
    info = {'verify_onsets':True,'samples':motion([0,.2,.3,.15,.1,.05,0])}
    selected, evidence = choose_audio([voice],{'in_sec':.1,'out_sec':.8},info,Counter())
    assert evidence['suggested_cut_in_sec'] == pytest.approx(.05)
    assert selected['segment_start_sec'] == .02
