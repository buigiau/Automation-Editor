# Golden real Gemini benchmark — 2026-10-15

Source: `YTSave_YouTube_Media_r7ZkiEazms8_️-Beach-in-the-Backyard-️-Talking-Tom-Shorts-Music-for-Kids_001_1080p.mp4`.
Duration 1791.208 seconds, 454,205,029 bytes. Model: `gemini-3.8-flash`.
Golden requires 14 unique fills and 14 sampler scene runs. No character references.

**The automatic first run did not complete successfully.** A usable review bundle
was recovered with additional Gemini input and explicit manual visual corrections.
Its rebuild time must not be presented as automatic scene-selection time.

| Measurement | Wall time | Outcome |
| --- | ---: | --- |
| Initial upload attempt | 146.219 s | 117.767 s upload/remote processing, then HTTP 503 |
| Retry using remote upload | 25.207 s | HTTP 503 |
| Request with original 300 s wait limit | 306.167 s | Timed out |
| Full-video selection with 900 s wait limit | 238.342 s | 12/14 structurally accepted; repair request returned 503 |
| Resumed repair on short video windows | 161.772 s | 13/14 accepted; repeated footage rejected |
| Export after manual visual recovery | 4.476 s | 14 fills, 14 sampler WAVs |
| Rebuild of the reviewed bundle | 1.579 s | Completed; zero API calls; worker processing 1.2 s |

The full-video selection took 218.336 seconds in API calls and 15.091 seconds
checking selected frames. The focused repair call took 156.006 seconds. Idle time
between resumed sessions is excluded. Active review time and host import/playback
were not timed, so there is no complete end-to-end speed measurement against the
5-minute manual baseline.

Visual review exposed errors beyond JSON validity:

- Original slot-02 was Ben rather than the claimed Tom. Focused Gemini repair
  replaced it with Tom at 120.00–121.72 seconds.
- Repaired slot-07 repeated slot-01 footage at another timestamp. Local frame
  matching rejected it. Manual review selected Tom at 2.15–4.39 seconds.
- Original slot-14 crossed a hard cut from the hamster to Tom. Manual review
  shifted it to 1405.10–1407.66 seconds.
- Broad motion events in slots 05 and 09 were inspected as zoom/pan rather than
  independent hard cuts. Some selected views have weak mouth visibility or
  inactive mouths. All 14 audio selections remain explicitly uncertain.

The reviewed bundle passes structural duration/gap checks, selected-frame duplicate
checks and sampler WAV checks. Minimum selected source gap is 12.41 seconds;
total selected source duration is 19.76 seconds. A separate local native-frame
audit found no remaining hard-cut events inside the recovered windows. These
checks do not prove exact character identity, shot boundaries or lip synchronization.
Source and template content fingerprints remained unchanged.

Artifacts are under `output/golden-gemini-benchmark-20261014/`:

- `report.json`: failed automatic attempts, recovery and cached rebuild metrics.
- `first/`: original failed automatic run and partial selections, preserved.
- `reviewed/edit-plan.json`: recovered plan with explicit review provenance.
- `reviewed/ai_review.html`: source-video/Voice comparison and corrections.
- `reviewed/cubase_import.json`, `reviewed/premiere_apply.json`: recovered autofill.
- `selected-scenes.jpg`, `audit.json`, `local-shot-audit.json`: local review evidence.

The automatic AI result cache was not populated with manual corrections. The
1.579-second measurement uses the normal review rebuild path, not a fresh AI job.
Playback in Cubase and listening confirmation remain necessary.

Code changes made during this measurement: configurable bounded request timeout,
API failure time recording, benchmark phase selection, and preservation of manual
review provenance across exports. The 23 related transport/pipeline tests pass.
