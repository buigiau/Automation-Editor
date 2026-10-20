# Cosita: Ruby and Bonnie, 2026-10-19

The supplied 37:08.917 source now produces all 14 fills and 15 sampler WAVs in
**145.745 seconds (2:26 rounded)** with a new application cache. The failed job
at `D:\cosita 2\191026` took 1062.567 seconds and accepted 12/14 fills.

Source: `YTSave_YouTube_Media_4CnWZP3vFp0_Ruby-and-Bonnie-BIGGEST-School-Adventures_001_1080p.mp4`.
Both measurements use the same Cosita project pair and source. The fresh run
uses local CPU analysis, 3 FPS, no source-duration limit, and no Qwen inference.
Installed model weights and OS disk caching are outside the cold-cache claim.

| Measurement | Failed job | Successful fresh run |
| --- | ---: | ---: |
| Worker elapsed | 1062.567 s | 145.257 s |
| Parent elapsed including worker launch | Not recorded | 145.745 s |
| Selection/discovery | 1055.324 s | 136.432 s |
| CPU preview images | 1785 | 99 |
| Candidate classifications | 17 | 14 |
| Completed fills | 12 / 14 | 14 / 14 |
| Sampler WAVs | 0 | 15 |
| Repairs | 1 | 0 |

The missing fills were 5.68 seconds (`slot-03`) and 9.28 seconds (`slot-14`).
The final block is now explicitly allowed to contain scene changes anywhere,
as requested. It is identified by timeline position, not its number or duration;
earlier uses of the same nested sequence prevent that exception from affecting
earlier blocks. Full source bounds, source gaps, duration and duplicate checks
still apply. Character/count evidence for this block describes its opening only.

Discovery also retains full observed shot windows after rolling previews,
overlaps seek regions, and avoids person/face inference on candidates too short
for the remaining fills. This recovers the useful middle of a shot instead of
trying only its unsafe edges. Ordinary checks accept `slot-03` at
1736.276667–1741.956667. The final fill uses 40.385–49.665 with its transition
exception recorded explicitly.

The successful run used `candidate-first-17-complete-shot-windows`. A subsequent
cache-provenance fix gives rolling and complete windows distinct thumbnail names
(`candidate-first-18-complete-shot-frames`), so they cannot overwrite each other's
images. That fix has a regression test; the timing above is the recorded v17 run.

Validation: 159 focused Python tests pass. The exported plan has no validation
errors, 14 unique slot IDs and 15 nonempty WAV files. Both source host projects
retain their original content hashes. Three additional frames per fill were
decoded and visually inspected; the final block visibly changes scenes as
allowed. Host import, full playback and voice/lip synchronization are outside
this timing and sampled visual audit.

Saved local preferences use `final_scene_transition_fallback=true`,
`transition_tail_fallback_sec=0`, `local_target_seconds=300` and
`local_max_seconds=570`.

Artifacts under `output/cosita-final-block-cold-20261019e/`:

- `report.json`
- `first/performance.json`
- `first/ai-validation.json`
- `first/ai_review.html`
- `first/edit-plan.json`
- `bundle-audit.json`
- `selected-frames-1.jpg` and `selected-frames-2.jpg`
