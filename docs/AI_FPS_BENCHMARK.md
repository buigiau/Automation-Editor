# Gemini FPS trial — Golden, 2026-10-15

The requested run uses the **entire selected source**, with Gemini video sampling
increased from the original 1 FPS to 3 FPS. There is no requirement to fill the
project from the first three minutes. `max_source_seconds` is 0, meaning full source.

Source: the supplied Talking Tom file containing YouTube ID `r7ZkiEazms8`.
Actual duration: 1791.208 seconds (29:51), native frame rate 24 FPS, size 454,205,029 bytes.
Preset: Golden, 14 unique fills and 14 sampler scene runs. Model: `gemini-3.8-flash`.
Both attempts reused the previously uploaded video and Voice cache.

| Full-source 3 FPS attempt | Wall time | Outcome |
| --- | ---: | --- |
| First | 24.164 s | HTTP 503, no selection/output |
| Retry | 20.244 s | HTTP 503, no selection/output |

These are failed request times, **not successful output latency**. No reliable
comparison between 1 and 3 FPS can be made from these two failures. Raw timings
and attempt history are in `output/golden-full-3fps-20261015/report.json`.

An earlier bounded-prefix experiment is retained separately at
`output/golden-fps-180s-20261015/`. It used a different evidence duration and must
not be presented as the full-source result requested here. Its interrupted
1 FPS retry is also unsuitable for timing comparison.

Saved app settings now use 3 FPS, full-source analysis, and shortage fallback.
Restart the GUI to load them. **Thiết lập → Nâng cao** exposes these controls.

Fallback behavior:

1. Prefer one selected segment per continuous shot.
2. If the allocation is incomplete, the single repair request can use different
   segments within the same shot.
3. Selected segments must still avoid overlap, honor the configured source gap,
   and pass local visual duplicate checks, including repeated footage across files.
4. Reused-shot selections are marked `same_shot_fallback` for review. Repeated
   long-footage group labels alone do not reject visually different segments in
   this mode. Actual repeated selected content remains blocked.

FPS, evidence window and fallback configuration are included in the AI cache key.
The analyzed window is validated separately from original source duration. Review
rebuilds preserve the deduplication policy and manual review provenance.

Validation: 73 AI sampling, selection, pipeline, transport, visual, GUI and real
template tests passed. Existing paired-template audio/MIDI/FX behavior is retained.
