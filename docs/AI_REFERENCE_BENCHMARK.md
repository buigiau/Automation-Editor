# Cosita reference appearance search, 2026-10-19

## Latest failure and template visibility fix

The latest supplied job stopped after **541.077 seconds**, with 11/14 fills and
49 candidates checked (23 reached the vision-classification path). Missing:
`slot-03`, `slot-09`, `slot-11`. Its saved tail fallback was **0**, while the final
block exception was enabled. Upload took zero seconds; Clearcut telemetry errors
were not the selection failure.

The template's nested fill extent is not its visible duration in the parent:

| Slot | Required nested overwrite | Parent uses opening |
| --- | ---: | ---: |
| slot-03 | 5.68 s | 0.56 s |
| slot-04 | 1.08 s | 0.68 s |
| slot-12 | 0.84 s | 0.68 s |
| slot-13 | 3.12 s | 0.68 s |

Preflight now derives `visible_duration_sec` from every actual parent scene
instance, retaining retiming/offset validation and the full overwrite duration.
The local selector audits transitions and costume samples over that visible
prefix, and fills the unused nested tail to the original required length. Rows
carry `timeline_visible_prefix=true` and an exact `identity_end_sec`. The validator
allows this only when the inspected template authorises a shorter prefix, and
requires evidence for its entire duration. Full source bounds and overwrite
lengths are retained. Gap and duplicate comparisons use the footage actually
shown by the parent: unplayed padding cannot consume the five-second source gap.
All normal full-visible selections and the final-block exception still reserve
their full interval. Review playback also uses the displayed prefix.

This addresses the erroneous 5.68-second continuous-shot requirement without
allowing camera cuts in the 0.56 seconds that the parent actually plays. The final
block's explicit scene-change policy is separate and unchanged.

The first visibility-fix trial in
`output/cosita-reference-visible-cold-20261019a` completed structural generation in
393.324 seconds, but the masked-frame visual audit found a black floral shirt at
slot-07 instead of the orange reference jacket. That diagnostic manifest has been
renamed `premiere_apply.rejected.json`; its bundle audit explicitly rejects import.

The paired model response now reports the actual outer-garment colour separately
for each reference and both cut samples. Even when the model claims reference=0,
the deterministic check rejects unknown colours or colours outside the reference's
main garment and substantial measured costume colours. Purple printed clothing/
accessories in the orange-jacket reference remain valid across camera angles;
black floral shirts and white/blue garments do not inherit that permission. A real
two-case probe rejects the floral-shirt cut at 116.2 seconds (purple/black versus
reference orange), while retaining the orange-jacket cut at 107.4 seconds (orange
in both samples). See `output/reference-outfit-colour-probe/probe-result.json`.

Trial b used overly strict exact main-colour agreement and was cancelled after
rejecting genuine orange/purple angle changes. It is not a complete benchmark.

Trial c retained the full-padding gap and reached 13/14 at the time budget. It
correctly rejected the floral-shirt counterexample but is not a complete bundle.

Trial d applied displayed-prefix gaps. The model completed the last valid cut
near the 540-second discovery deadline, but deterministic reallocation hit that
deadline and returned the older 13-row best result. Inference/discovery remain
bounded by the original deadline. A separate, at most ten-second allocation
reserve now finalises newly completed observations, including the authorised
same-shot fallback, before export. Already attempted results still return their
saved partial review on expiry. Twenty seconds of the configured 570-second
maximum remain reserved for export; model calls do not use the allocation reserve.

The corrected code recovered **14/14 and 15 WAVs in 91.576 seconds** from trial d's
analysis cache (`output/cosita-reference-visible-recovered-20261019a`). All 42
face-masked costume samples were reviewed; the floral-shirt cut is absent. The
verified result was copied into the user's shared `output/.ai-cache` and a separate
output rebuild completed **14/14 in 3.481 seconds**, with `ai_cache_hit=true` and
AI stage zero (`output/cosita-reference-shared-cache-20261019a`). These are cache-
assisted measurements, not fresh analysis. The current ready-to-import files are
in the latter run's `first` directory. Full host import/playback is not measured.

The final fresh application-cache benchmark is
`output/cosita-reference-visible-cold-20261019e`, using the same supplied reference,
CPU/3 FPS, tail fallback 0 and the final-block exception enabled.
It completed **14/14 fills and 15 sampler WAVs in 546.691 seconds (9:07 rounded)**,
including worker launch and artifact generation; worker processing was 546.0s.
`ai-validation.json` reports complete=true with no errors and no repair.

| Measurement | Latest failed user run | Verified fresh run | Shared-cache rebuild |
| --- | ---: | ---: | ---: |
| Parent wall time | Not recorded | 546.691 s | 3.481 s |
| Worker time | 541.077 s | 546.0 s | 3.0 s |
| Selection/discovery/allocation | 537.988 s | 531.529 s | 0.0 s |
| Completed fills | 11/14 | 14/14 | 14/14 |
| WAV files | 0 | 15 | 15 |
| Candidate classifications recorded | 49 | 23 | Cached |

The fresh run's source cuts, reference IDs, visible-prefix boundaries and Voice
IDs exactly match the recovered bundle's 42 manually reviewed masked samples.
Its audit records that comparison as provenance. All input video, references,
Voice files and host projects retain their original content hashes. Sampled
costume matching is not facial identification or full playback validation.

191 focused tests pass, including actual Cosita template offsets, full nested
overwrite duration, evidence bounds, hidden-tail duplicate comparison, review
corrections, contradictory garment-colour rejection, reference accent colours,
and retention of the final verified batch after the model/discovery deadline.
Native auditing also tries bounded alternative cut positions inside each observed
candidate before discarding it; this can recover safe short footage when the
candidate midpoint happens to land on a camera cut. Source gaps, exact duration,
evidence boundaries and duplicate checks apply independently at every position.

## Earlier investigation

The user's reference run in `D:\cosita 2\191026` stopped after 540.547 seconds
with 8/14 fills. AI/discovery consumed 537.795 seconds; upload consumed zero.
The source is the supplied 2228.916667-second Ruby and Bonnie school compilation,
with `C:\Users\admin\Downloads\bonnie.png` as the reference.

Adding references previously disabled CPU candidate screening. The generator
returned one candidate per distant region and deferred other usable windows,
while each candidate needed two slow local vision calls. Generic colour/species
IDs could also accumulate a reference match from an unrelated earlier actor.

Human reference search now compares costumes, clothing, accessories and hair
styling. Facial identity matching is disabled in the CPU detector. Reference and
cut samples have face regions masked before model input; the original files are
unchanged. This cannot follow a person through arbitrary wardrobe changes.

CPU wardrobe histograms exclude face pixels, consider each detected body in a
group, and only screen/rank candidates. The vision model checks distinctive
appearance details in two separate cut samples within one compact request. Both
must agree on a reference look. Selected rows explicitly state
`reference_match_scope=costume_appearance`, `identity_sample_consistent=false`,
and two appearance samples. Unmatched actors cannot inherit reference IDs from
generic colour labels or substitute for the requested appearance on exhaustion.

Reference discovery first searches for the longest required continuous evidence
(at most 75 seconds, bounded by one quarter of the desired job duration),
then revisits cached regions for shorter fills. The final block requires only one
second of opening evidence and does not take priority over scarce long continuous
takes. Long candidate windows rank before nearby short windows, so short cuts do
not consume source-gap space needed by long fills.

Promising windows are yielded immediately. Later regions are prioritised near
recent confirmed appearance matches; distributed search remains available when
those regions are exhausted. Native transition, source gap and selected-frame
duplicate checks still apply. The final timeline block retains the user's full
scene-change exception and opening-only appearance evidence.

Diagnostic runs `cosita-reference-cold-20261019a` through
`cosita-reference-cold-20261019e` were cancelled during development. None is a
completed 14/14 runtime benchmark. Run c reached 12/14, with paired checks taking
14.11–18.41 seconds after the initial 30.39-second model start/check. The remaining
two continuous fills were not secured. Run e tried the authorised two-second tail
fallback but exposed false model matches to different outfits, so it was stopped
before export. No new reference bundle is claimed complete or ready to import.

The final implementation adds an independent wardrobe-palette veto at both cut
samples. It requires two distinct dominant reference colour bands when present;
one shared accessory colour cannot pass by itself. On actual cached cut samples,
10/13 pairs from run c pass this conservative screen; 0/5 pairs from the incorrect
run e pass. This is a rejection aid, not proof of identity or recognition accuracy.
The specific white-coat counterexample is rejected while the tested orange-jacket
views at 80, 92 and 116.4 seconds remain compatible. See
`output/reference-palette-audit/results.json`.

At that earlier stage, 179 focused tests passed. Source/template files were not edited. Saved settings then
enable the user's two-second ordinary tail fallback; the final timeline block
keeps its full-length scene-change exception. A broader ordinary-block camera-cut
fallback awaited the user's preference. At that stage, 14/14 under ten minutes
had not been verified; the template visibility fix above supersedes that limitation.

All diagnostic runs use separate application caches. Installed weights and OS
disk caching are outside cold-cache claims. Review, editor import and full
playback are outside artifact-generation timing.
