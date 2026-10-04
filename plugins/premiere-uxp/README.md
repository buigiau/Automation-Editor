# AutoEdit Premiere UXP panel

Targets Premiere Pro 26.0.2 (manifest minimum 26.0.0). Version 0.3.1 uses source in/out marks and overwrite actions with a prepared video-only asset ([Adobe source-mark API](https://developer.adobe.com/premiere-pro/uxp/ppro-reference/classes/clipprojectitem/#createsetinoutpointsaction)). It does not require the 26.3 subclip API.

## Use

1. Load `manifest.json` through UXP Developer Tool, then reload the plugin after updates.
2. Generate a **new** plan with the Python GUI/CLI using the project you intend to open.
3. Open that project in Premiere and load `edit-plan.json` or `premiere_apply.json`.
4. Click **Fill nested clips** to fill all clips in the plan.
5. Review playback, then save manually. The panel does not save the project.

Updating the plugin: Unload then Load the plugin in UXP Developer Tool. If the old version is cached, remove the plugin entry and add this folder's `manifest.json` again. The panel log should say `AutoEdit 0.3.1 ready (Premiere 26.0+)`. Regenerate old plans to include the video-only fill asset and configured audio timings.

## Placement

- Multilevel nests are resolved to their footage sequences. For Zoomally, `BEAT MAU -> Nested Sequence 17 -> 4` fills `4` on V1, preserving the cuts on V2 of `Nested Sequence 17`. Shared leaves on other root tracks (including V4) are filled once by GUID. The panel verifies every saved wrapper/root reference before editing and refuses legacy plans targeting wrappers.
- Existing single-level templates use the original track/slot logic. If `PJ 5 - demo` is absent, Python can select the only unreferenced sequence with nests on the configured video track; ambiguous roots require an explicit `premiere.template_sequence`. Regenerate the plan and reload the panel for multilevel projects.
- Audio scenes follow the configured root video track, with nested source offsets and wrapper trims mapped onto the main timeline. Overlay tracks contribute video fills and review markers, without creating extra sampler tracks. Retimed nested clips require an explicit timing map and are rejected by the multilevel inspector.
- One source range per referenced nested-sequence GUID (14 in both reference projects).
- If there is an empty lead-in before scene 1, one separate `intro_fill` fills that gap with video only. Soda Pop uses 0–1s. It adds no nested sequence or audio assignment and does not move later clips. All video tracks are checked for occupied content before editing.
- Each range covers the nested V1 footage duration and every existing repeat's source out point.
- Fill V1 at zero inside the nested sequence. Main timeline cuts, repeat in/out points, effects on the main timeline, and adjustment layers on other tracks are retained.
- This replaces the base footage item on nested V1. Effects applied directly to that replaced footage item are not copied; put reusable styling on the nest or adjustment layers.
- Each source range is committed and read back before overwrite. The source used for fills contains no audio stream; audio index `-1` alone is not treated as a guarantee that existing audio is protected. The prepared asset is imported by its exact path. Source in/out marks are restored after filling, including when a fill fails.
- Configured timeline audio remains at its existing position, including through the intro. In the Soda Pop reference, A1 runs from timeline 0 to 65.8 seconds with source in/out 1 to 66.8 seconds. The panel compares these timings before editing, then checks audio clip positions, source marks and mute states after fills. Restore audio from the original project first if a previous run already damaged it.
- Sequence GUIDs disambiguate duplicate names. Old timeline-overwrite plans and changed template repeat layouts are rejected before editing.
- Insufficient source footage is an error; the tool does not silently shorten cuts.

The source analysis ranks mouth/expression matches. It does not detect camera shot boundaries; a selected continuous range can cross an original shot change.

If the host rejects an action during filling, the panel stops and reports the failure. Earlier completed fills remain undoable in Premiere.

Validation includes a mocked 26.0.2 API with no subclip method, 14 fills, source-mark restoration and host-failure paths. Loading and applying inside a real Premiere 26.0.2 host still needs user verification.

## Scene and audio selection

- In the Python GUI, `Source gap (seconds)` defaults to 5. This separates ranges in the source video; the template timeline timing is unchanged. If the source is too short, generation reports an error instead of reducing the gap or clip durations.
- The generator now analyzes the whole source by default. Ranking averages face size, mouth sharpness/visibility and activity across each complete cut, with a preference for different source regions. These are visual heuristics, not semantic understanding of an interesting scene. Face detection can miss cartoon faces; center fallback crops do not count as real face detections.
- Both source types use the single Voice folder. Whisper reads the original video's soundtrack and prioritizes the same word in the taxonomy. Animation uses the NEUTRAL pool when no word matches. Later reprises reuse the scene's sample.
- New sampler plans leave `Import audio after filling clips` off until a real Cubase export exists. The `Import mixdown WAV` button always opens a file picker; choose the Cubase export you want to place at 0s on an empty audio track from A2 onward. A1 and unrelated occupied tracks are preserved. The optional automatic import checkbox still uses the path configured in the plan. Export from Cubase time zero and retain its configured timing. Silence in the separate voice preview does not remove or shift the existing A1 beat. If all tracks are occupied, add an empty audio track and use `Import mixdown WAV` to retry only audio.
- Reimporting the same mixdown replaces its previous clip without ripple. A shorter replacement leaves no old audio tail. Host failures are reported instead of logging a false success.
- You can uncheck automatic audio import, or use the separate audio button. The existing limitation for effects directly on replaced nested V1 footage still applies.
