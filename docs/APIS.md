# Premiere Pro and Cubase automation APIs

Investigation notes for this repo. Versions observed on this machine:

- Adobe Premiere Pro 2025 (`C:\Program Files\Adobe\Adobe Premiere Pro 2025`)
- Steinberg Cubase 13 (`C:\Program Files\Steinberg\Cubase 13`)

## Premiere Pro

### Official surface: UXP (current)

Adobe moved 3rd-party extensibility to UXP. ExtendScript/CEP still exists through September 2026, but new work should use UXP.

Entry point:

```js
const ppro = require("premierepro");
const project = await ppro.Project.getActiveProject();
const sequences = await project.getSequences();
```

Relevant APIs for this workflow (Premiere 25.6+):

| Need | API | Notes |
| --- | --- | --- |
| List sequences / tracks / clips | `Project.getSequences`, `Sequence.getVideoTrack`, `VideoTrack.getTrackItems` | Fully available |
| Import media / mixdown WAV | `Project.importFiles(paths, suppressUI, targetBin, asNumberedStills)` | Fully available |
| Trim source before insert | `ClipProjectItem.createSetInOutPointsAction(in, out)` | Length of the next insert/overwrite |
| Hard subclip (26.3+) | `ClipProjectItem.createSubClipAction(name, start, end, hasHardBoundaries, options)` | Not used: this panel targets 26.0.2 |
| Place clip on a sequence | `SequenceEditor.createOverwriteItemAction` / `createInsertProjectItemAction` | This is the official cut-and-place path. There is **no razor/blade API** |
| Remove placeholder clips | `SequenceEditor.createRemoveItemsAction` | Use before overwrite when a slot already has media |
| Duplicate a clip to another track | `SequenceEditor.createCloneTrackItemAction` | Offset-based |
| Save a copy | `Project.saveAs(path)` | Becomes the active project, same as File > Save As |
| Export | `EncoderManager.exportSequence` | Needs an `.epr` preset |

Edit pattern that actually works (Adobe sample + forum confirmation):

1. Cast the source item to `ClipProjectItem`.
2. In **one** `lockedAccess` / `executeTransaction`, set in/out.
3. In a **second** transaction, overwrite/insert. Combining both in one compound action races and applies overwrite first.

```js
const editor = ppro.SequenceEditor.getEditor(templateSequence);
const inT = await ppro.TickTime.createWithSeconds(seg.in_sec);
const outT = await ppro.TickTime.createWithSeconds(seg.out_sec);
const at = await ppro.TickTime.createWithSeconds(slot.start_sec);

project.lockedAccess(() => {
  project.executeTransaction((ca) => {
    ca.addAction(clip.createSetInOutPointsAction(inT, outT));
  }, "Set in/out");
});
project.lockedAccess(() => {
  project.executeTransaction((ca) => {
    ca.addAction(editor.createOverwriteItemAction(clip, at, videoTrackIndex, audioTrackIndex));
  }, "Overwrite slot");
});
```

Plugin packaging: UXP `manifest.json` v5, `host.app = "premierepro"`, `minVersion` `26.0.0`. Load via UXP Developer Tool with Premiere Developer Mode enabled. The panel targets the user's 26.0.2 installation.

Offline inspection (no Premiere running): `.prproj` is gzipped XML (`PremiereData`). Sequences, nested-sequence slots, and clip start/end ticks (254016000000 ticks/sec) can be read from that XML. This repo's `autoedit.premiere.prproj` parser does that.

### Not available / avoid

- UXP has no razor/blade equivalent (confirmed through Premiere 26.2).
- Do not drive Premiere with mouse/keyboard macros.
- Do not rewrite `.prproj` XML to place clips. Write-back is unofficial and breaks project integrity.

## Cubase

Cubase has **no official project/timeline API** for “put this WAV on that track at this timestamp.”

What exists:

| Surface | What it can do | What it cannot do |
| --- | --- | --- |
| MIDI Remote API (JS, Cubase 12+) | Map a virtual controller to mixer, transport, and **named commands** such as File > Export Audio Mixdown | Cannot pass a file path or create audio events on a track |
| Project Logical Editor + key commands | Select/transform existing events, trigger macros | Cannot import an arbitrary file onto a named track by path |
| Generic Remote | Legacy CC mapping | Same limits |
| `.cpr` file | Proprietary binary | Unofficial to mutate |

Cubase 13 MIDI Remote scripts live at:

`Documents\Steinberg\Cubase\MIDI Remote\Driver Scripts\Local`

The Soda Pop `.cpr` contains **Sampler Track 01–16**; Golden contains **01–14**. Their paired Premiere templates have 16 and 14 contiguous scene runs respectively. Soda Pop ends by revisiting nested scenes 3 and 2. Individual repeated fragments within a scene do not create extra sampler tracks.

### Adapter strategy used here

1. Python maps each contiguous scene run to its own numbered Sampler Track. Source words (including explicit `I` → `ai` aliases) take priority over mouth shapes. Audio selection keeps the existing cuts fixed; later reprises reuse that scene's sample on the later track. Sample-only rematching preserves all scene and track bindings and does not generate Premiere actions.
2. Python writes a local WAV per track in `sampler_samples/`, starting at sample time zero. Loading a timeline stem padded with silence into Sampler Control is incorrect.
3. `cubase-import` validates audible WAV onsets and exports the sampler tracks selected manually by the user. It validates the exact sampler inventory before updating the SamplerTrackDoc filename field and SamplePath in a track archive. It imports only Channel and Inspector Settings onto matching existing tracks in the open project. Events/Parts and Automation are disabled. A second export verifies MIDI, sampler parameters and effect state; the project is left unsaved for user review. Temporary XML exports and the import-options screenshot are removed after successful verification, while failures retain them for diagnosis. Automatic track selection has been removed. Archive verification does not verify rendered audio.
4. Python's `premiere_audio_preview.wav` follows the Premiere repeat in/out offsets. It does not reproduce Cubase pitch, envelopes, MIDI or effects and is never automatically imported as the final mix.
5. Export the actual Cubase mix from time zero to the template end as `cubase_mixdown.wav`, then use the Premiere panel's **Import mixdown WAV**.

The CPR inspector only extracts embedded numbered track names. The separate host importer validates MIDI/plugin state through exported XML, including the compressed SamplerTrackDoc state. Unsupported envelopes/layouts stop. The tested profile uses English Cubase 13 and 100% display scaling. Save As can remove a redundant HRDT velocity mirror; verification normalizes it only when it exactly matches HrData. Original CPR bytes are never rewritten by Python.

Expected host dialogs are Export Selected Tracks / Save As (XML), Locate Track File, Import Options, then Export Selected Tracks / Save As (XML) for verification. No CPR Save As or project-save dialog is opened. Find Track is no longer opened by the importer.
