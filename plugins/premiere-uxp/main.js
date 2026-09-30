const ppro = require("premierepro");
const uxp = require("uxp");
const fs = uxp.storage.localFileSystem;

let plan = null;
let applyDoc = null;
let applying = false;

function log(msg) {
  const el = document.getElementById("log");
  const line = `[${new Date().toISOString().slice(11, 19)}] ${msg}`;
  el.textContent = `${el.textContent}${line}\n`;
  el.scrollTop = el.scrollHeight;
  console.log(msg);
}

async function getProject() {
  const project = await ppro.Project.getActiveProject();
  if (!project) throw new Error("No active Premiere project");
  return project;
}

async function sequencesByName(project) {
  const seqs = await project.getSequences();
  const map = {};
  for (const s of seqs) {
    map[s.name] = s;
  }
  return map;
}

async function walkFolder(folder, acc) {
  const items = await folder.getItems();
  for (const item of items) {
    acc.push(item);
    try {
      const asFolder = ppro.FolderItem.cast(item);
      if (asFolder) await walkFolder(asFolder, acc);
    } catch (_) {
      /* not a folder */
    }
  }
  return acc;
}

function toForwardSlashPath(mediaPath) {
  // Premiere re-reads Windows paths and treats \a \t \n as escapes,
  // so C:\Users\admin\... becomes C:\Usersadmin\... and the file cannot be opened.
  let s = String(mediaPath || "").trim();
  s = s.replace(/^file:\/*/i, "");
  s = s.replace(/\\/g, "/");
  if (/^\/[A-Za-z]:\//.test(s)) s = s.slice(1);
  return s;
}

async function findClipByMediaPath(project, mediaPath) {
  const root = await project.getRootItem();
  const items = await walkFolder(root, []);
  const needle = toForwardSlashPath(mediaPath).toLowerCase();
  const needleName = needle.split("/").pop();
  for (const item of items) {
    const clip = ppro.ClipProjectItem.cast(item);
    if (!clip) continue;
    try {
      const p = await clip.getMediaFilePath();
      if (!p) continue;
      const norm = toForwardSlashPath(p).toLowerCase();
      if (norm === needle || (!needle.includes("/") && norm.endsWith("/" + needleName))) {
        return item;
      }
    } catch (_) {
      if (!needle.includes("/") && clip.name && clip.name.toLowerCase() === needleName) return item;
    }
  }
  return null;
}

function runAction(project, label, build) {
  let ok = false;
  project.lockedAccess(() => {
    ok = project.executeTransaction((compound) => {
      const action = build();
      if (action) compound.addAction(action);
    }, label);
  });
  return ok;
}

async function inspectProject() {
  const project = await getProject();
  const seqs = await project.getSequences();
  log(`project: ${project.name || "(open)"}  sequences: ${seqs.length}`);
  for (const seq of seqs) {
    const vcount = await seq.getVideoTrackCount();
    const acount = await seq.getAudioTrackCount();
    log(`  seq ${JSON.stringify(seq.name)}  V${vcount} A${acount}`);
  }
}

async function loadPlan() {
  const file = await fs.getFileForOpening({ types: ["json"] });
  if (!file) return;
  const text = await file.read();
  const data = JSON.parse(text);
  if (data.operation_scope === "sample_assignment_only" || data.superseded_by) {
    plan = data;
    applyDoc = null;
    document.getElementById("autoMix").checked = false;
    log("Sample-only or superseded plan: Premiere is unchanged. Use the current plan with Import Cubase only.");
    return;
  }
  if (data.fill_nested_sequences) {
    applyDoc = data;
    plan = data;
    applyDoc.source_video = data.source_video || data.source_media;
    applyDoc.source_media = applyDoc.source_video;
    applyDoc.template_sequence = data.template_sequence || "PJ 5 - demo";
    applyDoc.save_project = false;
  } else {
    plan = data;
    applyDoc = {
      fill_policy: "nested_sequences_v2",
      source_duration_sec: (data.video_analysis_meta || {}).duration_sec,
      template_sequence: (data.project || {}).template_sequence || "PJ 5 - demo",
      source_video: (data.project || {}).source_video || (data.project || {}).source_media,
      source_media: (data.project || {}).source_video || (data.project || {}).source_media,
      video_only_source: (data.project || {}).video_only_source,
      template_audio_tracks: data.template_audio_tracks,
      import_source_video: true,
      save_project: false,
      mixdown_wav: (data.mixdown || {}).path,
      mixdown_duration_sec: (data.mixdown || {}).duration_sec,
      import_mixdown: Boolean((data.project || {}).import_mixdown),
      intro_fill: data.intro_fill || null,
      fill_nested_sequences: (data.slots || [])
        .filter((s) => s.video && (s.video.in_sec || s.video.in_sec === 0))
        .map((s) => {
          const prem = s.premiere || {};
          const mode = prem.mode || "nested_sequence";
          return {
            slot_id: s.id,
            mode,
            nested_sequence: prem.nested_sequence,
            nested_sequence_uid: prem.nested_sequence_uid,
            instances: prem.instances || [],
            template_video_track_index: prem.video_track_index,
            template_sequence: (data.project || {}).template_sequence,
            source_media: (data.project || {}).source_video || (data.project || {}).source_media,
            source_start_sec: s.source_start_sec != null ? s.source_start_sec : s.video.in_sec,
            source_end_sec: s.source_end_sec != null ? s.source_end_sec : s.video.out_sec,
            in_sec: s.source_start_sec != null ? s.source_start_sec : s.video.in_sec,
            out_sec: s.source_end_sec != null ? s.source_end_sec : s.video.out_sec,
            overwrite_at_sec: 0,
            slot_end_sec: prem.timeline_end_sec,
            slot_duration_sec: prem.duration_sec,
            video_track_index: Number(prem.fill_video_track_index || 0),
            audio_track_index: -1,
            character_selection: s.video.character_selection || null,
          };
        }),
    };
  }
  log(`loaded ${applyDoc.fill_nested_sequences.length} slot fills`);
  log(`source video ${applyDoc.source_video || applyDoc.source_media}`);
  log(`target ${applyDoc.template_sequence} (open project will be used; no save)`);
  document.getElementById("autoMix").checked = Boolean(applyDoc.import_mixdown);
  log("audio: Import mixdown WAV opens a file picker" + (applyDoc.import_mixdown
    ? " | automatic import after filling uses " + (applyDoc.mixdown_wav || "no mixdown path in plan")
    : ""));
}

async function importByPath(project, root, filePath) {
  try {
    return await project.importFiles([filePath], true, root, false);
  } catch (err) {
    log("importFiles: " + (err && err.message ? err.message : err));
    return false;
  }
}

async function pickFile(prompt, types) {
  log(prompt);
  const picked = await fs.getFileForOpening({ types: types });
  if (!picked || !picked.nativePath) {
    throw new Error("Import cancelled");
  }
  log("selected " + picked.nativePath);
  return picked.nativePath;
}

async function ensureSourceClip(project, mediaPath) {
  let clip = await findClipByMediaPath(project, mediaPath);
  if (!clip) {
    const root = await project.getRootItem();
    if (await importByPath(project, root, toForwardSlashPath(mediaPath))) {
      clip = await findClipByMediaPath(project, mediaPath);
    }
    if (!clip) {
      log("Video-only fill source: " + toForwardSlashPath(mediaPath));
      const picked = await fs.getFileForOpening({ types: ["mov", "mp4", "m4v", "mxf"] });
      if (!picked || toForwardSlashPath(picked.nativePath).toLowerCase() !== toForwardSlashPath(mediaPath).toLowerCase()) {
        throw new Error("Choose the exact video-only fill source in the plan; original video with audio cannot be used for fills.");
      }
      if (await importByPath(project, root, toForwardSlashPath(picked.nativePath))) {
        clip = await findClipByMediaPath(project, mediaPath);
      }
    }
  }
  if (!clip) {
    throw new Error("Imported video but the project item lookup returned null");
  }
  const projectItem = ppro.ProjectItem.cast(clip) || clip;
  const clipItem = ppro.ClipProjectItem.cast(projectItem);
  if (!clipItem || typeof clipItem.createSetInOutPointsAction !== "function") {
    throw new Error("Looked up a project item, but it is not a ClipProjectItem: " + (clip.name || ""));
  }
  let mediaPathFound = "";
  try {
    mediaPathFound = await clipItem.getMediaFilePath();
  } catch (err) {
    mediaPathFound = "(getMediaFilePath failed: " + (err && err.message ? err.message : err) + ")";
  }
  log(
    "source projectItem name=" + (projectItem.name || "") +
    " path=" + mediaPathFound +
    " overwriteArg=ProjectItem"
  );
  return { projectItem: projectItem, clipItem: clipItem };
}

const TICKS_PER_SECOND = 254016000000;

async function makeTick(seconds) {
  const n = Number(seconds);
  if (!Number.isFinite(n) || n < 0) throw new Error("time is not a positive number of seconds: " + seconds);
  let tick = ppro.TickTime.createWithSeconds(n);
  if (tick && typeof tick.then === "function") tick = await tick;
  if (!tick || tick.ticks == null) {
    tick = ppro.TickTime.createWithTicks(String(Math.round(n * TICKS_PER_SECOND)));
    if (tick && typeof tick.then === "function") tick = await tick;
  }
  if (!tick || tick.ticks == null) throw new Error("TickTime was not created for " + n + "s");
  return tick;
}

function tickSeconds(tick) {
  if (!tick) return null;
  if (typeof tick.seconds === "number") return tick.seconds;
  return null;
}

function tickId(tick) {
  if (!tick) return "null";
  if (tick.ticks != null) return String(tick.ticks);
  return String(tick);
}

async function sequenceFrameRate(sequence) {
  const names = ["getVideoFrameRate", "getFrameRate", "getTimebase"];
  for (const name of names) {
    if (typeof sequence[name] !== "function") continue;
    try {
      const value = await sequence[name]();
      log("sequence." + name + "=" + JSON.stringify(value));
      const numeric = Number(value && (value.value != null ? value.value : value.fps != null ? value.fps : value));
      if (Number.isFinite(numeric) && numeric > 1 && numeric < 240) return numeric;
    } catch (err) {
      log("sequence." + name + " failed: " + (err && err.message ? err.message : err));
    }
  }
  return null;
}

function snapToFrame(seconds, fps) {
  if (!fps) return seconds;
  return Math.round(seconds * fps) / fps;
}

async function clipDurationSeconds(clip) {
  const attempts = ["getDuration", "getMediaDuration", "getOutPoint"];
  for (const name of attempts) {
    if (typeof clip[name] !== "function") continue;
    try {
      const value = await clip[name]();
      const seconds = tickSeconds(value);
      if (Number.isFinite(seconds) && seconds > 0) return seconds;
    } catch (_) {
      /* try the next duration API */
    }
  }
  return null;
}

function runNamedAction(project, callName, build) {
  // Build the Action under lockedAccess, then commit it in a transaction.
  let thrown = null;
  let ok = false;
  try {
    project.lockedAccess(() => {
      let action = null;
      try {
        action = build();
      } catch (err) {
        thrown = err && err.message ? err.message : String(err);
        return;
      }
      if (!action) {
        thrown = callName + " returned no Action";
        return;
      }
      ok = project.executeTransaction((compound) => {
        compound.addAction(action);
      }, callName);
    });
  } catch (err) {
    thrown = err && err.message ? err.message : String(err);
  }
  return { ok: !thrown && ok !== false, error: thrown };
}

function normalizedGuid(value) {
  return String(value || "").replace(/[{}]/g, "").toLowerCase();
}

async function verifyTemplateInstances(template, actions) {
  const tracks = new Map();
  for (const action of actions) {
    const trackIndex = Number(action.template_video_track_index);
    if (!Number.isInteger(trackIndex) || trackIndex < 0) throw new Error("Missing template track index");
    if (!tracks.has(trackIndex)) {
      const groups = new Map();
      for (const item of await clipTrackItems(template, trackIndex)) {
        const clip = ppro.ClipProjectItem.cast(await item.getProjectItem());
        if (!clip || !(await clip.isSequence())) continue;
        const seq = await clip.getSequence();
        const uid = normalizedGuid(seq.guid);
        if (!groups.has(uid)) groups.set(uid, []);
        groups.get(uid).push({
          start_sec: await readTickSeconds(item, "getStartTime"),
          end_sec: await readTickSeconds(item, "getEndTime"),
          source_in_sec: await readTickSeconds(item, "getInPoint"),
          source_out_sec: await readTickSeconds(item, "getOutPoint"),
        });
      }
      tracks.set(trackIndex, groups);
    }
    const actual = tracks.get(trackIndex).get(normalizedGuid(action.nested_sequence_uid)) || [];
    const expected = action.instances || [];
    actual.sort((a, b) => a.start_sec - b.start_sec);
    if (!expected.length || actual.length !== expected.length || actual.some((item, i) =>
      ["start_sec", "end_sec", "source_in_sec", "source_out_sec"].some(key =>
        expected[i][key] == null || Math.abs(item[key] - expected[i][key]) > 0.000001))) {
      throw new Error("Template differs from plan for clip " + action.nested_sequence + ". Regenerate the plan from this project.");
    }
  }
}

async function snapshotAudio(sequence) {
  const tracks = [];
  for (let index = 0; index < await sequence.getAudioTrackCount(); index++) {
    const track = await sequence.getAudioTrack(index);
    const items = [];
    for (const item of await track.getTrackItems(ppro.Constants.TrackItemType.CLIP, false)) {
      const clip = ppro.ClipProjectItem.cast(await item.getProjectItem());
      items.push({
        path: clip ? toForwardSlashPath(await clip.getMediaFilePath()) : "",
        start_sec: await readTickSeconds(item, "getStartTime"),
        end_sec: await readTickSeconds(item, "getEndTime"),
        source_in_sec: await readTickSeconds(item, "getInPoint"),
        source_out_sec: await readTickSeconds(item, "getOutPoint"),
      });
    }
    items.sort((a, b) => a.start_sec - b.start_sec || a.end_sec - b.end_sec || a.path.localeCompare(b.path));
    tracks.push({ index, muted: await track.isMuted(), items });
  }
  return tracks;
}

function verifyConfiguredAudio(actual, expected) {
  for (const track of expected || []) {
    if (!track.items.length) continue;
    const before = [...track.items].sort((a, b) => a.start_sec - b.start_sec || a.end_sec - b.end_sec);
    const now = (actual.find(t => t.index === track.index) || {}).items || [];
    if (now.length !== before.length || now.some((item, i) =>
      ["start_sec", "end_sec", "source_in_sec", "source_out_sec"].some(key =>
        before[i][key] == null || Math.abs(item[key] - before[i][key]) > 0.000001))) {
      throw new Error("Configured audio on A" + (track.index + 1) +
        " differs from the saved template. Restore the original timeline audio before applying; intro must not move or trim it.");
    }
  }
}

async function verifyAudioUnchanged(sequence, before) {
  if (JSON.stringify(await snapshotAudio(sequence)) !== JSON.stringify(before)) {
    throw new Error("Configured audio changed while filling video in " + sequence.name +
      ". Stop and undo the fill; audio must keep its original timeline start and source marks.");
  }
}

async function applyPlan() {
  if (applying) throw new Error("A fill is already running. Wait for the summary.");
  applying = true;
  try {
    await applyNestedPlan();
    if (document.getElementById("autoMix").checked) {
      try {
        await importMixdownImpl(false);
      } catch (err) {
        throw new Error("Video fills completed, but audio import failed: " + err.message + ". Use Import mixdown WAV to retry audio only.");
      }
    }
  } finally {
    applying = false;
  }
}

async function applyNestedPlan() {
  if (!applyDoc || applyDoc.fill_policy !== "nested_sequences_v2") {
    throw new Error("Regenerate edit-plan.json with the updated tool before applying.");
  }
  if (!applyDoc.video_only_source) {
    throw new Error("Regenerate the plan with a video-only fill source to preserve timeline audio, including the intro.");
  }
  const project = await getProject();
  const sequences = await project.getSequences();
  const templates = sequences.filter(s => s.name === applyDoc.template_sequence);
  if (templates.length !== 1) throw new Error("Template sequence missing or ambiguous");
  const template = templates[0];
  const templateAudio = await snapshotAudio(template);
  verifyConfiguredAudio(templateAudio, applyDoc.template_audio_tracks);
  const all = applyDoc.fill_nested_sequences || [];
  if (!all.length) throw new Error("Plan has no nested fills");
  await verifyTemplateInstances(template, all);
  const seen = new Set();
  const prepared = [];
  // Validate every destination and range before importing or editing anything.
  for (const action of all) {
    const uid = normalizedGuid(action.nested_sequence_uid);
    if (action.mode !== "nested_sequence" || !uid || seen.has(uid)) {
      throw new Error("Plan must contain one fill per nested sequence. Regenerate the plan.");
    }
    seen.add(uid);
    const dest = sequences.find(s => normalizedGuid(s.guid) === uid);
    if (!dest || dest === template) throw new Error("Nested sequence not found: " + action.nested_sequence);
    const start = Number(action.in_sec);
    const end = Number(action.out_sec);
    const duration = Number(action.slot_duration_sec);
    const mediaDuration = Number(applyDoc.source_duration_sec);
    if (action.in_sec == null || !Number.isFinite(start) || start < 0 ||
        !Number.isFinite(end) || !Number.isFinite(duration) || duration <= 0 ||
        Math.abs(end - start - duration) > 0.000001 ||
        !Number.isFinite(mediaDuration) || mediaDuration <= 0 || end > mediaDuration + 0.000001) {
      throw new Error("Invalid or incomplete source range: " + action.slot_id);
    }
    if (Number(action.video_track_index) !== 0 || Number(action.overwrite_at_sec) !== 0) {
      throw new Error("Nested fills must start at zero on V1");
    }
    if (await dest.getVideoTrackCount() < 1) throw new Error("Nested sequence has no V1");
    const footage = await clipTrackItems(dest, 0);
    if (footage.length > 1 || (footage.length === 1 &&
        (await footage[0].isAdjustmentLayer() || await readTickSeconds(footage[0], "getStartTime") !== 0 ||
         await readTickSeconds(footage[0], "getEndTime") > duration + 0.000001))) {
      throw new Error("Nested V1 is not a single source placeholder: " + action.nested_sequence);
    }
    prepared.push({ action, dest, start, end });
  }
  if (!prepared.length) throw new Error("No matching nested sequence in plan");
  const intro = applyDoc.intro_fill;
  if (intro) {
    const start = Number(intro.in_sec), end = Number(intro.out_sec);
    const length = Number(intro.slot_duration_sec), track = Number(intro.video_track_index);
    if (intro.mode !== "intro_gap" || intro.overwrite_at_sec !== 0 ||
        !Number.isFinite(start) || start < 0 || !Number.isFinite(end) ||
        !Number.isFinite(length) || length <= 0 || Math.abs(end - start - length) > 1e-6 ||
        Math.abs(Number(intro.end_sec) - length) > 1e-6 || end > Number(applyDoc.source_duration_sec) ||
        !Number.isInteger(track) || track < 0 || track >= await template.getVideoTrackCount() ||
        intro.audio_track_index !== -1) throw new Error("Invalid intro fill; regenerate the plan");
    const firstScene = Math.min(...all.flatMap(a => a.instances.map(i => i.start_sec)));
    if (Math.abs(firstScene - length) > 1e-6) throw new Error("Intro no longer ends at scene 1");
    for (let t = 0; t < await template.getVideoTrackCount(); t++) {
      for (const item of await clipTrackItems(template, t)) {
        if (await readTickSeconds(item, "getStartTime") < length - 1e-6 &&
            await readTickSeconds(item, "getEndTime") > 0) {
          // A retry may replace precisely our previous source-only intro.
          const clip = ppro.ClipProjectItem.cast(await item.getProjectItem());
          const path = clip && toForwardSlashPath(await clip.getMediaFilePath()).toLowerCase();
          if (t !== track || await item.isAdjustmentLayer() ||
              await readTickSeconds(item, "getStartTime") !== 0 ||
              Math.abs(await readTickSeconds(item, "getEndTime") - length) > 1e-6 ||
              ![applyDoc.video_only_source, applyDoc.source_video, applyDoc.source_media]
                .filter(Boolean).some(p => path === toForwardSlashPath(p).toLowerCase())) {
            throw new Error("Intro gap contains existing content; no clips were changed");
          }
        }
      }
    }
    prepared.push({ action: {...intro, nested_sequence: "intro"}, dest: template, start, end });
  }
  const characterMarkers = await prepareCharacterMarkers(template, all, intro);
  const source = await ensureSourceClip(project, applyDoc.video_only_source);
  // These source-mark APIs are available in 26.0.2. Save marks before touching them.
  const mediaType = ppro.Constants.MediaType.VIDEO;
  const savedIn = await source.clipItem.getInPoint(mediaType);
  const savedOut = await source.clipItem.getOutPoint(mediaType);
  if (!savedIn || savedIn.ticks == null || !savedOut || savedOut.ticks == null) {
    throw new Error("Could not read source in/out points; no nested clips were changed.");
  }
  let filled = 0;
  let marksTouched = false;
  let failure = null;
  try {
    for (const entry of prepared) {
      const { action, dest, start, end } = entry;
      const inT = await makeTick(start);
      const outT = await makeTick(end);
      const editor = ppro.SequenceEditor.getEditor(dest);
      const zero = await makeTick(0);
      const audioBefore = await snapshotAudio(dest);
      // Commit the range first: overwrite must see this range, not the previous cut.
      marksTouched = true;
      const range = runNamedAction(project, "Set source range " + action.slot_id, () =>
        source.clipItem.createSetInOutPointsAction(inT, outT)
      );
      if (!range.ok) throw new Error("Source range failed: " + (range.error || action.slot_id));
      const actualIn = await source.clipItem.getInPoint(mediaType);
      const actualOut = await source.clipItem.getOutPoint(mediaType);
      if (tickId(actualIn) !== tickId(inT) || tickId(actualOut) !== tickId(outT)) {
        throw new Error("Premiere did not accept the source range for " + action.slot_id);
      }
      // The fill asset has no audio stream. The overwrite API does not document
      // -1 as an audio-exclusion flag, so that argument alone is insufficient.
      const result = runNamedAction(project, "Fill nested sequence " + action.nested_sequence, () =>
        editor.createOverwriteItemAction(source.projectItem, zero, Number(action.video_track_index), -1)
      );
      if (!result.ok) throw new Error("Fill failed after " + filled + " nests: " + (result.error || action.slot_id));
      await verifyAudioUnchanged(dest, audioBefore);
      filled++;
      log("Filled " + action.nested_sequence + " from " + start.toFixed(3) + " to " + end.toFixed(3) + "s");
    }
  } catch (err) {
    failure = err;
  } finally {
    if (marksTouched) {
      const restored = runNamedAction(project, "Restore source in/out points", () =>
        source.clipItem.createSetInOutPointsAction(savedIn, savedOut)
      );
      if (!restored.ok) {
        const message = "Could not restore source in/out points: " + (restored.error || "transaction failed");
        failure = new Error((failure ? failure.message + "; " : "") + message);
      }
    }
  }
  if (failure) throw failure;
  await verifyAudioUnchanged(template, templateAudio);
  if (characterMarkers) await applyCharacterMarkers(project, characterMarkers);
  log("Filled " + all.length + " nested sequences" + (intro ? " + 1 video intro" : "") + ". Configured timeline audio retained, including audio at 0s. Template cuts, repeat in/out points and adjustment layers retained. Project not saved.");
}

async function prepareCharacterMarkers(template, actions, intro) {
  const subjects = actions.concat(intro ? [intro] : []).filter(a => a.character_selection);
  if (!subjects.length) return null;
  if (!ppro.Markers || typeof ppro.Markers.getMarkers !== "function") {
    throw new Error("Character review markers require the Premiere Markers API; no fills were applied.");
  }
  const collection = await ppro.Markers.getMarkers(template);
  if (!collection || typeof collection.createAddMarkerAction !== "function" ||
      typeof collection.createRemoveMarkerAction !== "function") {
    throw new Error("Premiere cannot create character review markers; no fills were applied.");
  }
  const prefixes = subjects.map(a => "AUTOEDIT_CHARACTER:" + a.slot_id + ":");
  const existing = await collection.getMarkers();
  const removals = [];
  for (const marker of existing) {
    const name = await marker.getName();
    if (prefixes.some(prefix => String(name).startsWith(prefix))) removals.push(marker);
  }
  const additions = [];
  for (const action of subjects) {
    const selection = action.character_selection;
    if (!["main", "uncertain", "supporting"].includes(selection.decision) ||
        typeof selection.needs_review !== "boolean" ||
        selection.needs_review !== (selection.decision !== "main")) {
      throw new Error("Invalid character decision for " + action.slot_id);
    }
    if (!selection.needs_review) continue;
    const instances = action.mode === "intro_gap"
      ? [{ start_sec: action.overwrite_at_sec, end_sec: action.end_sec }]
      : action.instances;
    if (!instances || !instances.length) throw new Error("Missing marker instances for " + action.slot_id);
    for (let index = 0; index < instances.length; index++) {
      const instance = instances[index];
      const start = Number(instance.start_sec), end = Number(instance.end_sec);
      if (!Number.isFinite(start) || start < 0 || !Number.isFinite(end) || end <= start) {
        throw new Error("Invalid marker timing for " + action.slot_id);
      }
      additions.push({
        name: "AUTOEDIT_CHARACTER:" + action.slot_id + ":" + index,
        start: await makeTick(start), duration: await makeTick(end - start),
        comments: JSON.stringify({ reason: selection.reason, characters: selection.character_ids,
          rank: selection.best_rank, main_fraction: selection.main_fraction,
          confidence: selection.identity_confidence, source_in_sec: action.in_sec,
          source_out_sec: action.out_sec, nested_sequence: action.nested_sequence || "intro" }),
      });
    }
  }
  return { collection, removals, additions };
}

async function applyCharacterMarkers(project, prepared) {
  const { collection, removals, additions } = prepared;
  if (!removals.length && !additions.length) return;
  let ok = false;
  try {
    project.lockedAccess(() => {
      ok = project.executeTransaction(compound => {
        for (const marker of removals) compound.addAction(collection.createRemoveMarkerAction(marker));
        for (const marker of additions) {
          compound.addAction(collection.createAddMarkerAction(marker.name, "Comment", marker.start,
                                                              marker.duration, marker.comments));
        }
      }, "Update character review markers");
    });
  } catch (error) {
    throw new Error("Video fills completed, but character markers failed: " + error.message);
  }
  if (!ok) throw new Error("Video fills completed, but Premiere rejected character markers. Reapply to retry.");
  log("Character review: " + additions.length + " markers on the template timeline");
}

async function importMixdown() {
  if (applying) throw new Error("Wait for the current fill/import to finish.");
  applying = true;
  try {
    await importMixdownImpl(true);
  } finally {
    applying = false;
  }
}

async function importMixdownImpl(alwaysPickFile = false) {
  if (!applyDoc) throw new Error("Load an edit plan before importing a mixdown");
  if (!alwaysPickFile && !applyDoc.mixdown_wav) throw new Error("No mixdown path in plan");
  const duration = Number(applyDoc.mixdown_duration_sec);
  if (!Number.isFinite(duration) || duration <= 0) throw new Error("Regenerate plan: mixdown duration is missing");
  const project = await getProject();
  const root = await project.getRootItem();
  const templates = (await project.getSequences()).filter(s => s.name === applyDoc.template_sequence);
  if (templates.length !== 1) throw new Error("Template sequence missing or ambiguous");
  const template = templates[0];
  // The manual mixdown button always asks so it cannot silently use an old export.
  const previousWav = toForwardSlashPath(applyDoc.mixdown_wav);
  let wav = alwaysPickFile
    ? toForwardSlashPath(await pickFile("Choose the Cubase mixdown WAV to import.", ["wav"]))
    : previousWav;
  // Reuse our mix track on retry; otherwise choose an empty audio track after A1.
  let audioIndex = -1;
  let emptyIndex = -1;
  let previousMix = null;
  for (let i = 1; i < await template.getAudioTrackCount(); i++) {
    const track = await template.getAudioTrack(i);
    const items = await track.getTrackItems(ppro.Constants.TrackItemType.CLIP, false);
    if (!items.length && emptyIndex < 0) emptyIndex = i;
    if (items.length === 1) {
      const item = ppro.ClipProjectItem.cast(await items[0].getProjectItem());
      const path = item ? toForwardSlashPath(await item.getMediaFilePath()).toLowerCase() : "";
      if ((path === wav.toLowerCase() || path === previousWav.toLowerCase()) &&
          await readTickSeconds(items[0], "getStartTime") === 0) {
        audioIndex = i;
        previousMix = items[0];
      }
    }
  }
  if (audioIndex < 0) audioIndex = emptyIndex;
  if (audioIndex < 0) throw new Error("No empty audio track after A1. Add an empty audio track and retry Import mixdown WAV.");
  let clip = await findClipByMediaPath(project, wav);
  if (!clip) {
    const imported = await importByPath(project, root, wav);
    if (imported) clip = await findClipByMediaPath(project, wav);
    if (!clip && !alwaysPickFile) {
      wav = toForwardSlashPath(await pickFile("Choose the generated mixdown WAV.", ["wav"]));
      clip = await findClipByMediaPath(project, wav);
      if (!clip) {
        const pickedImport = await importByPath(project, root, wav);
        if (pickedImport) clip = await findClipByMediaPath(project, wav);
      }
    }
  }
  if (!clip) throw new Error("Mixdown project item could not be found");
  const clipItem = ppro.ClipProjectItem.cast(clip);
  if (typeof clipItem.refreshMedia === "function" && await clipItem.refreshMedia() === false) {
    throw new Error("Premiere could not refresh the generated mixdown");
  }
  const editor = ppro.SequenceEditor.getEditor(template);
  const at = await makeTick(0);
  const end = await makeTick(duration);
  const savedIn = await clipItem.getInPoint(ppro.Constants.MediaType.AUDIO);
  const savedOut = await clipItem.getOutPoint(ppro.Constants.MediaType.AUDIO);
  if (!savedIn || !savedOut) throw new Error("Could not read mixdown source marks");
  try {
    const range = runNamedAction(project, "Set full mixdown range", () => clipItem.createSetInOutPointsAction(at, end));
    if (!range.ok) throw new Error("Could not set mixdown range: " + (range.error || "transaction failed"));
    if (previousMix) {
      // Remove only our previous mix, without ripple; a shorter new mix leaves no old tail.
      let ok = false;
      project.lockedAccess(() => {
        ppro.TrackItemSelection.createEmptySelection(selection => {
          if (selection.addItem(previousMix, false) === false) throw new Error("Could not select previous mixdown");
          const remove = editor.createRemoveItemsAction(selection, false, ppro.Constants.MediaType.AUDIO, false);
          const overwrite = editor.createOverwriteItemAction(ppro.ProjectItem.cast(clip) || clip, at, -1, audioIndex);
          ok = project.executeTransaction(compound => {
            compound.addAction(remove);
            compound.addAction(overwrite);
          }, "Replace AutoEdit mixdown");
        });
      });
      if (!ok) throw new Error("Mixdown replacement transaction failed");
    } else {
      const result = runNamedAction(project, "Place mixdown audio", () =>
        editor.createOverwriteItemAction(ppro.ProjectItem.cast(clip) || clip, at, -1, audioIndex)
      );
      if (!result.ok) throw new Error("Mixdown placement failed: " + (result.error || "transaction failed"));
    }
  } finally {
    const restored = runNamedAction(project, "Restore mixdown marks", () => clipItem.createSetInOutPointsAction(savedIn, savedOut));
    if (!restored.ok) log("Could not restore mixdown source marks: " + (restored.error || "transaction failed"));
  }
  if (alwaysPickFile) applyDoc.mixdown_wav = wav;
  log("Imported mixdown on A" + (audioIndex + 1) + " of " + applyDoc.template_sequence + " at 0s; A1 beat retained.");
}

function bind(id, fn) {
  document.getElementById(id).addEventListener("click", async () => {
    try {
      await fn();
    } catch (err) {
      log("error: " + (err && err.message ? err.message : err));
    }
  });
}

async function clipTrackItems(sequence, videoTrackIndex) {
  const track = await sequence.getVideoTrack(videoTrackIndex);
  if (typeof track.getTrackItems === "function") {
    return await track.getTrackItems(ppro.Constants.TrackItemType.CLIP, false);
  }
  return await track.getItems();
}

async function readItemName(item) {
  try {
    if (typeof item.getName === "function") return String(await item.getName());
  } catch (_) {}
  return String(item.name || "");
}

async function readTickSeconds(item, method) {
  const tick = await item[method]();
  if (tick && typeof tick.then === "function") return (await tick).seconds;
  return tick.seconds;
}

bind("inspect", inspectProject);
bind("load", loadPlan);
bind("apply", () => applyPlan());
bind("mix", importMixdown);
log("AutoEdit 0.3.1 ready (Premiere 26.0+). Video-only fills preserve configured audio from 0s. Load edit-plan.json, then Fill nested clips. Project is not saved automatically.");
