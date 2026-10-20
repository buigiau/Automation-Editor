const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(format = 'payload', options = {}) {
  const edits = [];
  const cuts = [];
  const rootItems = [];
  const mixEdits = [];
  const removed = [];
  const imports = [];
  const clones = [];
  const markers = [{ name: 'User note', getName() { return this.name; } }];
  let filePickerCalls = 0;
  let refreshCount = 0;
  const tick = seconds => ({ seconds, ticks: String(Math.round(seconds * 254016000000)) });
  const nestTracks = Array.from({ length: 14 }, () => [[], []]);
  const videoItem = values => {
    const data = { name: 'existing.mp4', start: 0, end: 4, inPoint: 0, outPoint: 4,
      adjustment: false, disabled: false, effects: [], ...values };
    return { data, name: data.name, getName: async () => data.name,
      getProjectItem: async () => ({ isSequence: async () => false }),
      isAdjustmentLayer: async () => data.adjustment, isDisabled: async () => data.disabled,
      getStartTime: async () => tick(data.start), getEndTime: async () => tick(data.end),
      getInPoint: async () => tick(data.inPoint), getOutPoint: async () => tick(data.outPoint) };
  };
  const nests = Array.from({ length: 14 }, (_, i) => ({
    name: String(i + 1), guid: { toString: () => `uid-${i}` },
    getVideoTrackCount: async () => nestTracks[i].length,
    getVideoTrack: async index => ({ getTrackItems: async () => nestTracks[i][index] }),
    getAudioTrackCount: async () => 0,
  }));
  const instances = [{ start_sec: 0, end_sec: 1, source_in_sec: 0, source_out_sec: 1 },
                     { start_sec: 1, end_sec: 2, source_in_sec: 0, source_out_sec: 1 }];
  if (options.intro) instances.forEach(i => { i.start_sec++; i.end_sec++; });
  let mixIn = tick(2), mixOut = tick(8);
  let mixPath = 'C:/output/mixdown.wav';
  const mixClip = { name: 'mixdown.wav', getMediaFilePath: async () => mixPath,
    getInPoint: async () => mixIn, getOutPoint: async () => mixOut,
    createSetInOutPointsAction: (start, end) => () => { mixIn = start; mixOut = end; },
    refreshMedia: async () => { refreshCount++; return true; },
  };
  if (options.existingMix) rootItems.push(mixClip);
  const audioItem = (clip, bounds = [0, 65.8, 1, 66.8]) => ({
    bounds, getProjectItem: async () => clip, getStartTime: async () => tick(bounds[0]),
    getEndTime: async () => tick(bounds[1]), getInPoint: async () => tick(bounds[2]), getOutPoint: async () => tick(bounds[3]),
  });
  const occupied = audioItem({ getMediaFilePath: async () => 'C:/music.wav' });
  const audioTracks = [[occupied], [occupied], options.existingMix ? [audioItem(mixClip)] : options.noEmptyTrack ? [occupied] : []];
  const template = { name: 'PJ 5 - demo', guid: 'main',
    getVideoTrackCount: async () => 3,
    getAudioTrackCount: async () => audioTracks.length,
    getAudioTrack: async i => ({ getTrackItems: async () => audioTracks[i], isMuted: async () => i === 1 }),
    getVideoTrack: async index => ({ getTrackItems: async () => index !== 1 ? [] : [
      ...(options.occupiedIntro ? [{ getStartTime: async () => tick(0), getEndTime: async () => tick(.5),
        getProjectItem: async () => ({ isSequence: async () => false, getMediaFilePath: async () => 'C:/existing.mp4' }),
        isAdjustmentLayer: async () => false }] : []), ...nests.flatMap(nest => instances.map(row => ({
      getProjectItem: async () => ({ isSequence: async () => true, getSequence: async () => nest }),
      getStartTime: async () => tick(row.start_sec), getEndTime: async () => tick(row.end_sec),
      getInPoint: async () => tick(row.source_in_sec), getOutPoint: async () => tick(row.source_out_sec),
    })))] }),
  };
  let sourceIn = tick(7);
  let sourceOut = tick(97);
  const source = { name: 'video-only.mov', hasAudio: Boolean(options.unsafeFillAsset),
    getMediaFilePath: async () => 'C:/video-only.mov',
    getInPoint: async mediaType => {
      assert.equal(mediaType, 1);
      return options.unreadableMarks ? null : sourceIn;
    },
    getOutPoint: async mediaType => { assert.equal(mediaType, 1); return sourceOut; },
    // Simulate 26.0.2: createSubClipAction is deliberately absent.
    createSetInOutPointsAction: (start, end) => () => {
      const restoring = start.seconds === 7 && end.seconds === 97;
      if (restoring && options.failRestore) throw Error('restore rejected');
      if (!restoring && options.failRange) throw Error('range rejected');
      cuts.push({ start, end });
      if (restoring || !options.ignoreRange) { sourceIn = start; sourceOut = end; }
    },
  };
  rootItems.push(source);
  const secondSource = { name: 'second-video-only.mov', currentIn: tick(3), currentOut: tick(55),
    getMediaFilePath: async () => 'C:/second-video-only.mov',
    getInPoint: async () => options.unreadableSecond ? null : secondSource.currentIn,
    getOutPoint: async () => secondSource.currentOut,
    createSetInOutPointsAction: (start, end) => () => {
      secondSource.currentIn = start; secondSource.currentOut = end;
    },
  };
  if (options.multiple) rootItems.push(secondSource);
  rootItems.push({ name: 'source.mp4', hasAudio: true, getMediaFilePath: async () => 'C:/source.mp4' });
  const project = {
    getSequences: async () => [template, ...nests, { name: '1', guid: 'unused' }],
    getRootItem: async () => ({ getItems: async () => rootItems }),
    lockedAccess: fn => fn(),
    executeTransaction: fn => { fn({ addAction: action => action() }); return true; },
    importFiles: async paths => { imports.push(...paths); mixPath = paths[0]; rootItems.push(mixClip); return true; },
  };
  const actions = nests.map((nest, i) => ({
    slot_id: `slot-${i}`, mode: 'nested_sequence', nested_sequence: nest.name,
    nested_sequence_uid: `uid-${i}`, in_sec: i * 4, out_sec: i * 4 + 4,
    slot_duration_sec: 4, video_track_index: 0, overwrite_at_sec: 0,
    template_video_track_index: 1, instances,
  }));
  const payload = { fill_policy: 'nested_sequences_v2', template_sequence: template.name,
    source_video: 'C:/source.mp4', source_duration_sec: 100, fill_nested_sequences: actions,
    video_only_source: 'C:/video-only.mov',
    template_audio_tracks: [{ index: 0, items: [{ start_sec: 0, end_sec: 65.8, source_in_sec: 1, source_out_sec: 66.8 }] }],
    import_mixdown: Boolean(options.autoAudio), mixdown_wav: 'C:/output/mixdown.wav', mixdown_duration_sec: 20 };
  if (options.characters) actions.forEach((action, i) => {
    action.character_selection = { decision: i ? 'main' : 'supporting', needs_review: i === 0,
      reason: i ? 'main-character' : 'supporting-character-fallback', character_ids: ['character-003'],
      best_rank: 3, main_fraction: i ? 1 : 0, identity_confidence: 1 };
  });
  if (options.intro) payload.intro_fill = { slot_id: 'intro', mode: 'intro_gap', in_sec: 80, out_sec: 81,
    overwrite_at_sec: 0, slot_duration_sec: 1, end_sec: 1, video_track_index: 1, audio_track_index: -1 };
  if (options.characters && options.intro) payload.intro_fill.character_selection = {
    ...actions[0].character_selection, decision: 'uncertain', reason: 'uncertain-character-identity' };
  if (options.multiple) {
    payload.source_videos = [
      { path: 'C:/source.mp4', duration_sec: 100, video_only_source: 'C:/video-only.mov' },
      { path: 'C:/second.mp4', duration_sec: 60, video_only_source: 'C:/second-video-only.mov' },
    ];
    actions.forEach((action, i) => {
      const entry = payload.source_videos[i < 7 ? 0 : 1];
      Object.assign(action, { source_media: entry.path, video_only_source: entry.video_only_source,
        source_duration_sec: entry.duration_sec, in_sec: (i % 7) * 4, out_sec: (i % 7) * 4 + 4 });
    });
    if (payload.intro_fill) Object.assign(payload.intro_fill, {
      source_video: 'C:/second.mp4', video_only_source: 'C:/second-video-only.mov',
      source_duration_sec: 60, in_sec: 40, out_sec: 41,
    });
  }
  const fullPlan = { intro_fill: payload.intro_fill, template_audio_tracks: payload.template_audio_tracks,
    project: { template_sequence: template.name, source_video: 'C:/source.mp4', video_only_source: 'C:/video-only.mov',
      source_videos: payload.source_videos },
    video_analysis_meta: { duration_sec: 100 }, slots: actions.map(a => ({ id: a.slot_id,
      source_video: a.source_media,
      video: { in_sec: a.in_sec, out_sec: a.out_sec, character_selection: a.character_selection,
        video_only_source: a.video_only_source, source_duration_sec: a.source_duration_sec }, premiere: {
        mode: a.mode, nested_sequence: a.nested_sequence, nested_sequence_uid: a.nested_sequence_uid,
        duration_sec: 4, instances, video_track_index: 1, fill_video_track_index: 0,
      } })) };
  let doc = format === 'payload' ? payload : fullPlan;
  if (options.samplesOnly) doc.operation_scope = 'sample_assignment_only';
  const ppro = {
    Markers: { getMarkers: async owner => {
      assert.equal(owner, template);
      return { getMarkers: () => markers,
        createRemoveMarkerAction: marker => () => markers.splice(markers.indexOf(marker), 1),
        createAddMarkerAction: (name, type, start, duration, comments) => {
          if (options.failMarkers) throw Error('marker rejected');
          return () => markers.push({ name, type, start, duration, comments, getName() { return this.name; } });
        } };
    } },
    Project: { getActiveProject: async () => project },
    ClipProjectItem: { cast: x => x }, ProjectItem: { cast: x => x },
    FolderItem: { cast: () => null },
    TickTime: { createWithSeconds: tick },
    Constants: { TrackItemType: { CLIP: 1 }, MediaType: { VIDEO: 1, AUDIO: 2 } },
    TrackItemSelection: { createEmptySelection: callback => {
      const selection = { items: [], addItem: item => { selection.items.push(item); return true; } };
      callback(selection); return true;
    } },
    SequenceEditor: { getEditor: dest => {
      const editor = {
      createCloneTrackItemAction: (item, offset, vOffset, aOffset, align, insert) => {
        assert.equal(offset.seconds, 0); assert.equal(vOffset, 1); assert.equal(aOffset, 0);
        assert.equal(align, true); assert.equal(insert, false);
        return () => {
          if (options.failClone) throw Error('clone rejected');
          if (options.ignoreClone) return;
          const tracks = nestTracks[nests.indexOf(dest)];
          while (tracks.length < 2) tracks.push([]);
          const copy = videoItem(JSON.parse(JSON.stringify(item.data)));
          if (options.wrongCloneTiming) copy.data.end += 1;
          tracks[1].push(copy); clones.push({ dest, original: item, copy });
        };
      },
      createOverwriteItemAction: (item, time, v, a) => {
        if (item === mixClip) {
          if (options.failMix) throw Error('audio overwrite rejected');
          const start = mixIn.seconds, end = mixOut.seconds;
          return () => { mixEdits.push({ dest, item, time, v, a, start, end }); audioTracks[a] = [audioItem(mixClip)]; };
        }
        if (options.failOverwriteAt === edits.length) throw Error('overwrite rejected');
        const start = item === secondSource ? secondSource.currentIn.seconds : sourceIn.seconds;
        const end = item === secondSource ? secondSource.currentOut.seconds : sourceOut.seconds;
        return () => {
          edits.push({ dest, item, time, v, a, start, end });
          const index = nests.indexOf(dest);
          if (index >= 0) nestTracks[index][v] = [videoItem({ end: end-start, inPoint: start, outPoint: end })];
          // Simulate a host routing a source's audio to A1 despite a=-1.
          if (dest === template && item.hasAudio) audioTracks[0][0].bounds[0] = end-start;
        };
      },
      createRemoveItemsAction: (selection, ripple, mediaType, shift) => () => {
        assert.equal(ripple, false); assert.equal(shift, false);
        assert([1, 2].includes(mediaType));
        if (mediaType === 1) {
          const tracks = nestTracks[nests.indexOf(dest)];
          tracks.forEach((items, index) => { tracks[index] = items.filter(item => !selection.items.includes(item)); });
        }
        removed.push(...selection.items);
      },
      };
      if (options.noClone) delete editor.createCloneTrackItemAction;
      return editor;
    } },
  };
  if (options.missingMarkers) delete ppro.Markers;
  const elements = new Map();
  const context = vm.createContext({
    require: name => name === 'premierepro' ? ppro : { storage: { localFileSystem: {
      getFileForOpening: async ({ types }) => {
        filePickerCalls++;
        return types.includes('json')
          ? { read: async () => JSON.stringify(doc) }
          : { nativePath: options.pickedMixdownPath || 'C:/output/mixdown.wav' };
      },
    } } },
    document: { getElementById: id => {
      if (!elements.has(id)) elements.set(id, { textContent: '', checked: false, addEventListener: () => {} });
      return elements.get(id);
    } },
    console: { log: () => {} },
  });
  vm.runInContext(fs.readFileSync('plugins/premiere-uxp/main.js', 'utf8'), context);
  return { context, edits, cuts, payload, fullPlan, project, template, nests, nestTracks, videoItem, clones, tick, source, secondSource, mixEdits, removed, imports, audioTracks, markers,
    refreshCount: () => refreshCount, mixMarks: () => [mixIn.seconds, mixOut.seconds],
    filePickerCalls: () => filePickerCalls,
    marks: () => [sourceIn.seconds, sourceOut.seconds],
    load: () => context.loadPlan() };
}

for (const format of ['payload', 'plan']) {
  test(`multiple sources fill from each file and restore all source marks via ${format}`, async () => {
    const env = setup(format, { multiple: true, intro: true });
    await env.load();
    await env.context.applyNestedPlan();
    assert.equal(env.edits.length, 15);
    assert.equal(env.edits.filter(e => e.item === env.source).length, 7);
    assert.equal(env.edits.filter(e => e.item === env.secondSource).length, 8);
    assert.deepEqual(env.marks(), [7, 97]);
    assert.equal(env.secondSource.currentIn.seconds, 3);
    assert.equal(env.secondSource.currentOut.seconds, 55);
    assert.equal(env.edits[7].start, 0);
    assert.equal(env.edits[7].end, 4);
    assert.equal(env.edits[14].start, 40);
    assert.equal(env.edits[14].dest, env.template);
  });
}

test('unreadable second source marks abort before any fill', async () => {
  const env = setup('payload', { multiple: true, unreadableSecond: true });
  await env.load();
  await assert.rejects(env.context.applyNestedPlan(), /Could not read source/);
  assert.equal(env.edits.length, 0);
  assert.deepEqual(env.marks(), [7, 97]);
});

test('multiple sources restore both marks after a later overwrite fails', async () => {
  const env = setup('payload', { multiple: true, failOverwriteAt: 8 });
  await env.load();
  await assert.rejects(env.context.applyNestedPlan(), /Fill failed after 8 nests/);
  assert.deepEqual(env.marks(), [7, 97]);
  assert.equal(env.secondSource.currentIn.seconds, 3);
  assert.equal(env.secondSource.currentOut.seconds, 55);
});

test('a range exceeding its own source duration aborts before any fill', async () => {
  const env = setup('payload', { multiple: true });
  env.payload.fill_nested_sequences[13].in_sec = 58;
  env.payload.fill_nested_sequences[13].out_sec = 62;
  await env.load();
  await assert.rejects(env.context.applyNestedPlan(), /Invalid or incomplete source range/);
  assert.equal(env.edits.length, 0);
});

test('missing per-action source cannot fall back to the first video', async () => {
  const env = setup('payload', { multiple: true });
  delete env.payload.fill_nested_sequences[13].video_only_source;
  await env.load();
  await assert.rejects(env.context.applyNestedPlan(), /Missing video-only source/);
  assert.equal(env.edits.length, 0);
});

function recursiveSetup(format = 'payload') {
  const env = setup(format);
  const { template, nests, tick } = env;
  template.name = 'BEAT MAU';
  const [a, b] = nests;
  a.name = '4'; b.name = '5';
  const item = row => ({
    getProjectItem: async () => ({ isSequence: async () => true, getSequence: async () => row.sequence }),
    getStartTime: async () => tick(row.start_sec), getEndTime: async () => tick(row.end_sec),
    getInPoint: async () => tick(row.source_in_sec), getOutPoint: async () => tick(row.source_out_sec),
  });
  const inner = [
    { sequence: a, start_sec: 0, end_sec: 2, source_in_sec: 3, source_out_sec: 5 },
    { sequence: a, start_sec: 2, end_sec: 4, source_in_sec: 0, source_out_sec: 2 },
  ];
  const wrapper = { name: 'Nested Sequence 17', guid: 'wrap',
    getVideoTrackCount: async () => 2,
    getVideoTrack: async index => ({ getTrackItems: async () => index === 1 ? inner.map(item) : [] }),
    getAudioTrackCount: async () => 0 };
  const rootTracks = [[], [
    { sequence: wrapper, start_sec: 10, end_sec: 12, source_in_sec: 1, source_out_sec: 3 },
    { sequence: b, start_sec: 12, end_sec: 13, source_in_sec: 0, source_out_sec: 1 },
  ], [], [{ sequence: a, start_sec: 12, end_sec: 12.5, source_in_sec: .5, source_out_sec: 1 }]];
  template.getVideoTrackCount = async () => rootTracks.length;
  template.getVideoTrack = async index => ({ getTrackItems: async () => rootTracks[index].map(item) });
  env.project.getSequences = async () => [template, wrapper, a, b];
  const structure = [
    ...rootTracks.map((rows, index) => ({ sequence_uid: 'main', sequence_name: template.name,
      video_track_index: index, instances: rows.map(({ sequence, ...row }) => ({ ...row, nested_sequence_uid: String(sequence.guid) })) })),
    ...[[], inner].map((rows, index) => ({ sequence_uid: 'wrap', sequence_name: wrapper.name,
      video_track_index: index, instances: rows.map(({ sequence, ...row }) => ({ ...row, nested_sequence_uid: String(sequence.guid) })) })),
  ];
  const instances = [
    [{ start_sec: 10, end_sec: 11, source_in_sec: 4, source_out_sec: 5, template_video_track_index: 1 },
     { start_sec: 11, end_sec: 12, source_in_sec: 0, source_out_sec: 1, template_video_track_index: 1 },
     { start_sec: 12, end_sec: 12.5, source_in_sec: .5, source_out_sec: 1, template_video_track_index: 3 }],
    [{ start_sec: 12, end_sec: 13, source_in_sec: 0, source_out_sec: 1, template_video_track_index: 1 }],
  ];
  env.payload.template_sequence = template.name;
  env.payload.nested_structure = structure;
  env.payload.fill_nested_sequences = env.payload.fill_nested_sequences.slice(0, 2).map((action, i) => ({
    ...action, nested_sequence: [a, b][i].name, instances: instances[i],
    in_sec: i * 10, out_sec: i * 10 + [5, 1][i], slot_duration_sec: [5, 1][i] }));
  env.fullPlan.project.template_sequence = template.name;
  env.fullPlan.nested_structure = structure;
  env.fullPlan.slots = env.payload.fill_nested_sequences.map(action => ({ id: action.slot_id,
    video: { in_sec: action.in_sec, out_sec: action.out_sec }, premiere: {
      mode: 'nested_sequence', nested_sequence: action.nested_sequence, nested_sequence_uid: action.nested_sequence_uid,
      duration_sec: action.slot_duration_sec, instances: action.instances, video_track_index: 1 } }));
  return { ...env, wrapper, inner, rootTracks };
}

for (const format of ['payload', 'plan']) {
  test(`recursive fills resolve leaves and preserve wrapper cuts via ${format}`, async () => {
    const env = recursiveSetup(format);
    const saved = env.inner.map(i => [i.start_sec, i.end_sec, i.source_in_sec, i.source_out_sec]);
    await env.load(); await env.context.applyPlan();
    assert.equal(env.edits.length, 2);
    assert.deepEqual(env.edits.map(e => e.dest.name), ['4', '5']);
    assert(env.edits.every(e => e.v === 0 && e.time.seconds === 0));
    assert.deepEqual(env.inner.map(i => [i.start_sec, i.end_sec, i.source_in_sec, i.source_out_sec]), saved);
    assert.deepEqual(env.marks(), [7, 97]);
  });
  test(`changed inner nest is rejected before any fill via ${format}`, async () => {
    const env = recursiveSetup(format);
    env.inner[1].source_in_sec = .1;
    await env.load();
    await assert.rejects(env.context.applyPlan(), /Nested structure differs/);
    assert.equal(env.edits.length, 0); assert.equal(env.cuts.length, 0);
  });
}
test('changed V4 overlay is rejected before any recursive fill', async () => {
  const env = recursiveSetup();
  env.rootTracks[3][0].end_sec = 12.6;
  await env.load();
  await assert.rejects(env.context.applyPlan(), /Nested structure differs/);
  assert.equal(env.edits.length, 0);
});
test('incomplete recursive track snapshots are rejected', async () => {
  const env = recursiveSetup();
  env.payload.nested_structure.splice(2, 1); // Missing root V3.
  await env.load();
  await assert.rejects(env.context.applyPlan(), /Nested track count differs/);
  assert.equal(env.edits.length, 0);
});
test('legacy wrapper-target plan is rejected despite its empty V1', async () => {
  const env = recursiveSetup();
  env.payload.nested_structure = null;
  env.payload.fill_nested_sequences = [{ ...env.payload.fill_nested_sequences[0],
    nested_sequence: env.wrapper.name, nested_sequence_uid: 'wrap',
    instances: [{ start_sec: 10, end_sec: 12, source_in_sec: 1, source_out_sec: 3 }] }];
  await env.load();
  await assert.rejects(env.context.applyPlan(), /Fill target contains nested cuts/);
  assert.equal(env.edits.length, 0); assert.equal(env.cuts.length, 0);
});
test('recursive review markers use global timing including V4 instances', async () => {
  const env = recursiveSetup();
  env.payload.fill_nested_sequences[0].character_selection = {
    decision: 'supporting', needs_review: true, reason: 'supporting-character-fallback' };
  await env.load(); await env.context.applyPlan();
  const owned = env.markers.filter(m => m.name.startsWith('AUTOEDIT_CHARACTER:'));
  assert.deepEqual(owned.map(m => m.start.seconds), [10, 11, 12]);
  assert.deepEqual(owned.map(m => m.duration.seconds), [1, 1, .5]);
});

for (const format of ['payload', 'plan']) {
  test(`AI character estimates fill all nests and retain review markers via ${format}`, async () => {
    const env = setup(format);
    for (const action of env.payload.fill_nested_sequences) {
      const selection = { method: 'qwen-local-visual', character_id: 'red-person',
        character_ids: ['red-person'], decision: 'uncertain', needs_review: true,
        reference_fallback: false, reason: 'Appearance agreed at two sampled cut frames; inspect full playback.' };
      action.character_selection = selection;
      env.fullPlan.slots.find(s => s.id === action.slot_id).video.character_selection = selection;
    }
    await env.load(); await env.context.applyPlan();
    assert.equal(env.edits.length, 14);
    assert.deepEqual(env.marks(), [7, 97]);
    const owned = env.markers.filter(m => m.name.startsWith('AUTOEDIT_CHARACTER:'));
    assert.equal(owned.length, 28);
    assert.deepEqual(JSON.parse(owned[0].comments).characters, ['red-person']);
    assert.match(JSON.parse(owned[0].comments).reason, /sampled cut frames/);
    assert.equal(env.markers[0].name, 'User note');
    await env.context.applyPlan();
    assert.equal(env.markers.length, 29);
  });
  test(`incomplete AI character metadata aborts before any fill via ${format}`, async () => {
    const env = setup(format);
    const selection = { method: 'qwen-local-visual', character_id: 'red-person', needs_review: true };
    env.payload.fill_nested_sequences[0].character_selection = selection;
    env.fullPlan.slots[0].video.character_selection = selection;
    await env.load();
    await assert.rejects(env.context.applyPlan(), /Invalid character decision for slot-0/);
    assert.equal(env.edits.length, 0);
    assert.equal(env.cuts.length, 0);
  });
  test(`character review marks repeated instances and intro without duplication via ${format}`, async () => {
    const env = setup(format, { characters: true, intro: true });
    await env.load(); await env.context.applyPlan();
    assert.equal(env.markers.length, 4); // User note + two instances + intro.
    const owned = env.markers.filter(m => m.name.startsWith('AUTOEDIT_CHARACTER:'));
    assert.deepEqual(owned.map(m => m.start.seconds), [1, 2, 0]);
    assert.deepEqual(owned.map(m => m.duration.seconds), [1, 1, 1]);
    assert.equal(JSON.parse(owned[0].comments).reason, 'supporting-character-fallback');
    assert.equal(JSON.parse(owned[2].comments).reason, 'uncertain-character-identity');
    assert(owned.every(m => m.type === 'Comment'));
    await env.context.applyPlan();
    assert.equal(env.markers.length, 4);
    assert.equal(env.markers[0].name, 'User note');
  });
  test(`sample-only plan cannot modify Premiere via ${format}`, async () => {
    const env = setup(format, { samplesOnly: true, autoAudio: true, intro: true });
    await env.load();
    await assert.rejects(env.context.applyPlan());
    await assert.rejects(env.context.importMixdown());
    assert.equal(env.edits.length, 0);
    assert.equal(env.cuts.length, 0);
    assert.equal(env.imports.length, 0);
    assert.equal(env.mixEdits.length, 0);
  });
  test(`intro preserves beat at zero and its source in-point via ${format}`, async () => {
    const env = setup(format, { intro: true });
    await env.load(); await env.context.applyPlan();
    assert.equal(env.edits.length, 15);
    const intro = env.edits[14];
    assert.equal(intro.dest.guid, 'main');
    assert.equal(intro.time.seconds, 0);
    assert.equal(intro.v, 1); assert.equal(intro.a, -1);
    assert.equal(intro.start, 80); assert.equal(intro.end, 81);
    assert.deepEqual(env.marks(), [7, 97]);
    assert.equal(env.mixEdits.length, 0);
    assert.equal(intro.item.hasAudio, false);
    assert.deepEqual(env.audioTracks[0][0].bounds, [0, 65.8, 1, 66.8]);
  });
  test(`14 nested fills preserve the main timeline via ${format}`, async () => {
    const env = setup(format);
    await env.load(); await env.context.applyPlan();
    assert.equal(env.edits.length, 14);
    assert.equal(env.cuts.length, 15); // 14 ranges, then restore the original marks.
    assert.equal(env.source.createSubClipAction, undefined);
    assert.deepEqual(env.marks(), [7, 97]);
    env.edits.forEach((edit, i) => {
      assert.equal(String(edit.dest.guid), `uid-${i}`);
      assert.equal(edit.time.seconds, 0);
      assert.equal(edit.v, 0);
      assert.equal(edit.a, -1);
      assert.equal(edit.item, env.source);
      assert.equal(edit.start, i * 4);
      assert.equal(edit.end, i * 4 + 4);
      assert.equal(env.cuts[i].end.seconds - env.cuts[i].start.seconds, 4);
    });
  });
}
test('missing marker API aborts before video fills', async () => {
  const env = setup('payload', { characters: true, missingMarkers: true });
  await env.load();
  await assert.rejects(env.context.applyPlan(), /Markers API/);
  assert.equal(env.edits.length, 0);
  assert.equal(env.cuts.length, 0);
});
test('marker failure reports completed video fills and restores source marks', async () => {
  const env = setup('payload', { characters: true, failMarkers: true });
  await env.load();
  await assert.rejects(env.context.applyPlan(), /Video fills completed, but character markers failed/);
  assert.equal(env.edits.length, 14);
  assert.deepEqual(env.marks(), [7, 97]);
});
test('reapplying a main-character selection removes old owned markers', async () => {
  const env = setup('payload', { characters: true });
  await env.load(); await env.context.applyPlan();
  env.payload.fill_nested_sequences[0].character_selection.decision = 'main';
  env.payload.fill_nested_sequences[0].character_selection.needs_review = false;
  await env.load(); await env.context.applyPlan();
  assert.deepEqual(env.markers.map(m => m.name), ['User note']);
});
test('occupied intro aborts before editing any of the 14 nests', async () => {
  const env = setup('payload', { intro: true, occupiedIntro: true });
  await env.load(); await assert.rejects(env.context.applyPlan(), /Intro gap contains/);
  assert.equal(env.edits.length, 0); assert.equal(env.cuts.length, 0);
});

test('legacy source with audio is rejected before editing', async () => {
  const env = setup('payload', { intro: true });
  delete env.payload.video_only_source;
  await env.load();
  await assert.rejects(env.context.applyPlan(), /video-only fill source/);
  assert.equal(env.edits.length, 0);
});

test('previously displaced beat is rejected before applying an intro', async () => {
  const env = setup('payload', { intro: true });
  env.audioTracks[0][0].bounds[0] = 1;
  await env.load();
  await assert.rejects(env.context.applyPlan(), /Configured audio on A1 differs/);
  assert.equal(env.edits.length, 0);
});

test('audio mutation by the host never reports a successful intro fill', async () => {
  const env = setup('payload', { intro: true, unsafeFillAsset: true });
  await env.load();
  await assert.rejects(env.context.applyPlan(), /Configured audio changed/);
  assert.deepEqual(env.marks(), [7, 97]);
});
test('invalid last slot is rejected before any cut or overwrite', async () => {
  const env = setup(); env.payload.fill_nested_sequences[13].out_sec = 200;
  await env.load(); await assert.rejects(env.context.applyPlan(), /Invalid/);
  assert.equal(env.edits.length, 0); assert.equal(env.cuts.length, 0);
});

for (const format of ['payload', 'plan']) {
  test(`V1 Adjustment Layer is cloned with styling before the footage fill via ${format}`, async () => {
    const env = setup(format);
    const layer = env.videoItem({ name: 'Adjustment Layer', adjustment: true,
      inPoint: 3600, outPoint: 3604, effects: [{ name: 'Transform', scale: 150, keys: [[0, 100], [4, 150]] }] });
    env.nestTracks[0] = [[layer]]; // Cosita nest 1 has no V2 yet.
    await env.load(); await env.context.applyPlan();
    assert.equal(env.edits.length, 14);
    assert.equal(env.clones.length, 1);
    assert.equal(env.nestTracks[0].length, 2);
    assert.equal(env.nestTracks[0][0][0].data.adjustment, false);
    const copy = env.nestTracks[0][1][0];
    assert.deepEqual(copy.data, layer.data);
    assert.notEqual(copy, layer);
    assert.deepEqual(env.marks(), [7, 97]);
    assert.deepEqual(env.audioTracks[0][0].bounds, [0, 65.8, 1, 66.8]);
    await env.context.applyPlan();
    assert.equal(env.edits.length, 28);
    assert.equal(env.clones.length, 1); // Retry must not stack/double the effects.
    assert.equal(env.nestTracks[0][1][0], copy);
  });
}

test('gaps and multiple source fragments inside the fill range are replaced without touching V2', async () => {
  const env = setup();
  const upper = env.videoItem({ adjustment: true, end: .92, outPoint: 3600.92, inPoint: 3600 });
  env.nestTracks[2] = [[env.videoItem({ start: .2, end: .5 }),
    env.videoItem({ start: 3.52, end: 4, inPoint: 63.76, outPoint: 64.24 })], [upper]];
  await env.load(); await env.context.applyPlan();
  assert.equal(env.edits.length, 14);
  assert.equal(env.nestTracks[2][0].length, 1);
  assert.equal(env.nestTracks[2][0][0].data.start, 0);
  assert.equal(env.nestTracks[2][0][0].data.end, 4);
  assert.equal(env.nestTracks[2][1][0], upper);
  assert.equal(env.clones.length, 0);
});

test('multiple V1 Adjustment Layers retain source marks, disabled state and effects on V2', async () => {
  const env = setup();
  const layers = [env.videoItem({ adjustment: true, end: 2, inPoint: 3600, outPoint: 3602 }),
    env.videoItem({ adjustment: true, start: 2, end: 4, inPoint: 3602, outPoint: 3604,
      disabled: true, effects: [{ name: 'Lumetri', exposure: .8 }] })];
  env.nestTracks[3][0] = layers;
  await env.load(); await env.context.applyPlan();
  assert.equal(env.edits.length, 14);
  assert.deepEqual(env.nestTracks[3][1].map(item => item.data), layers.map(item => item.data));
});

test('a later V1 item outside the fill range aborts every fill before any mutation', async () => {
  const env = setup();
  env.nestTracks[0][0] = [env.videoItem({ adjustment: true })];
  env.nestTracks[13][0] = [env.videoItem({ start: 3, end: 4.04 })];
  await env.load(); await assert.rejects(env.context.applyPlan(), /outside the planned fill: 14/);
  assert.equal(env.edits.length, 0); assert.equal(env.clones.length, 0); assert.equal(env.cuts.length, 0);
});

test('occupied V2 aborts all fills rather than overwriting an existing overlay', async () => {
  const env = setup();
  env.nestTracks[13][0] = [env.videoItem({ adjustment: true })];
  const overlay = env.videoItem({ name: 'existing overlay' });
  env.nestTracks[13][1] = [overlay];
  await env.load(); await assert.rejects(env.context.applyPlan(), /empty V2/);
  assert.equal(env.edits.length, 0); assert.equal(env.clones.length, 0); assert.equal(env.cuts.length, 0);
  assert.equal(env.nestTracks[13][1][0], overlay);
});

test('missing adjustment clone API aborts before any source marks or fills', async () => {
  const env = setup('payload', { noClone: true });
  env.nestTracks[0][0] = [env.videoItem({ adjustment: true })];
  await env.load(); await assert.rejects(env.context.applyPlan(), /cannot clone Adjustment Layers/);
  assert.equal(env.edits.length, 0); assert.equal(env.cuts.length, 0);
});

for (const failure of ['failClone', 'ignoreClone', 'wrongCloneTiming']) {
  test(`${failure} never overwrites the original Adjustment Layer`, async () => {
    const env = setup('payload', { [failure]: true });
    const layer = env.videoItem({ adjustment: true, effects: [{ scale: 170 }] });
    env.nestTracks[0][0] = [layer];
    await env.load(); await assert.rejects(env.context.applyPlan(), /Could not preserve V1 Adjustment Layer/);
    assert.equal(env.edits.length, 0);
    assert.equal(env.nestTracks[0][0][0], layer);
    assert.equal(env.nestTracks[0][1].length, 0);
    assert.deepEqual(env.marks(), [7, 97]);
  });
}

test('a failed overwrite removes staged copies and permits a clean retry', async () => {
  const options = { failOverwriteAt: 0 };
  const env = setup('payload', options);
  const layer = env.videoItem({ adjustment: true, effects: [{ scale: 170 }] });
  env.nestTracks[0][0] = [layer];
  await env.load(); await assert.rejects(env.context.applyPlan(), /Fill failed after 0 nests/);
  assert.equal(env.nestTracks[0][0][0], layer);
  assert.equal(env.nestTracks[0][1].length, 0);
  assert.deepEqual(env.marks(), [7, 97]);
  delete options.failOverwriteAt;
  await env.context.applyPlan();
  assert.equal(env.edits.length, 14);
  assert.equal(env.nestTracks[0][1].length, 1);
  assert.deepEqual(env.nestTracks[0][1][0].data, layer.data);
});
test('overwrite failure stops later fills and restores source marks', async () => {
  const env = setup('payload', { failOverwriteAt: 2 });
  await env.load(); await assert.rejects(env.context.applyPlan(), /Fill failed after 2 nests/);
  assert.equal(env.edits.length, 2);
  assert.deepEqual(env.marks(), [7, 97]);
});
test('rejected source range never overwrites and restores marks', async () => {
  const env = setup('payload', { failRange: true });
  await env.load(); await assert.rejects(env.context.applyPlan(), /range rejected/);
  assert.equal(env.edits.length, 0);
  assert.deepEqual(env.marks(), [7, 97]);
});
test('host ignoring source marks never overwrites', async () => {
  const env = setup('payload', { ignoreRange: true });
  await env.load(); await assert.rejects(env.context.applyPlan(), /did not accept/);
  assert.equal(env.edits.length, 0);
  assert.deepEqual(env.marks(), [7, 97]);
});
test('unreadable original marks abort before edits', async () => {
  const env = setup('payload', { unreadableMarks: true });
  await env.load(); await assert.rejects(env.context.applyPlan(), /Could not read/);
  assert.equal(env.edits.length, 0); assert.equal(env.cuts.length, 0);
});
test('restoration failure is reported alongside the original failure', async () => {
  const env = setup('payload', { failOverwriteAt: 0, failRestore: true });
  await env.load(); await assert.rejects(env.context.applyPlan(), /overwrite rejected.*restore rejected/);
  assert.equal(env.edits.length, 0);
});
test('concurrent apply is rejected', async () => {
  const env = setup(); await env.load();
  const first = env.context.applyPlan();
  await assert.rejects(env.context.applyPlan(), /already running/);
  await first;
  assert.equal(env.edits.length, 14);
});
test('manifest permits Premiere 26.0.2 and the test clip button is absent', () => {
  const manifest = JSON.parse(fs.readFileSync('plugins/premiere-uxp/manifest.json', 'utf8'));
  assert.equal(manifest.host.minVersion, '26.0.0');
  assert.equal(manifest.version, '0.3.2');
  assert.doesNotMatch(fs.readFileSync('plugins/premiere-uxp/index.html', 'utf8'), /testVid2/);
});
test('legacy timeline plan is rejected', async () => {
  const env = setup(); delete env.payload.fill_policy;
  await env.load(); await assert.rejects(env.context.applyPlan(), /Regenerate/);
  assert.equal(env.edits.length, 0);
});
test('a changed template is rejected before any edits', async () => {
  const env = setup(); env.payload.fill_nested_sequences[0].instances = [];
  await env.load(); await assert.rejects(env.context.applyPlan(), /Template differs/);
  assert.equal(env.edits.length, 0); assert.equal(env.cuts.length, 0);
});
test('automatic audio import uses a free audio track and preserves beat and occupied tracks', async () => {
  const env = setup('payload', { autoAudio: true });
  const beat = env.audioTracks[0], music = env.audioTracks[1];
  await env.load(); await env.context.applyPlan();
  assert.equal(env.edits.length, 14);
  assert.equal(env.mixEdits.length, 1);
  const edit = env.mixEdits[0];
  assert.equal(edit.dest.name, 'PJ 5 - demo');
  assert.equal(edit.v, -1); assert.equal(edit.a, 2);
  assert.equal(edit.start, 0); assert.equal(edit.end, 20);
  assert.equal(env.audioTracks[0], beat); assert.equal(env.audioTracks[1], music);
  assert.deepEqual(env.imports, ['C:/output/mixdown.wav']);
  assert.equal(env.filePickerCalls(), 1); // Plan load; checkbox-based automatic import uses its configured path.
  assert.deepEqual(env.mixMarks(), [2, 8]);
});
test('audio retry replaces previous mix without ripple or duplicate audio', async () => {
  const env = setup('payload', { existingMix: true });
  await env.load(); await env.context.importMixdown();
  assert.equal(env.removed.length, 1);
  assert.equal(env.mixEdits.length, 1);
  assert.equal(env.mixEdits[0].a, 2);
  assert.equal(env.imports.length, 0);
  assert.equal(env.refreshCount(), 1);
  assert.equal(env.filePickerCalls(), 2);
});
test('mixdown button imports the WAV selected by the user, not the plan path', async () => {
  const env = setup('payload', { existingMix: true, pickedMixdownPath: 'D:/exports/final_voice.wav' });
  await env.load(); await env.context.importMixdown();
  assert.deepEqual(env.imports, ['D:/exports/final_voice.wav']);
  assert.equal(env.mixEdits.length, 1);
  assert.equal(env.mixEdits[0].a, 2);
  assert.equal(env.removed.length, 1);
});
test('audio placement failure is reported and does not claim success', async () => {
  const env = setup('payload', { autoAudio: true, failMix: true });
  await env.load(); await assert.rejects(env.context.applyPlan(), /Video fills completed, but audio import failed/);
  assert.equal(env.edits.length, 14); assert.equal(env.mixEdits.length, 0);
  assert.deepEqual(env.mixMarks(), [2, 8]);
});
test('audio import refuses to overwrite unrelated occupied tracks', async () => {
  const env = setup('payload', { noEmptyTrack: true });
  await env.load(); await assert.rejects(env.context.importMixdown(), /No empty audio track/);
  assert.equal(env.imports.length, 0); assert.equal(env.mixEdits.length, 0);
});
test('automatic audio can be unchecked without disabling video fills', async () => {
  const env = setup('payload', { autoAudio: true });
  await env.load(); env.context.document.getElementById('autoMix').checked = false;
  await env.context.applyPlan();
  assert.equal(env.edits.length, 14); assert.equal(env.mixEdits.length, 0);
});
