/**
 * Cubase MIDI Remote helper for AutoEdit.
 *
 * Cubase cannot place files on tracks via this API. This script only
 * binds host *commands* (export mixdown, import audio) to a virtual
 * surface so a loopMIDI port named "AutoEdit" can trigger them.
 *
 * Install:
 *   copy this file to
 *   Documents\\Steinberg\\Cubase\\MIDI Remote\\Driver Scripts\\Local\\AutoEdit.js
 * Create a loopMIDI (or similar) port named AutoEdit, then add this
 * controller in Cubase MIDI Remote.
 *
 * CC 20 = File > Export Audio Mixdown
 * CC 21 = File > Import > Audio File
 */

var midiremote_api = require("midiremote_api_v1");

var deviceDriver = midiremote_api.makeDeviceDriver(
  "AutoEdit",
  "PremiereCubaseBridge",
  "AutoEdit"
);

var midiInput = deviceDriver.mPorts.makeMidiInput("AutoEdit In");
var midiOutput = deviceDriver.mPorts.makeMidiOutput("AutoEdit Out");

deviceDriver
  .makeDetectionUnit()
  .detectPortPair(midiInput, midiOutput)
  .expectInputNameContains("AutoEdit")
  .expectOutputNameContains("AutoEdit");

var surface = deviceDriver.mSurface;
var page = deviceDriver.mMapping.makePage("AutoEdit");

function bindCC(cc, category, command, label) {
  var btn = surface.makeButton(cc - 20, 0, 1, 1);
  btn.mSurfaceValue.mMidiBinding
    .setInputPort(midiInput)
    .bindToControlChange(0, cc);
  page.makeCommandBinding(btn.mSurfaceValue, category, command);
}

bindCC(20, "File", "Export Audio Mixdown", "export");
bindCC(21, "File", "Import Audio File", "import");
