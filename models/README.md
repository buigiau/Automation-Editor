# Landmark and speech models

Run `.venv\Scripts\python scripts/setup_face_model.py` to download Google's Face Landmarker v1 and Pose Landmarker Lite (`float16`) bundles here. Both run locally. The video analysis cache includes both model modification times.

Whisper defaults to multilingual `small`, CPU `int8`, with English transcription. An existing Hugging Face model cache is reused offline; otherwise the first transcription downloads weights into `models/whisper/`. `video.speech_model` accepts a model size or local model directory. Set `video.speech_language: null` for language detection. Audio stays local.

Official model: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Documentation and model information: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

Pose model: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

Pose documentation: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python

Whisper implementation: https://github.com/SYSTRAN/faster-whisper

## Character recognition

Install ONNX Runtime and Hugging Face Hub through `requirements.txt`. Run
`scripts/setup_character_models.py --ccip` for `characters/ccip.onnx`, using
DeepGHS `ccip-caformer-24-randaug-pruned/model_feat.onnx`.
Run `--arcface-file PATH` to install your licensed ArcFace recognition model as
`characters/arcface.onnx`. `--arcface` instead installs InsightFace's public
buffalo_sc `w600k_mbf.onnx`; those pretrained weights are for non-commercial research.

The default contracts are NCHW RGB, 112x112 ArcFace with `(pixel - 127.5) / 127.5`,
and 384x384 CCIP with RGB means `(0.48145466, 0.4578275, 0.40821073)` and standard
deviations `(0.26862954, 0.26130258, 0.27577711)`. ArcFace crops are aligned from
MediaPipe eye/nose/mouth landmarks. CCIP uses an expanded head crop and skips
crops containing another detected face. Models with different contracts require
a matching adapter; arbitrary ONNX image classifiers are not interchangeable.

Only model outputs are cached; crop images are not persisted. Model SHA-256
changes invalidate embeddings. `video.characters.arcface_model` and `ccip_model`
override the local paths. The face-exposure heuristic does not establish a
character's narrative role. Similar-looking characters, mirrored faces, changes
in costume and missed face detections can affect clustering. DINOv2 is not used
in this implementation; animation uses CCIP.

CCIP model and preprocessing: https://huggingface.co/deepghs/ccip_onnx

InsightFace weights and terms: https://github.com/deepinsight/insightface/tree/master/python-package
