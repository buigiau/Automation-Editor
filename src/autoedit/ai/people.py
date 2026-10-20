"""CPU-only person counts and aligned face embeddings for real-person cuts.

YOLOX weights: OpenCV Zoo / Megvii, Apache-2.0; see models/people/LICENSE.
Unknown/occluded faces retain unresolved, cut-local identities. Counts and identity are sampled, never certified
for every frame or treated as evidence of speech.
Reference appearance searches set enable_identity=False and use detection/masks only.
"""
from pathlib import Path
import hashlib
import time

from .cache import digest
from .frames import cut_frames

VERSION = 'cpu-people-3'
MODEL_SHA256 = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'


class FastPeople:
    def __init__(self, *, enable_identity=True):
        import cv2
        import numpy as np
        import onnxruntime as ort
        from autoedit.presets import ROOT
        from autoedit.video.faces import FaceLandmarker, MODEL_PATH
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker as Task, FaceLandmarkerOptions, RunningMode
        import mediapipe as mp
        path = ROOT/'models/people/yolox.onnx'
        if not path.is_file():
            raise FileNotFoundError('Run scripts/setup_fast_people.py to install CPU person detection.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('CPU person detector checksum mismatch; reinstall it.')
        self.enable_identity = enable_identity
        if enable_identity and not (ROOT/'models/characters/arcface.onnx').is_file():
            raise FileNotFoundError('Aligned human face embedding model is missing.')
        if not MODEL_PATH.is_file():
            raise FileNotFoundError('Face landmark model is missing; run scripts/setup_face_model.py.')
        cv2.setNumThreads(2)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        self.net = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.input_name = self.net.get_inputs()[0].name
        self.preview_net = cv2.dnn.readNet(str(path))
        self.preview_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.preview_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.mp, self.describe = mp, FaceLandmarker._describe
        self.faces = Task.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH), delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.IMAGE, num_faces=8,
            min_face_detection_confidence=.5, min_face_presence_confidence=.5))
        grids, strides = [], []
        for stride in (8, 16, 32):
            x, y = np.meshgrid(np.arange(640//stride), np.arange(640//stride))
            grid = np.stack((x, y), axis=-1).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((len(grid), 1), stride))
        self.grids, self.strides = np.concatenate(grids), np.concatenate(strides)
        grids, strides = [], []
        for stride in (8, 16, 32):
            x, y = np.meshgrid(np.arange(320//stride), np.arange(320//stride))
            grid = np.stack((x, y), axis=-1).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((len(grid), 1), stride))
        self.preview_grids, self.preview_strides = np.concatenate(grids), np.concatenate(strides)
        self.embedding = None
        self.anchors = []
        self.human_mode = False
        self.preview_cache = {}
        self.stats = {'previews': 0, 'verified_cuts': 0, 'rejected_cuts': 0}

    def close(self):
        self.faces.close()
        self.net = self.preview_net = self.embedding = None

    def inspect(self, rgb, minimum_face=32, preview=False):
        import cv2
        import numpy as np
        h, w = rgb.shape[:2]
        size = 320 if preview else 640
        scale = min(size/w, size/h)
        resized = cv2.resize(rgb, (round(w*scale), round(h*scale)))
        image = np.full((size, size, 3), 114, dtype=np.float32)
        image[:resized.shape[0], :resized.shape[1]] = resized
        blob = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        if preview:
            self.preview_net.setInput(blob)
            outputs = self.preview_net.forward()[0]
            grids, strides = self.preview_grids, self.preview_strides
        else:
            outputs = self.net.run(None, {self.input_name: blob})[0][0]
            grids, strides = self.grids, self.strides
        centers = (outputs[:, :2]+grids)*strides
        sizes = np.exp(outputs[:, 2:4])*strides
        boxes = np.concatenate((centers-sizes/2, sizes), axis=1)/scale
        scores = outputs[:, 4:5]*outputs[:, 5:]
        actors = []
        # People, cats, dogs and the remaining COCO animals count as neighbours.
        for cls in (0, *range(14, 24)):
            indices = np.flatnonzero(scores[:, cls] >= .3)
            if not len(indices):
                continue
            kept = cv2.dnn.NMSBoxes(boxes[indices].tolist(), scores[indices, cls].tolist(), .3, .45)
            for index in np.asarray(kept).reshape(-1):
                box = boxes[indices[index]].tolist()
                if box[2]*box[3] < 4:
                    continue
                actors.append({'class': cls, 'box': box})
        result = self.faces.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        faces = [self.describe(points, w, h) for points in result.face_landmarks]
        people = [a for a in actors if a['class'] == 0]
        count = max(len(actors), len(faces))
        clear = [f for f in faces if min(f['face'][2:]) >= minimum_face and f['frontal_score'] >= .35
            and any(self.inside(f['face'], p['box']) for p in people)]
        return {'count': count, 'people': len(people), 'body_boxes': [p['box'] for p in people],
                'faces': faces, 'clear': clear}

    @staticmethod
    def inside(face, box):
        x, y, w, h = face
        bx, by, bw, bh = box
        return bx <= x+w/2 <= bx+bw and by <= y+h/2 <= by+bh

    def preview(self, path):
        import cv2
        if str(path) not in self.preview_cache:
            image = cv2.imread(str(path))
            if image is None:
                return {'priority': 2, 'subject_count': None, 'face_area': 0}
            data = self.inspect(image[:, :, ::-1].copy(), minimum_face=20, preview=True)
            if data['people']:
                self.human_mode = True
            area = max((f['face'][2]*f['face'][3]/(image.shape[0]*image.shape[1]) for f in data['clear']), default=0)
            group = data['count'] > 1
            preview = {'priority': 0 if data['count'] == 1 and data['people'] else 1 if group else 2,
                'subject_count': data['count'], 'human_count': data['people'], 'face_area': area}
            appearance = getattr(self, 'reference_appearance', None)
            if appearance is not None:
                preview['reference_appearance_score'] = appearance.preview_score(image[:, :, ::-1].copy(), data)
                score = preview['reference_appearance_score']
                if score is not None:
                    preview['priority'] = 0 if score >= .25 and data['people'] else 3
            self.preview_cache[str(path)] = preview
            self.stats['previews'] += 1
        return self.preview_cache[str(path)]

    def classify(self, window, source, cache, *, allow_group=False):
        if not getattr(self, 'enable_identity', True):
            raise RuntimeError('Facial identity matching is disabled for reference appearance searches.')
        import numpy as np
        from autoedit.video.embeddings import EmbeddingModel
        a, b = window['start_sec']+.08, window['end_sec']-.08
        b -= window.get('transition_tail_fallback_sec', 0)
        observations, vectors, mouths = [], [], []
        for _, rgb in cut_frames(source['path'], [a+(b-a)*f for f in (.15, .5, .85)], 640):
            info = self.inspect(rgb)
            if not allow_group and info['count'] > 1:
                return {'deferred_group': True}
            if not info.get('people', 1) or (not allow_group and info['count'] != 1):
                self.last_rejection = f"subjects={info['count']}, faces={len(info.get('faces', []))}, clear={len(info['clear'])}"
                self.stats['rejected_cuts'] += 1
                return None
            if not info.get('faces') and any(box[1] <= 2 for box in info.get('body_boxes', [])):
                self.last_rejection = 'head may be cropped outside the top of the frame'
                self.stats['rejected_cuts'] += 1
                return None
            observations.append(info)
            if not info['clear']:
                continue
            # A group fallback must track the same largest clear face.
            face = max(info['clear'], key=lambda f: f['face'][2]*f['face'][3])
            if self.embedding is None:
                self.embedding = EmbeddingModel('live_action', {'device': 'cpu'})
            vector = self.embedding.extract(rgb[:, :, ::-1].copy(), face, rgb.shape[1], rgb.shape[0])
            if vector is None:
                self.last_rejection = 'face alignment unavailable'
                continue
            vectors.append(vector)
            mouths.append(2 if face['lip_aperture'] >= .15 else 1)
        if any(1-float(np.dot(x, y)) > .3 for i, x in enumerate(vectors) for y in vectors[i+1:]):
            self.last_rejection = 'face embeddings disagree across cut samples'
            self.stats['rejected_cuts'] += 1
            return None
        confirmed = len(vectors) >= 2
        if not confirmed and allow_group:
            self.last_rejection = 'group fallback requires a matching clear face in at least two samples'
            return None
        if confirmed:
            anchor = np.mean(vectors, axis=0)
            anchor /= np.linalg.norm(anchor)
            match = next((i for i, existing in enumerate(self.anchors) if 1-float(np.dot(anchor, existing)) <= .25), None)
            if match is None:
                match = len(self.anchors)
                self.anchors.append(anchor)
            cid = f'person-{match+1:03d}'
        else:
            # One measured body in every sample is useful solo footage, but it
            # does not identify a person. Never merge these cuts into a cast ID.
            cid = 'person-unresolved-'+digest([source.get('id'), source['path'], a, b])[:12]
        self.stats['verified_cuts'] += 1
        return {'id': cid, 'reference_ids': [],
            'subject_count': max(o['count'] for o in observations),
            'mouth': min(mouths) if confirmed else 0, 'identity_consistent': confirmed,
            'identity_sample_frames': len(vectors) if confirmed else 0,
            'method': VERSION, 'sample_frames': 3}
