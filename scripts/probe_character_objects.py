"""Evaluate whole-character detection on the actual manual Zoomally cuts."""
from pathlib import Path
import argparse
import json
import time

MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
MODEL_REVISION = 'a2bb814'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--include-props',action='store_true')
    parser.add_argument('--props-only',action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model_dir = root / 'models/objects/grounding-dino-tiny'
    installed_revision = model_dir / '.autoedit-revision'
    if (not (model_dir / 'model.safetensors').is_file() or not installed_revision.is_file()
            or installed_revision.read_text().strip() != MODEL_REVISION):
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(model_dir),
                          allow_patterns=['*.json', '*.txt', 'model.safetensors'])
        installed_revision.write_text(MODEL_REVISION)
    if args.download_only:
        print('Object detection model installed:', model_dir, flush=True)
        return
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, AutoConfig, AutoModelForZeroShotObjectDetection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Whole-character detector:', device, torch.__version__, flush=True)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    config = AutoConfig.from_pretrained(str(model_dir),local_files_only=True)
    config.disable_custom_kernels = True
    model,loading = AutoModelForZeroShotObjectDetection.from_pretrained(str(model_dir),config=config,
        local_files_only=True,output_loading_info=True)
    if loading['missing_keys']:
        raise RuntimeError('Object model weights do not match the runtime: '+str(loading['missing_keys']))
    model = model.to(device)
    model.eval()
    comparison = json.loads((root/'output/zoomally-diagnosis/comparison.json').read_text(encoding='utf-8'))
    video = json.loads((root/'output/zoomally-fixed/video_analysis.json').read_text(encoding='utf-8'))
    cuts = comparison['manual_openings_at_failed_template_lengths']
    requests = sorted((c['start_sec']+fraction*(c['end_sec']-c['start_sec']),c['sequence'])
                      for c in cuts for fraction in (0,.5,.95))
    if args.include_props:
        requests=sorted(requests+[(t,'prop') for t in (29.5,49.5,76.5,86.5,92.9)])
    if args.props_only:
        requests=[(t,'prop-check') for t in (43.5,49.5,50.5,76.5,86.5,92.9)]
    cap = cv2.VideoCapture(video['path'])
    rows, results = [], []
    import sys
    sys.path.insert(0,str(root/'src'))
    from autoedit.video.objects import DEFAULT_PROMPT
    prompt = DEFAULT_PROMPT
    started = time.perf_counter()
    for timestamp, scene in requests:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Missing source frame')
        frame = cv2.resize(frame,(960,540))
        image = Image.fromarray(frame[:,:,::-1])
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(device)
        with torch.inference_mode():
            output = model(**inputs)
        detected = processor.post_process_grounded_object_detection(output, inputs.input_ids,
            threshold=.3, text_threshold=.25, target_sizes=[(540,960)])[0]
        objects = [{'box':list(map(float,box)), 'score':float(score), 'label':label}
                   for box,score,label in zip(detected['boxes'].cpu().tolist(),
                       detected['scores'].cpu().tolist(),detected.get('text_labels',detected['labels']))]
        results.append({'scene':scene,'time_sec':timestamp,'objects':objects})
        for obj in objects:
            x0,y0,x1,y1=map(round,obj['box'])
            cv2.rectangle(frame,(x0,y0),(x1,y1),(0,200,255),2)
            cv2.putText(frame,f"{obj['label']} {obj['score']:.2f}",(x0,max(16,y0)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,255),1)
        cv2.putText(frame,f'Scene {scene} | {timestamp:.2f}s',(5,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        rows.append(cv2.resize(frame,(320,180)))
        print(f'{len(rows)}/{len(requests)} scene {scene}: {[(o["label"],round(o["score"],2)) for o in objects]}',flush=True)
    out = root/'output/zoomally-objects'
    out.mkdir(parents=True,exist_ok=True)
    (out/'probe.json').write_text(json.dumps({'device':device,'model':MODEL_ID,'prompt':prompt,
        'elapsed_sec':time.perf_counter()-started,'frames':results},indent=2),encoding='utf-8')
    if args.props_only:
        cv2.imwrite(str(out/'prop-check.jpg'),np.vstack([np.hstack(rows[i:i+3]) for i in range(0,len(rows),3)]))
        return
    for page in range(2):
        cv2.imwrite(str(out/f'probe-{page+1}.jpg'),np.vstack([np.hstack(rows[i:i+3]) for i in range(page*18,(page+1)*18,3)]))
    if args.include_props:
        leftovers=rows[36:]
        while len(leftovers)%3: leftovers.append(np.zeros_like(rows[0]))
        cv2.imwrite(str(out/'probe-props.jpg'),np.vstack([np.hstack(leftovers[i:i+3]) for i in range(0,len(leftovers),3)]))


if __name__ == '__main__':
    main()
