"""Isolated, local-only Grounding DINO inference; no app dependencies imported."""
import json
from pathlib import Path
import sys
import time


def load_detector(model_dir, device='auto'):
    import torch
    from transformers import AutoConfig, AutoProcessor, AutoModelForZeroShotObjectDetection
    if device not in ('auto','cpu','cuda'):
        raise ValueError('Object device must be auto, cpu or cuda')
    selected = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    config = AutoConfig.from_pretrained(model_dir,local_files_only=True)
    config.disable_custom_kernels = True
    processor = AutoProcessor.from_pretrained(model_dir,local_files_only=True)
    model,loading = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir,config=config,
        local_files_only=True,output_loading_info=True)
    if loading['missing_keys']:
        raise RuntimeError('Object weights do not match the runtime: '+str(loading['missing_keys']))
    model = model.to(selected).eval()
    return processor,model,selected


def detect(processor,model,device,image,settings):
    import torch
    inputs=processor(images=image,text=settings['prompt'],return_tensors='pt').to(device)
    with torch.inference_mode():
        outputs=model(**inputs)
    result=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
        threshold=settings['threshold'],text_threshold=.25,target_sizes=[image.size[::-1]])[0]
    labels=result.get('text_labels',result['labels'])
    return [{'box':list(map(float,box)),'score':float(score),'label':str(label)}
            for box,score,label in zip(result['boxes'].cpu().tolist(),result['scores'].cpu().tolist(),labels)]


def detect_batch(processor, model, device, images, settings):
    import torch
    from contextlib import nullcontext
    inputs = processor(images=images, text=[settings['prompt']] * len(images),
                       return_tensors='pt', padding=True).to(device)
    precision = settings.get('compute_type', 'float32')
    if precision not in ('float32', 'float16'):
        raise ValueError('Object compute_type must be float32 or float16')
    context = torch.autocast('cuda', dtype=torch.float16) if precision == 'float16' and device.startswith('cuda') else nullcontext()
    with torch.inference_mode(), context:
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
        threshold=settings['threshold'], text_threshold=.25, target_sizes=[im.size[::-1] for im in images])
    return [[{'box':list(map(float, box)), 'score':float(score), 'label':str(label)}
        for box, score, label in zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(),
                                     result.get('text_labels', result.get('labels', [])))] for result in results]


def batch_size(settings, device):
    requested = settings.get('batch_size', 'auto')
    if requested != 'auto':
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError('Object batch_size must be auto or a positive integer')
        return requested
    if not device.startswith('cuda'):
        return 1
    import torch
    free, _ = torch.cuda.mem_get_info(device)
    return max(1, min(4, (free - 1024*1024*1024) // (768*1024*1024)))


def requested_frames(request, frame_cache):
    import av
    from PIL import Image
    samples = request['samples']
    def key(sample):
        return (request['path'], sample['time_sec'], sample['width'], sample['height'])
    missing = [s for s in samples if s.get('roi') or key(s) not in frame_cache]
    container = None
    try:
        if missing:
            container = av.open(request['path'])
            stream = container.streams.video[0]
            origin = float(stream.start_time*stream.time_base) if stream.start_time is not None else 0.
            stream.thread_type = 'AUTO'
            stream.codec_context.thread_count = 4
            container.seek(int((missing[0]['time_sec']+origin)/float(stream.time_base)), stream=stream, backward=True)
            decoded = iter(container.decode(stream))
        for expected in samples:
            k = key(expected)
            if not expected.get('roi') and k in frame_cache:
                image = frame_cache[k]
                frame_cache.move_to_end(k)
                yield expected, Image.fromarray(image), image, None
                continue
            while True:
                frame = next(decoded, None)
                if frame is None:
                    raise RuntimeError('Incomplete object scan; no cache written')
                if frame.time is None:
                    raise RuntimeError('Object frame has no timestamp')
                timestamp = float(frame.time)-origin
                if timestamp >= expected['time_sec']-1e-5:
                    break
            if abs(timestamp-expected['time_sec']) > 1e-4:
                raise RuntimeError('Could not recover the requested object-detection frame')
            image = frame.reformat(width=expected['width'],height=expected['height'],format='rgb24').to_ndarray()
            transform = None
            if expected.get('roi'):
                native = Image.fromarray(frame.to_ndarray(format='rgb24'))
                sx,sy = native.width/expected['width'],native.height/expected['height']
                x0,y0,x1,y1 = expected['roi']
                crop = (round(x0*sx),round(y0*sy),round(x1*sx),round(y1*sy))
                inference_image = native.crop(crop)
                transform = (sx, sy, crop[0], crop[1])
            else:
                frame_cache[k] = image
                while len(frame_cache) > 256:
                    frame_cache.popitem(last=False)
                inference_image = Image.fromarray(image)
            yield expected, inference_image, image, transform
    finally:
        if container is not None:
            container.close()


def scan(request, models, frame_cache):
    import cv2
    import torch
    settings = request['settings']
    index = settings.get('device_index', 0)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError('Object device_index must be a nonnegative integer')
    if settings['device'] != 'cpu' and torch.cuda.is_available() and index >= getattr(torch.cuda, 'device_count', lambda: 1)():
        raise ValueError('Object device_index is outside the available CUDA devices')
    model_key = (request['model_dir'], settings['device'], index)
    if model_key not in models:
        try:
            processor, model, device = load_detector(request['model_dir'], settings['device'])
            if device == 'cuda' and index:
                device = 'cuda:' + str(index)
                model = model.to(device)
            models[model_key] = processor, model, device
            if device == 'cpu' and settings['device'] == 'auto':
                model._autoedit_fallback_reason = 'CUDA runtime unavailable'
        except (RuntimeError, OSError) as exc:
            if settings['device'] != 'auto' or not any(t in str(exc).lower() for t in ('cuda', 'cudnn', 'out of memory')):
                raise
            print('OBJECT: CUDA unavailable; using CPU: ' + str(exc), flush=True)
            models[model_key] = load_detector(request['model_dir'], 'cpu')
            models[model_key][1]._autoedit_fallback_reason = str(exc)
    processor, model, device = models[model_key]
    if device == 'cpu' and getattr(model, '_autoedit_offloaded', False):
        try:
            device = 'cuda:' + str(index)
            model = model.to(device)
            model._autoedit_offloaded = False
            models[model_key] = processor, model, device
        except RuntimeError as exc:
            if settings['device'] != 'auto' or not any(t in str(exc).lower() for t in ('cuda', 'cudnn', 'out of memory')):
                raise
            device = 'cpu'
            model = model.to('cpu')
            model._autoedit_offloaded = False
            models[model_key] = processor, model, device
            print('OBJECT: CUDA restore unavailable; continuing on CPU: ' + str(exc), flush=True)
    size = batch_size(settings, device)
    print(f'OBJECT: Whole-character scan on {device}; batch={size}; precision={settings.get("compute_type", "float32")}', flush=True)
    started = reported = time.perf_counter()
    rows = []
    pending = []
    def flush():
        nonlocal size, device, model, reported
        position = 0
        while position < len(pending):
            group = pending[position:position+size]
            query = group[0][0].get('query') or settings['prompt']
            try:
                detections = detect_batch(processor, model, device, [entry[1] for entry in group], {**settings, 'prompt':query})
            except torch.cuda.OutOfMemoryError as exc:
                if size > 1:
                    size = max(1, size//2)
                    torch.cuda.empty_cache()
                    print(f'OBJECT: VRAM pressure; reducing batch to {size}', flush=True)
                    continue
                if settings['device'] != 'auto':
                    raise
                model = model.to('cpu')
                model._autoedit_offloaded = False
                model._autoedit_fallback_reason = str(exc)
                device = 'cpu'
                models[model_key] = processor, model, device
                torch.cuda.empty_cache()
                print('OBJECT: VRAM pressure; retrying affected batch on CPU', flush=True)
                continue
            except RuntimeError as exc:
                compatible_error = any(token in str(exc).lower() for token in (
                    'no kernel image', 'invalid device function', 'cuda driver',
                    'nvidia driver', 'cudnn_status', 'cublas_status'))
                if not device.startswith('cuda') or settings['device'] != 'auto' or not compatible_error:
                    raise
                model = model.to('cpu')
                model._autoedit_offloaded = False
                model._autoedit_fallback_reason = str(exc)
                device, size = 'cpu', 1
                models[model_key] = processor, model, device
                torch.cuda.empty_cache()
                print('OBJECT: CUDA incompatible; retrying affected batch on CPU: ' + str(exc), flush=True)
                continue
            if len(detections) != len(group):
                raise RuntimeError('Incomplete object batch')
            for (expected, _, image, transform), objects in zip(group, detections):
                if transform:
                    sx,sy,x,y = transform
                    for obj in objects:
                        x0,y0,x1,y1 = obj['box']
                        obj['box'] = [(x0+x)/sx,(y0+y)/sy,(x1+x)/sx,(y1+y)/sy]
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                for obj in objects:
                    x0,y0,x1,y1 = map(round,obj['box'])
                    crop = gray[max(0,y0):min(gray.shape[0],y1),max(0,x0):min(gray.shape[1],x1)]
                    obj['clarity'] = min(1.,float(cv2.Laplacian(crop,cv2.CV_64F).var())/150) if crop.size else 0.
                rows.append({'time_sec':expected['time_sec'], 'objects':objects})
            position += len(group)
            if time.perf_counter()-reported >= 3:
                elapsed = time.perf_counter()-started
                print(f'OBJECT: Objects {len(rows)}/{len(request["samples"])} | source {rows[-1]["time_sec"]:.1f}s | elapsed {elapsed:.0f}s', flush=True)
                reported = time.perf_counter()
        pending.clear()
    frames = requested_frames(request, frame_cache)
    try:
        for entry in frames:
            if pending and (entry[0].get('query') or settings['prompt']) != (pending[0][0].get('query') or settings['prompt']):
                flush()
            pending.append(entry)
            if len(pending) >= size:
                flush()
        flush()
    finally:
        frames.close()
    return {'runtime':{'device':device, 'batch_size':size,
                      'fallback_reason':getattr(model, '_autoedit_fallback_reason', None),
                      'compute_type':settings.get('compute_type','float32') if device.startswith('cuda') else 'float32'},
            'samples':rows, 'elapsed_sec':time.perf_counter()-started}


def main():
    from collections import OrderedDict
    import traceback
    models, frames = {}, OrderedDict()
    def execute(path):
        request = json.loads(Path(path).read_text(encoding='utf-8'))
        result = scan(request, models, frames)
        Path(request['result']).write_text(json.dumps(result), encoding='utf-8')
    if sys.argv[1] != '--serve':
        execute(sys.argv[1])
        return
    for line in sys.stdin:
        try:
            value = json.loads(line)
            if isinstance(value, dict) and value.get('action') == 'offload':
                import torch
                for key, (processor, model, device) in list(models.items()):
                    if device.startswith('cuda'):
                        model = model.to('cpu')
                        model._autoedit_offloaded = True
                        models[key] = processor, model, 'cpu'
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                execute(value)
            print('RESULT: {}', flush=True)
        except Exception as exc:
            traceback.print_exc()
            print('RESULT: ' + json.dumps({'error':str(exc)}), flush=True)


if __name__ == '__main__':
    main()
