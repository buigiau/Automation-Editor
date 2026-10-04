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


def main():
    import av
    import cv2
    from PIL import Image
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    settings=request['settings']
    processor,model,device=load_detector(request['model_dir'],settings['device'])
    print(f'OBJECT: Whole-character scan on {device}; independent boxes at every sampled timestamp',flush=True)
    frames=request['samples']
    rows=[]
    position=0
    started=reported=time.perf_counter()
    with av.open(request['path']) as container:
        stream=container.streams.video[0]
        origin=float(stream.start_time*stream.time_base) if stream.start_time is not None else 0.
        stream.thread_type='AUTO'
        stream.codec_context.thread_count=4
        for frame in container.decode(stream):
            if position >= len(frames):
                break
            expected=frames[position]
            timestamp=float(frame.time)-origin
            if timestamp < expected['time_sec']-1e-5:
                continue
            if abs(timestamp-expected['time_sec']) > 1e-4:
                raise RuntimeError('Could not recover the requested object-detection frame')
            image=frame.reformat(width=expected['width'],height=expected['height'],format='rgb24').to_ndarray()
            if expected.get('roi'):
                native=Image.fromarray(frame.to_ndarray(format='rgb24'))
                sx,sy=native.width/expected['width'],native.height/expected['height']
                x0,y0,x1,y1=expected['roi']
                crop_box=(round(x0*sx),round(y0*sy),round(x1*sx),round(y1*sy))
                region_settings={**settings,'prompt':expected.get('query') or settings['prompt']}
                objects=detect(processor,model,device,native.crop(crop_box),region_settings)
                for obj in objects:
                    ox0,oy0,ox1,oy1=obj['box']
                    obj['box']=[(ox0+crop_box[0])/sx,(oy0+crop_box[1])/sy,
                                (ox1+crop_box[0])/sx,(oy1+crop_box[1])/sy]
            else:
                objects=detect(processor,model,device,Image.fromarray(image),settings)
            gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            for obj in objects:
                x0,y0,x1,y1=map(round,obj['box'])
                crop=gray[max(0,y0):min(gray.shape[0],y1),max(0,x0):min(gray.shape[1],x1)]
                obj['clarity']=min(1.,float(cv2.Laplacian(crop,cv2.CV_64F).var())/150) if crop.size else 0.
            rows.append({'time_sec':timestamp,'objects':objects})
            position+=1
            if time.perf_counter()-reported >= 3:
                elapsed=time.perf_counter()-started
                print(f'OBJECT: Objects {position}/{len(frames)} | source {timestamp:.1f}s | elapsed {elapsed:.0f}s | ETA {elapsed*(len(frames)/position-1):.0f}s',flush=True)
                reported=time.perf_counter()
    if position != len(frames):
        raise RuntimeError('Incomplete object scan; no cache written')
    result={'runtime':{'device':device},'samples':rows,'elapsed_sec':time.perf_counter()-started}
    Path(request['result']).write_text(json.dumps(result),encoding='utf-8')


if __name__ == '__main__':
    main()
