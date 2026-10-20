"""Offline review sheet; corrections are revalidated by the worker before export."""
import json
from html import escape
from pathlib import Path


def write_review(snapshot, out, plan=None):
    out = Path(out)
    sources = {s['id']: s for s in snapshot['sources']}
    samples = {(s.get('audio') or {}).get('voice_id'): s['cubase']['sample_path']
               for s in (plan or {}).get('cubase_slots', [])}
    rows = []
    for row in snapshot['accepted']:
        source = sources[row['source_id']]
        rows.append({**row, 'source_uri': Path(source['path']).as_uri(), 'source_name': Path(source['path']).name,
                     'source_duration_sec': source['duration_sec']})
    data = {'revision': snapshot['revision'], 'rows': rows,
        'voices': [{'id': v['voice_id'], 'name': v['name'],
                    'uri': Path(samples.get(v['voice_id'], v['path'])).as_uri(),
                    'start': 0 if v['voice_id'] in samples else v.get('active_start_sec', 0)} for v in snapshot['voices']]}
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('&', '\\u0026')
    errors = ''.join('<li>' + escape(str(e['slot_id']) + ': ' + e['error']) + '</li>' for e in snapshot['errors'])
    completion = snapshot['result'].get('completion', {})
    if completion.get('fallback_count'):
        errors += ('<li>Bản nháp đủ cảnh: '+str(completion['validated_count'])+' cảnh qua kiểm tra AI, '
                   +str(completion['fallback_count'])+' cảnh dự phòng. Cần xem lại nhân vật, chuyển cảnh và Voice.</li>')
    html = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AutoEdit — Review AI</title><style>
body{font:16px system-ui;margin:32px;background:#f6f7f9;color:#17212c}h1{font-size:26px}
article{padding:20px;background:white;border:1px solid #ddd;border-radius:12px;margin:18px 0}
video{width:460px;max-width:100%;display:block;margin:12px 0}audio{display:block;margin:12px 0}
label{display:inline-block;margin:8px 16px 8px 0}input,select,button{font:inherit;padding:8px}
.flag{color:#994900;font-weight:600}small{color:#556}button{cursor:pointer}p{max-width:1050px;line-height:1.6}
</style><h1>Xem / nghe đối chiếu</h1>
<p>Video phát tiếng gốc. Nghe Voice riêng hoặc bấm “Phát với Voice” để xem video với WAV thay thế.
Chỉnh mốc bắt đầu trong cùng shot hoặc đổi Voice, tải tệp chỉnh sửa rồi chọn “Áp dụng chỉnh sửa” trong AutoEdit.
Các cảnh lặp của template dùng chung một đoạn và Voice. MIDI/FX Cubase cần kiểm tra bằng playback thực tế.</p>
<p>Nếu trình duyệt không hỗ trợ codec của nguồn, mở video bằng trình phát trên máy để xem đúng mốc ghi bên dưới.</p>
<ul class="flag">''' + errors + '''</ul><div id="rows"></div><button id="save">Tải chỉnh sửa JSON</button>
<script id="data" type="application/json">''' + payload + '''</script><script>
const data=JSON.parse(document.getElementById('data').textContent), controls=[];
function el(tag,text,parent){const x=document.createElement(tag);if(text)x.textContent=text;if(parent)parent.append(x);return x;}
for(const row of data.rows){
 const card=el('article',null,document.getElementById('rows'));
 const identity=row.completion_fallback ? 'Cảnh dự phòng · chưa xác nhận nhân vật'
   : row.reference_match_scope==='costume_appearance' ? 'Khớp trang phục theo ảnh tham chiếu'
   : row.identity_sample_consistent===false
   ? (row.subject_count===1?'Một người · chưa xác định danh tính':'Nhóm nhân vật · cần duyệt danh tính')
   : row.character_id.replace(/^person-([0-9]+)$/,'Nhân vật $1');
 el('h2',row.slot_id+' · '+identity,card);
 el('p',row.reason,card);
 const match=row.audio_match||{};
 const matchLabel=match.method==='source-audio-whisper-exact-word'?'Khớp từ nguồn':
   match.method==='source-audio-pronunciation'?'Khớp phát âm nguồn':
   match.method==='source-audio-event'?'Sự kiện âm nguồn · cần duyệt':'Âm dự phòng · cần duyệt';
 el('p',matchLabel+(match.source_word?' · '+match.source_word:match.source_action?' · '+match.source_action:'')+
   (typeof match.source_word_start_sec==='number'?' · nguồn '+match.source_word_start_sec.toFixed(3)+'s':'')+
   (match.manual_override?' · đã sửa thủ công':''),card).className='flag';
 if(row.completion_fallback)el('p','Bản nháp dự phòng: '+row.fallback_relaxations.join(', '),card).className='flag';
 if(row.timeline_visible_prefix)el('p','Timeline chỉ phát '+(row.identity_end_sec-row.in_sec).toFixed(2)+' giây đầu; phần đuôi nest được điền đủ nhưng không xuất hiện trong bản dựng.',card).className='flag';
 el('p',(row.final_scene_transition_fallback?'Cục cuối timeline: cho phép chuyển cảnh · ':'')+(row.transition_tail_fallback_sec?'Fallback chuyển cảnh trong '+row.transition_tail_fallback_sec+' giây cuối · ':'')+(row.group_scene_fallback?'Fallback cảnh nhiều nhân vật · ':'')+(row.identity_sample_consistent===false?'Danh tính chưa xác định · ':'')+(row.same_shot_fallback?'Cùng shot, đoạn khác · ':'')+(row.reference_fallback?'Nhân vật thay thế · ':'')+(row.audio_uncertain?'Âm chưa chắc chắn':'Cần nghe xác nhận'),card).className='flag';
 el('small',row.source_name+' · Shot '+row.shot_start_sec.toFixed(3)+'–'+row.shot_end_sec.toFixed(3)+'s',card);
 const cutDuration=row.out_sec-row.in_sec, previewDuration=row.timeline_visible_prefix?row.identity_end_sec-row.in_sec:cutDuration;
 const video=el('video',null,card);video.controls=true;video.preload='metadata';video.src=row.source_uri+'#t='+row.in_sec+','+(row.in_sec+previewDuration);
 const startLabel=el('label','Mốc bắt đầu (giây) ',card), start=el('input',null,startLabel);
 start.type='number';start.step='.001';start.min=row.completion_fallback?0:row.shot_start_sec;start.max=row.completion_fallback?row.source_duration_sec-cutDuration:row.timeline_visible_prefix?Math.min(row.shot_end_sec-previewDuration,row.source_duration_sec-cutDuration):row.shot_end_sec-cutDuration;start.value=row.in_sec;
 const voiceLabel=el('label','Voice ',card), select=el('select',null,voiceLabel);
 for(const voice of data.voices){const option=el('option',voice.name,select);option.value=voice.id;}
 select.value=row.voice_id;
 const audio=el('audio',null,card);audio.controls=true;audio.preload='none';
 function updateVoice(){audio.src=data.voices.find(v=>v.id===select.value).uri;}
 updateVoice();select.onchange=updateVoice;
 const play=el('button','Phát với Voice',card);play.onclick=()=>{video.muted=true;video.currentTime=Number(start.value);audio.currentTime=data.voices.find(v=>v.id===select.value).start;video.play();audio.play();};
 const original=el('button','Phát tiếng gốc',card);original.onclick=()=>{audio.pause();video.muted=false;video.currentTime=Number(start.value);video.play();};
 video.addEventListener('timeupdate',()=>{if(video.currentTime>=Number(start.value)+previewDuration){video.pause();audio.pause();}});
 start.onchange=()=>{video.currentTime=Number(start.value);};
 controls.push({row,start,select});
}
document.getElementById('save').onclick=()=>{
 if(controls.some(c=>!c.start.checkValidity())){alert('Mốc cắt phải nằm trong shot.');return;}
 const edits=controls.map(c=>({slot_id:c.row.slot_id,in_sec:Number(c.start.value),voice_id:c.select.value}));
 const url=URL.createObjectURL(new Blob([JSON.stringify({revision:data.revision,edits},null,2)],{type:'application/json'}));
 const a=el('a');a.href=url;a.download='autoedit-corrections.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};</script></html>'''
    path = out / 'ai_review.html'
    path.write_text(html, encoding='utf-8')
    return path
