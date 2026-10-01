"""Local listening sheet: source soundtrack beside each selected voice."""
from html import escape
from pathlib import Path


def write_audio_review(plan, output_dir, speech):
    import av
    import soundfile as sf
    from autoedit.audio.source_speech import _decode_window

    out = Path(output_dir)
    clips = out / "review_source_audio"
    clips.mkdir(parents=True, exist_ok=True)
    source = plan["project"]["source_video"]
    originals = {}
    with av.open(source) as container:
        if container.streams.audio:
            stream = container.streams.audio[0]
            video = container.streams.video[0] if container.streams.video else stream
            origin = float(video.start_time * video.time_base) if video.start_time is not None else 0
            for slot in plan["cubase_slots"]:
                uid = slot["premiere"]["nested_sequence_uid"]
                if uid in originals:
                    continue
                cut = slot["video"]
                path = clips / f"scene_{len(originals)+1:02d}.wav"
                pcm = _decode_window(container, stream, origin, cut["in_sec"], cut["out_sec"])
                sf.write(path, pcm, 16000, subtype="PCM_16")
                originals[uid] = path.relative_to(out).as_posix()
    rows = []
    for slot in plan["cubase_slots"]:
        prem, audio, video, match, cubase = (slot[k] for k in ("premiere", "audio", "video", "audio_match", "cubase"))
        uid = prem["nested_sequence_uid"]
        words = " ".join(w["word"] for w in speech.get("words", [])
                         if w["start"] < video["out_sec"] and w["end"] > video["in_sec"])
        matched = match.get("source_word") or match.get("source_action")
        method = f'Từ nguồn: {matched}' if matched else 'Ước lượng theo hình miệng / âm dự phòng'
        status = "Cần kiểm tra" if match.get("needs_review") else "Đã đối chiếu từ; nghe xác nhận"
        src_player = (f'<audio controls preload="none" src="{escape(originals[uid], quote=True)}"></audio>'
                      if uid in originals else 'Video không có kênh âm thanh')
        sample = Path(cubase["sample_path"]).relative_to(out.resolve()).as_posix()
        rows.append(f'''<tr><td><strong>{escape(cubase['track_name'])}</strong><br>Cảnh {escape(str(prem['nested_sequence']))}
        <br><small>{prem['timeline_start_sec']:.2f}–{prem['timeline_end_sec']:.2f}s trên timeline</small></td>
        <td>{src_player}<br><small>Nguồn {video['in_sec']:.3f}–{video['out_sec']:.3f}s</small>
        <p>{escape(words or 'Không nhận diện được từ')}</p></td>
        <td><audio controls preload="none" src="{escape(sample, quote=True)}"></audio>
        <p>{escape(Path(audio['path']).name)}</p></td>
        <td>{escape(method)}<p class="status">{status}</p>
        <small>Độ lệch từ nguồn: {float(audio.get('placement_offset_sec') or 0):.3f}s</small></td></tr>''')
    html = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Kiểm tra âm thanh AutoEdit</title><style>
    body{font:15px system-ui,sans-serif;margin:32px;color:#222;background:#fafafa}h1{font-size:26px}
    table{border-collapse:collapse;width:100%;background:white}td,th{padding:16px;border:1px solid #ddd;text-align:left;vertical-align:top}
    audio{width:260px;max-width:100%}small{color:#555}.status{font-weight:600;color:#945300}p{line-height:1.5}
    </style><h1>Nghe đối chiếu âm thanh</h1>
    <p>Nghe âm thanh gốc của đoạn video, rồi nghe voice được chọn. Transcript là nhận diện tự động.
    Các cảnh chỉ đoán theo hình miệng đều cần kiểm tra.</p>
    <p>Sample Cubase bắt đầu ngay ở tiếng nói. Độ lệch từ nguồn chỉ áp dụng trong preview;
    bản phối Cubase còn phụ thuộc nốt MIDI, cao độ và hiệu ứng. Bảng này chưa xác nhận bản phối cuối.</p>
    <table><thead><tr><th>Track / cảnh</th><th>Âm thanh video gốc</th><th>Voice đã chọn</th><th>Đối chiếu</th></tr></thead><tbody>'''
    html += "\n".join(rows) + "</tbody></table></html>"
    destination = out / "audio_review.html"
    destination.write_text(html, encoding="utf-8")
    plan["audio_review"] = str(destination.resolve())
    return destination
