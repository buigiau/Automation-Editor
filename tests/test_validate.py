import gzip
from pathlib import Path

from autoedit.config import apply_inputs, load_config
from autoedit.validate import validate_inputs


def _tiny_prproj(path: Path, sequence_name: str) -> None:
    xml = (
        '<?xml version="1.0"?>\n'
        "<PremiereData>\n"
        f'  <Sequence ObjectUID="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa">'
        f"<Name>{sequence_name}</Name></Sequence>\n"
        "</PremiereData>\n"
    ).encode("utf-8")
    path.write_bytes(gzip.compress(xml))


def test_validate_reports_missing_inputs():
    errors = validate_inputs()
    assert any("Premiere" in e for e in errors)
    assert any("Cubase" in e for e in errors)
    assert any("Source video" in e for e in errors)
    assert any("Audio folder" in e for e in errors)


def test_validate_accepts_selected_files(tmp_path):
    prproj = tmp_path / "job.prproj"
    _tiny_prproj(prproj, "PJ 5 - demo")
    cpr = tmp_path / "job.cpr"
    cpr.write_bytes(b"cpr")
    video = tmp_path / "source.mp4"
    video.write_bytes(b"mp4")
    audio = tmp_path / "voice"
    audio.mkdir()
    (audio / "a.wav").write_bytes(b"RIFF")
    errors = validate_inputs(
        premiere_project=prproj,
        cubase_project=cpr,
        source_video=video,
        audio_directory=audio,
    )
    # Tiny prproj has the sequence name but no clip slots.
    assert any("no video slots" in e for e in errors)
    assert not any("does not exist" in e for e in errors)
    assert not any("no sequence named" in e for e in errors)


def test_apply_inputs_sets_runtime_paths(tmp_path):
    cfg = load_config(None)
    apply_inputs(
        cfg,
        premiere=tmp_path / "a.prproj",
        cubase=tmp_path / "b.cpr",
        video=tmp_path / "c.mp4",
        audio_dir=tmp_path / "voice",
    )
    assert cfg["premiere"]["project"].endswith("a.prproj")
    assert cfg["cubase"]["project"].endswith("b.cpr")
    assert cfg["premiere"]["source_media"].endswith("c.mp4")
    assert cfg["audio"]["directory"].endswith("voice")
    assert cfg["premiere"]["source_sequence"] == ""
