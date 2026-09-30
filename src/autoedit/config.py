"""YAML / dict config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from autoedit.video.characters import DEFAULTS as CHARACTER_DEFAULTS

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULTS: dict[str, Any] = {
    "job": {"name": "job", "output_dir": "./output"},
    "premiere": {
        "project": "",
        "source_sequence": "",
        "template_sequence": "PJ 5 - demo",
        "source_media": "",
        "slot_mode": "nested_sequences",
        "video_track_index": 1,
        "import_mixdown": True,
    },
    "audio": {"directory": r"D:\Editor\Voice", "glob": "*.wav", "files": [], "max_files": 0},
    "video": {
        "sample_fps": 6,
        "max_seconds": 0,
        "source_kind": "live_action",
        "source_gap_sec": 5.0,
        "min_segment_sec": 0.25,
        "merge_gap_sec": 0.20,
        "backend": "auto",
        "speech_model": "small",
        "speech_language": "en",
        "characters": dict(CHARACTER_DEFAULTS),
    },
    "cubase": {
        "project": "",
        "track_name_pattern": "Sampler Track {index:02d}",
        "sampler_track_count": 16,
        "sample_rate": 48000,
        "channels": 2,
    },
    "categories": ["A", "E", "O", "HAHA", "CRY", "CLOSED", "NEUTRAL"],
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = _deep_merge({}, DEFAULTS)
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files")
    loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return _deep_merge(cfg, loaded)


def apply_inputs(
    cfg: dict[str, Any],
    *,
    premiere: str | Path | None = None,
    cubase: str | Path | None = None,
    video: str | Path | None = None,
    audio_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Overlay runtime-selected paths onto a config dict."""
    if premiere:
        cfg.setdefault("premiere", {})["project"] = str(Path(premiere))
    if cubase:
        cfg.setdefault("cubase", {})["project"] = str(Path(cubase))
    if video:
        cfg.setdefault("premiere", {})["source_media"] = str(Path(video))
    if audio_dir:
        cfg.setdefault("audio", {})["directory"] = str(Path(audio_dir))
    if output_dir:
        cfg.setdefault("job", {})["output_dir"] = str(Path(output_dir))
    return cfg
