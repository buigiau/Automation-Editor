"""Validate runtime-selected Premiere, Cubase, source video, and audio folder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoedit.audio.analyzer import iter_audio_files
from autoedit.config import source_paths
from autoedit.premiere.prproj import inspect_prproj
from autoedit.cubase.inspect import inspect_sampler_tracks, parse_sampler_tracks, select_sampler_tracks

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv", ".mpg", ".mpeg", ".mxf"}
TEMPLATE_SEQUENCE = "PJ 5 - demo"


class InputError(ValueError):
    """One or more selected inputs are invalid."""


def _exist_file(path: str | Path, label: str, suffixes: set[str]) -> list[str]:
    errors: list[str] = []
    if not path:
        errors.append(f"{label} is not selected")
        return errors
    p = Path(path)
    if not p.exists():
        errors.append(f"{label} does not exist: {p}")
        return errors
    if not p.is_file():
        errors.append(f"{label} is not a file: {p}")
        return errors
    if suffixes and p.suffix.lower() not in suffixes:
        errors.append(f"{label} must be one of {sorted(suffixes)} (got {p.suffix})")
    return errors


def validate_inputs(
    *,
    premiere_project: str | Path = "",
    cubase_project: str | Path = "",
    source_video: str | Path | list[str | Path] = "",
    audio_directory: str | Path = "",
    template_sequence: str = TEMPLATE_SEQUENCE,
    video_track_index: int = 1,
    sampler_tracks=None,
) -> list[str]:
    errors: list[str] = []
    errors += _exist_file(premiere_project, "Premiere project", {".prproj"})
    errors += _exist_file(cubase_project, "Cubase project", {".cpr"})
    videos = source_paths(source_video)
    if not videos:
        errors.append("Source video is not selected")
    for path in videos:
        errors += _exist_file(path, "Source video", VIDEO_SUFFIXES)

    if not audio_directory:
        errors.append("Audio folder is not selected")
    else:
        audio = Path(audio_directory)
        if not audio.exists():
            errors.append(f"Audio folder does not exist: {audio}")
        elif not audio.is_dir():
            errors.append(f"Audio folder is not a directory: {audio}")
        else:
            wavs = iter_audio_files(audio)
            if not wavs:
                errors.append(f"Audio folder has no WAV files: {audio}")

    info = None
    if premiere_project and Path(premiere_project).is_file() and Path(premiere_project).suffix.lower() == ".prproj":
        try:
            info = inspect_prproj(
                premiere_project,
                source_sequence="",
                template_sequence=template_sequence,
                video_track_index=video_track_index,
            )
        except Exception as exc:
            errors.append(f"Premiere project could not be read: {exc}")
        else:
            names = {s.get("name") for s in (info.get("sequences") or [])}
            if not info.get("template_sequence"):
                errors.append(
                    f"Premiere project has no sequence named {template_sequence!r}. "
                    f"Found: {sorted(n for n in names if n)}"
                )
            elif not (info.get("slots") or info.get("instance_slots")):
                errors.append(
                    f"Sequence {template_sequence!r} has no video slots on the template track"
                )
    try:
        indices = parse_sampler_tracks(sampler_tracks)
        if (indices is not None and info is not None and cubase_project
                and Path(cubase_project).is_file() and Path(cubase_project).suffix.lower() == ".cpr"):
            select_sampler_tracks(inspect_sampler_tracks(cubase_project), indices, len(info["scene_slots"]))
    except ValueError as exc:
        errors.append(f"Cubase sampler selection: {exc}")
    return errors


def require_inputs(**kwargs: Any) -> None:
    errors = validate_inputs(**kwargs)
    if errors:
        raise InputError("Invalid inputs:\n- " + "\n- ".join(errors))
