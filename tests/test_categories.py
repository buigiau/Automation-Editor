from autoedit.audio.categories import category_from_stem
from autoedit.audio.analyzer import classify_filename


def test_lexicon_hits():
    assert category_from_stem("a")[0] == "A"
    assert category_from_stem("ohh")[0] == "OH"
    assert category_from_stem("heheha")[0] == "LAUGH"
    assert category_from_stem("yeah")[0] == "YEAH"
    assert category_from_stem("unknown")[0] == "UNCLEAR"


def test_filename_path():
    cat, conf = classify_filename(r"D:\Editor\Voice\heheha.WAV")
    assert cat == "LAUGH"
    assert conf > 0.7


def test_voice_folder_loads_every_wav():
    from pathlib import Path

    from autoedit.audio.analyzer import analyze_audio_dir, iter_audio_files

    root = Path(r"D:\Editor\Voice")
    if not root.exists():
        return
    files = iter_audio_files(root)
    assert len(files) >= 100
    items = analyze_audio_dir(root, max_files=0)
    assert len(items) == len(files)
