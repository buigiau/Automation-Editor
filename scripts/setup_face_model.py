"""Download Google's face and lite pose landmark models into this checkout."""
from pathlib import Path
from urllib.request import urlretrieve

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
if __name__ == "__main__":
    for name, url in MODELS.items():
        dest = Path(__file__).resolve().parents[1] / "models" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file():
            temp = dest.with_suffix(".download")
            urlretrieve(url, temp)
            temp.replace(dest)
        print(dest)
