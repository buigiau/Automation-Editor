"""Install the small CPU person detector from OpenCV Zoo, verifying its hash."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'models/people'
SHA256 = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
URL = 'https://huggingface.co/opencv/opencv_zoo/resolve/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx'

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST/'yolox.onnx'
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
        print('CPU person detector already installed.')
        return
    temporary = path.with_suffix('.download')
    try:
        with urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
            while chunk := response.read(1024*1024):
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Person detector checksum mismatch.')
        temporary.replace(path)
        with urlopen('https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/object_detection_yolox/LICENSE', timeout=30) as response:
            (DEST/'LICENSE').write_bytes(response.read())
    finally:
        temporary.unlink(missing_ok=True)
    print('Installed CPU person detector (35.9 MB).')

if __name__ == '__main__':
    main()
