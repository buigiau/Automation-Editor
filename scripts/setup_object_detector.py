"""Install an isolated Grounding DINO runtime and official safetensors weights."""
from pathlib import Path
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = root / '.venv-objects'
    python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.is_file():
        subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
    pip = [str(python), '-m', 'pip', '--cache-dir', str(root / 'output/pip-object-cache'),
           '--timeout', '180', '--retries', '5']
    index = 'https://download.pytorch.org/whl/' + ('cpu' if args.cpu else 'cu124')
    subprocess.run([*pip, 'install', 'torch==2.6.0', 'torchvision==0.21.0', '--index-url', index], check=True)
    subprocess.run([*pip, 'install', 'transformers==4.50.3', 'numpy>=2,<3',
                    'pillow>=10', 'av>=16', 'opencv-python-headless>=4.10'], check=True)
    subprocess.run([str(python), str(root / 'scripts/probe_character_objects.py'), '--download-only'], check=True)


if __name__ == '__main__':
    main()
