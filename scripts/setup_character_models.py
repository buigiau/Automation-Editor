"""Install CCIP or a user-supplied licensed ArcFace ONNX recognition model."""
from pathlib import Path
import argparse
import shutil
from urllib.request import urlretrieve
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccip", action="store_true", help="Download the standard CCIP feature model")
    parser.add_argument("--arcface", action="store_true", help="Download InsightFace buffalo_sc public research weights")
    parser.add_argument("--arcface-file", type=Path, help="Copy an ArcFace RGB 112x112 recognition ONNX model")
    args = parser.parse_args()
    if not args.ccip and not args.arcface_file and not args.arcface:
        parser.error("Specify --ccip, --arcface or --arcface-file PATH")
    if args.arcface and args.arcface_file:
        parser.error("Choose --arcface or --arcface-file, not both")
    root = Path(__file__).resolve().parents[1] / "models" / "characters"
    root.mkdir(parents=True, exist_ok=True)
    if args.ccip:
        from huggingface_hub import hf_hub_download
        source = hf_hub_download("deepghs/ccip_onnx", "ccip-caformer-24-randaug-pruned/model_feat.onnx",
                                 cache_dir=str(root / ".downloads"))
        temporary = root / "ccip.onnx.download"
        shutil.copyfile(source, temporary)
        temporary.replace(root / "ccip.onnx")
        print(root / "ccip.onnx")
    if args.arcface:
        print("InsightFace public pretrained weights: non-commercial research use only.", flush=True)
        archive = root / "buffalo_sc.zip"
        if not archive.is_file():
            temporary = archive.with_suffix(".download")
            urlretrieve("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip", temporary)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            names = [n for n in bundle.namelist() if Path(n).name == "w600k_mbf.onnx"]
            if len(names) != 1:
                raise ValueError("ArcFace recognition model not found in the official buffalo_sc archive")
            temporary = root / "arcface.onnx.download"
            with bundle.open(names[0]) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            temporary.replace(root / "arcface.onnx")
        print(root / "arcface.onnx")
    if args.arcface_file:
        if not args.arcface_file.is_file():
            parser.error("ArcFace model file does not exist")
        dest = root / "arcface.onnx"
        if args.arcface_file.resolve() != dest.resolve():
            temporary = dest.with_suffix(".download")
            shutil.copyfile(args.arcface_file, temporary)
            temporary.replace(dest)
        print(dest)


if __name__ == "__main__":
    main()
