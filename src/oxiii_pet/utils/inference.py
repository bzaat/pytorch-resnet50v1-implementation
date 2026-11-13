"""Shared single-image / folder inference used by model/test.py and transfer_learning/test.py."""
import glob
import os

import torch
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

from oxiii_pet.utils.run_utils import get_class_names

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def resolve_checkpoint(checkpoint, kind):
    """checkpoint may be a .pth file, a run dir, or None (newest runs/*_<kind>_*/checkpoints/best_model.pth)."""
    if checkpoint is None:
        candidates = glob.glob(os.path.join("runs", f"*{kind}*", "checkpoints", "best_model.pth"))
        if not candidates:
            raise SystemExit(f"No checkpoint given and none found under runs/*{kind}*/checkpoints/")
        return max(candidates, key=os.path.getmtime)
    if os.path.isdir(checkpoint):
        return os.path.join(checkpoint, "checkpoints", "best_model.pth")
    return checkpoint


def load_weights(model, checkpoint_path, device):
    """Load new-style dict checkpoints ({"model": ...}) and legacy bare state_dicts. Returns the raw dict."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt if isinstance(ckpt, dict) else {}


def list_images(path):
    if os.path.isdir(path):
        return sorted(p for p in glob.glob(os.path.join(path, "*")) if p.lower().endswith(IMAGE_EXTS))
    return [path]


def predict(model, ckpt, args, device):
    class_names = ckpt.get("class_names") or get_class_names(args.num_class, args.data_dir)
    transform = Compose([
        Resize((args.images_size, args.images_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model.eval()
    k = min(3, len(class_names))
    for path in list_images(args.test_path):
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(image), dim=1)[0]
        top = torch.topk(probs, k)
        top_str = ", ".join(f"{class_names[i]} {p:.3f}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"{os.path.basename(path)}: {class_names[top.indices[0]]} ({top.values[0]:.3f}) | top-{k}: {top_str}")


def add_common_args(parser):
    parser.add_argument("--images_size", "-i", type=int, default=224, help="size of images")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help=".pth file or run dir; default: newest best_model.pth under runs/")
    parser.add_argument("--test_path", "-t", type=str, default="test_images", help="image file or directory")
    parser.add_argument("--num_class", "-n", type=int, default=37)
    parser.add_argument("--data_dir", "-d", type=str, default=os.getenv("DATA_DIR", "data"),
                        help="dataset root, used for class names if the checkpoint has none")
