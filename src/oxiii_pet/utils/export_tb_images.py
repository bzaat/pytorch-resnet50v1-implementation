"""Export an image logged to TensorBoard (default: the confusion matrix of the last epoch) as a PNG.

Usage: uv run python -m oxiii_pet.utils.export_tb_images runs/<run> -o results/<name>_confusion_matrix.png
"""
import io
import os
from argparse import ArgumentParser

from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import IMAGES, EventAccumulator


def main():
    parser = ArgumentParser(description="Export a TensorBoard image to PNG")
    parser.add_argument("run_dir", help="run folder containing tensorboard/")
    parser.add_argument("--output", "-o", required=True, help="output PNG path")
    parser.add_argument("--tag", default="confusion_matrix")
    parser.add_argument("--step", type=int, default=None, help="epoch index (0-based); default: last")
    parser.add_argument("--max_size", type=int, default=1100, help="longest side in pixels after resizing")
    args = parser.parse_args()

    acc = EventAccumulator(os.path.join(args.run_dir, "tensorboard"), size_guidance={IMAGES: 0})
    acc.Reload()
    events = acc.Images(args.tag)
    event = events[-1] if args.step is None else next(e for e in events if e.step == args.step)
    image = Image.open(io.BytesIO(event.encoded_image_string)).convert("RGB")
    image.thumbnail((args.max_size, args.max_size), Image.LANCZOS)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image.save(args.output, optimize=True)
    print(f"Saved {args.output} (tag {args.tag}, step {event.step} = epoch {event.step + 1}, {image.size[0]}x{image.size[1]})")


if __name__ == "__main__":
    main()
