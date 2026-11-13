from argparse import ArgumentParser

import torch

from oxiii_pet.model.models import Resnet50
from oxiii_pet.utils.inference import add_common_args, load_weights, predict, resolve_checkpoint


def get_args():
    parser = ArgumentParser(description="CNN Test")
    add_common_args(parser)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint, "scratch")
    print(f"Checkpoint: {checkpoint_path}")
    model = Resnet50(args.num_class).to(device)
    ckpt = load_weights(model, checkpoint_path, device)
    predict(model, ckpt, args, device)

if __name__ == '__main__':
    main()
