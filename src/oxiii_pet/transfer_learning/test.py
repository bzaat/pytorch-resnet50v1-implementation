from argparse import ArgumentParser

import torch

from oxiii_pet.transfer_learning.transfer_learning import MyModel
from oxiii_pet.utils.inference import add_common_args, load_weights, predict, resolve_checkpoint


def get_args():
    parser = ArgumentParser(description="CNN Test")
    add_common_args(parser)
    parser.add_argument("--version", "-v", type=int, default=2, help="Version of ImageNet1K Dataset: 1 = V1, 2 = V2")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint, "transfer")
    print(f"Checkpoint: {checkpoint_path}")
    model = MyModel(args.num_class, args.version).to(device)
    ckpt = load_weights(model, checkpoint_path, device)
    predict(model, ckpt, args, device)


if __name__ == '__main__':
    main()
