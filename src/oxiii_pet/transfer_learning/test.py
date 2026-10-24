from argparse import ArgumentParser

import torch
import torch.nn as nn
import cv2
import numpy as np

from oxiii_pet.model.models import Resnet50
from oxiii_pet.transfer_learning.transfer_learning import MyModel


def get_args():
    parser = ArgumentParser(description="CNN Test")
    parser.add_argument("--images_size", "-i", type=int, default=224, help="size of images")
    parser.add_argument("--checkpoint", "-c", type=str, default="trained_model/best_model.pth")
    parser.add_argument("--test_path", "-t", type=str, default="test_images")
    parser.add_argument("--num_class", "-n", type=int, default=37)
    parser.add_argument("--version", "-v", type=int, default=2, help="Version of ImageNet1K Dataset: 1 = V1, 2 = V2")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # classes = ['cat','dog']
    classes = [
        # cat
        "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
        "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
        "Siamese", "Sphynx",

        # dog
        "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle",
        "boxer", "chihuahua", "english_cocker_spaniel", "english_setter",
        "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin",
        "keeshond", "leonberger", "miniature_pinscher", "newfoundland",
        "pomeranian", "pug", "saint_bernard", "samoyed", "scottish_terrier",
        "shiba_inu", "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"
    ]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # model = Resnet50(args.num_class).to(device)
    model = MyModel(args.num_class, args.version).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # model.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found")
        exit(0)
    model.eval()
    # preprocess images
    image = cv2.imread(args.test_path)
    original_image = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.images_size, args.images_size))
    image = np.transpose(image, (2, 0, 1)) / 255.0
    image = image[None, :, :, :]  # add forth dimension:-> 1 x 3 x img_size x img_size
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        print(output)
        probs = softmax(output)
        print(probs)
    max_ind = torch.argmax(probs)
    predicted_class = classes[max_ind]
    print("result: {} with confident score: {}".format(predicted_class, probs[0, max_ind]))
    filename = f"result_{predicted_class}_{probs[0, max_ind] * 100:.0f}.png"
    cv2.imwrite(filename, original_image)
    cv2.imshow(filename, original_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()