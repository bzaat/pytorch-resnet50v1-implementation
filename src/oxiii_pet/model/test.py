from argparse import ArgumentParser

import torch
import torch.nn as nn
import cv2
import numpy as np

from oxiii_pet.model.models import Resnet50


def get_args():
    parser = ArgumentParser(description="CNN Test")
    parser.add_argument("--images_size","-i", type=int, default=224, help="size of images")
    parser.add_argument("--checkpoint","-c", type=str, default="trained_model/best_model.pth")
    parser.add_argument("--test_path","-t", type=str, default="test_images")
    parser.add_argument("--num_class","-n", type=int, default=37)
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    classes = ['cat','dog']
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Resnet50(args.num_class).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        print("No checkpoint found")
        exit(0)
    model.eval()
    #preprocess images
    image = cv2.imread(args.test_path)
    original_image = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.images_size,args.images_size))
    image = np.transpose(image, (2,0,1))/255.0
    image = image[None,:,:,:] #add forth dimension:-> 1 x 3 x img_size x img_size
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        print(output)
        probs = softmax(output)
        print(probs)
    max_ind = torch.argmax(probs)
    predicted_class = classes[max_ind]
    print("result: {} with confident score: {}".format(predicted_class,probs[0,max_ind]))
    cv2.imshow('class:{} {}'.format(predicted_class,probs[0,max_ind]),original_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()