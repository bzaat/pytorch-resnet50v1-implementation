from argparse import ArgumentParser

from torch.utils.data import DataLoader
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from oxiii_pet.dataset.dataset_oxiiit_pet import OxfordData


def get_args():
    parser = ArgumentParser("RESNET50 TRANSFER LEARNING")
    parser.add_argument("--num_classes", "-n",type=int, default=2,help="Number of class")
    parser.add_argument("--version", "-v",type=int, default=1,help="Version of ImageNet1K Dataset: 1 = V1, 2 = V2")
    args = parser.parse_args()
    return args

class MyModel(nn.Module):
    def __init__(self,num_classes,version):
        super().__init__()
        if version ==1:
            dataset = "IMAGENET1K_V1"
        else:
            dataset = "IMAGENET1K_V2"

        self.model = resnet50(weights=dataset)
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)
        del self.model.fc
        self.drop_out = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1  = nn.Linear(2048, 1024)
        self.fc2  = nn.Linear(1024, num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self



def main():
    args = get_args()
    model = MyModel(args.num_classes,args.version)
    for name, param in model.named_parameters():
        if not "fc1." in name and not "fc2."  in name and not "layer4."  in name:
            param.requires_grad = False
        print(name,param.requires_grad)
    # model = MyModel(37, 2)
    # model.train()
    # print(model.model.bn1.training)  # phải là False
    # print(model.fc1.training)  # phải là True

if __name__ == '__main__':
    main()