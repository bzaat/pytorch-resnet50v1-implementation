import torch.nn as nn


class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.stem = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,),
            nn.MaxPool2d(3,2)
        )
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=37)
        )
        self.conv2_x1, self.shortcut2_x1 = self.make_block(in_channels= 64,out_channels=64,last_out=256)
        self.conv2_x2, self.shortcut2_x2 = self.make_block(in_channels= 256,out_channels=64,last_out=256)
        self.conv2_x3, self.shortcut2_x3 = self.make_block(in_channels= 256,out_channels=64,last_out=256)

        self.conv3_x1, self.shortcut3_x1 = self.make_block(in_channels= 256,out_channels=128,last_out=512, first_block=True)
        self.conv3_x2, self.shortcut3_x2 = self.make_block(in_channels= 512,out_channels=128,last_out=512)
        self.conv3_x3, self.shortcut3_x3 = self.make_block(in_channels= 512,out_channels=128,last_out=512)
        self.conv3_x4,self.shortcut3_x4  = self.make_block(in_channels= 512,out_channels=128,last_out=512)

        self.conv4_x1, self.shortcut4_x1 = self.make_block(in_channels= 512,out_channels=256,last_out=1024, first_block=True)
        self.conv4_x2, self.shortcut4_x2 = self.make_block(in_channels= 1024,out_channels=256,last_out=1024)
        self.conv4_x3, self.shortcut4_x3 = self.make_block(in_channels= 1024,out_channels=256,last_out=1024)
        self.conv4_x4, self.shortcut4_x4 = self.make_block(in_channels= 1024,out_channels=256,last_out=1024)
        self.conv4_x5, self.shortcut4_x5 = self.make_block(in_channels= 1024,out_channels=256,last_out=1024)
        self.conv4_x6, self.shortcut4_x6 = self.make_block(in_channels= 1024,out_channels=256,last_out=1024)

        self.conv5_x1, self.shortcut5_x1 = self.make_block(in_channels= 1024,out_channels=512,last_out=2048, first_block=True)
        self.conv5_x2, self.shortcut5_x2 = self.make_block(in_channels= 2048,out_channels=512,last_out=2048)
        self.conv5_x3, self.shortcut5_x3 = self.make_block(in_channels= 2048,out_channels=512,last_out=2048)

    def make_block(self,in_channels,out_channels,last_out, first_block = False):
        stride = 1
        padding = "same"
        if first_block:
            stride = 2
            padding = 1
        residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1,stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3,stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, last_out, 1,stride=1, padding="same"),
            nn.BatchNorm2d(last_out)
        )
        if in_channels != last_out or first_block:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, last_out, 1,stride= stride , bias=False),
                nn.BatchNorm2d(last_out),
            )
        else:
            shortcut = nn.Identity()

        return residual, shortcut

    def forward(self,x):
        x = self.stem(x)
# Stage 1
        identity = self.shortcut2_x1(x)           #shape_x N = 64
        x = self.conv2_x1(x)
        x+= identity
        x = self.relu(x)

        identity = self.shortcut2_x2(x)
        x = self.conv2_x2(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut2_x3(x)
        x = self.conv2_x3(x)     #shape_x N =256
        x+= identity             #skip connection
        x = self.relu(x)

        identity = self.shortcut3_x1(x)
        x = self.conv3_x1(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut3_x2(x)
        x = self.conv3_x2(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut3_x3(x)
        x = self.conv3_x3(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut3_x4(x)
        x = self.conv3_x4(x)    #shape_x N = 512
        x += identity
        x = self.relu(x)

        identity = self.shortcut4_x1(x)
        x = self.conv4_x1(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut4_x2(x)
        x = self.conv4_x2(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut4_x3(x)
        x = self.conv4_x3(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut4_x4(x)
        x = self.conv4_x4(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut4_x5(x)
        x = self.conv4_x5(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut4_x6(x)
        x = self.conv4_x6(x)    #shape_x N = 1024
        x += identity
        x = self.relu(x)

        identity = self.shortcut5_x1(x)
        x = self.conv5_x1(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut5_x2(x)
        x = self.conv5_x2(x)
        x += identity
        x = self.relu(x)

        identity = self.shortcut5_x3(x)
        x = self.conv5_x3(x)    #shape_x N= 2048
        x += identity
        x= self.fc(x)
        return x
def main():
    pass
if __name__ == '__main__':
    main()