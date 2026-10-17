import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, bottleneck_channels, stride=1):
        super().__init__()

        out_channels = bottleneck_channels * self.expansion

        # Main branch: 1x1 -> 3x3 -> 1x1
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels, bottleneck_channels,
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                bottleneck_channels, bottleneck_channels,
                kernel_size=3, stride=stride,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                bottleneck_channels, out_channels,
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut branch
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.residual(x)
        out += identity
        out = self.relu(out)
        return out

def make_layer(in_channels, bottleneck_channels, num_blocks, stride=1):
    layers = [
        Bottleneck(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            stride=stride
        )
    ]

    out_channels = bottleneck_channels * Bottleneck.expansion

    for _ in range(1, num_blocks):
        layers.append(
            Bottleneck(
                in_channels=out_channels,
                bottleneck_channels=bottleneck_channels
            )
        )

    return nn.Sequential(*layers)



class Resnet50(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=7,
                stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        )

        self.conv2_x = make_layer(
            64, 64, num_blocks=3
        )
        self.conv3_x = make_layer(
            256, 128, num_blocks=4, stride=2
        )
        self.conv4_x = make_layer(
            512, 256, num_blocks=6, stride=2
        )
        self.conv5_x = make_layer(
            1024, 512, num_blocks=3, stride=2
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.fc(x)
        return x

def main():
    pass
if __name__ == '__main__':
    main()