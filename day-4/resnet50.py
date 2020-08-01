import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

class BottleNeck(nn.Module):
    """Bottleneck Design for ResNet
    Improves performance with reduced number of parameters
    This implementation follows the modified version wherein stride is located at conv3 rather than conv1
    """

    def __init__(self, in_channels, out_channels, neck_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, neck_channels, kernel_size=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(neck_channels)
        self.conv2 = nn.Conv2d(neck_channels, neck_channels, kernel_size=(3,3), stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(neck_channels)
        self.conv3 = nn.Conv2d(neck_channels, out_channels, bias=False, kernel_size=(1,1))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample: #Downsampling came from the modified version. Otherwise, use stride=2
            shortcut = self.downsample(shortcut)
        
        out += shortcut #Before we can actually add them, notice that both needs to have the same dimension [n,c,h,w]
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super().__init__()    
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7,7), stride=2, padding=3, bias=False) #TODO: compute padding
        self.bn1 = nn.BatchNorm2d(64)
        
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        downsample1 = nn.Sequential(nn.Conv2d(64,256,kernel_size=1,stride=1, bias=False), nn.BatchNorm2d(256))
        self.layer1 = nn.Sequential(BottleNeck(64,256, neck_channels=64, stride=1, downsample=downsample1),
                                    BottleNeck(256,256,neck_channels=64),
                                    BottleNeck(256,256,neck_channels=64))

        downsample2 = nn.Sequential(nn.Conv2d(256,512,kernel_size=1,stride=2, bias=False), nn.BatchNorm2d(512))
        self.layer2 = nn.Sequential(BottleNeck(256, 512,neck_channels=128, stride=2, downsample=downsample2),
                                    BottleNeck(512, 512,neck_channels=128),
                                    BottleNeck(512, 512,neck_channels=128),
                                    BottleNeck(512, 512,neck_channels=128))

        downsample3 = nn.Sequential(nn.Conv2d(512,1024, kernel_size=1,stride=2, bias=False), nn.BatchNorm2d(1024))
        self.layer3 = nn.Sequential(BottleNeck(512, 1024,neck_channels=256, stride=2, downsample=downsample3),
                                    BottleNeck(1024, 1024,neck_channels=256),
                                    BottleNeck(1024, 1024,neck_channels=256),
                                    BottleNeck(1024, 1024,neck_channels=256),
                                    BottleNeck(1024, 1024,neck_channels=256),
                                    BottleNeck(1024, 1024,neck_channels=256))

        downsample4 = nn.Sequential(nn.Conv2d(1024,2048, kernel_size=1,stride=2, bias=False), nn.BatchNorm2d(2048))
        self.layer4 = nn.Sequential(BottleNeck(1024, 2048,neck_channels=512, stride=2, downsample=downsample4),
                                    BottleNeck(2048, 2048,neck_channels=512),
                                    BottleNeck(2048, 2048,neck_channels=512))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.pool1(out)
        out = self.layer1(out) #conv2_x in paper
        out = self.layer2(out) #conv3_x in paper
        out = self.layer3(out) #conv4_x in paper
        out = self.layer4(out) #conv5_x in paper
        out = torch.flatten(out, 1) # We need to flatten it into 1D vector before going thru linear layer
        out = self.fc(out)

        return out

def resnet50(pretrained=False, progress=True):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    init_weights=True
    if pretrained:
        init_weights=False

    model = ResNet(init_weights=init_weights)

    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model