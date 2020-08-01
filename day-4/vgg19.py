import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

class VGG(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096), #Due to AdaptiveAvgPool ,that's where we get 7*7, the 512 comes from prev layer
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes))
        
        if init_weights:
            self._initialize_weights()
            
        
    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out
   
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
                
def vgg19(pretrained=True):
    init_weights=True
    if pretrained:
        init_weights=False
    
    model = VGG(init_weights=init_weights)
    
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',progress=True)
        model.load_state_dict(state_dict)
        
    return model