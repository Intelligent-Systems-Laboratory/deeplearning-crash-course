import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
class VGG(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64,128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.avg_pool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,num_classes))
        
        if init_weights:
            self._initialize_weights()
            
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.reshape(x.size(0),-1)
        x = self.classifier(x)
        return x
   
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
                
                
def vgg16(pretrained=True):
    init_weights=True
    if pretrained:
        init_weights=False
    
    model = VGG(init_weights=init_weights)
    
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth',progress=True)
        model.load_state_dict(state_dict)
        
    return model