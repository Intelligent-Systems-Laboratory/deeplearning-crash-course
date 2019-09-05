import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,64,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,128,3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(128,256,3)
        self.conv6 = nn.Conv2d(256,256,3)
        self.conv7 = nn.Conv2d(256,256,3)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv8 = nn.Conv2d(256,512,3)
        self.conv9 = nn.Conv2d(512,512,3)
        self.conv10 = nn.Conv2d(512,512,3)
        self.pool4 = nn.MaxPool2d(2,2)
        self.conv11 = nn.Conv2d(512,512,3)        
        self.conv12 = nn.Conv2d(512,512,3)
        self.conv13 = nn.Conv2d(512,512,3)
        self.pool5 = nn.MaxPool2d(2,2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes))
        
        if init_weights:
            self._initialize_weights()
            
        
    def forward(self, x):
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv7(self.conv5(self.conv6(x))))
        x = self.pool4(self.conv10(self.conv9(self.conv8(x)))) 
        x = self.pool5(self.conv13(self.conv12(self.conv11(x))))
        x = self.avg_pool(x)
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