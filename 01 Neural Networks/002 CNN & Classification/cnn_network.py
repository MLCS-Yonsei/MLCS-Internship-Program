#==========================================#
# Title:  cnn network
# Author: Jaewoong Han
# Date:   2025-06-27
#==========================================#
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, batch_size):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5), # [100,1,28,28] -> [100,16,24,24]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5), # [100,16,24,24] -> [100,32,20,20]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # [100,32,20,20] -> [100,32,10,10]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), # [100,32,10,10] -> [100,64,6,6]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # [100,64,6,6] -> [100,64,3,3]       
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,100), # [100,64*3*3] -> [100,100]
            nn.ReLU(),
            nn.Linear(100,10) # [100,100] -> [100,10]
        )       
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(self.batch_size,-1) # -> (100, remainder)
        out = self.fc_layer(out)
        return out