import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# 定义MLP模型
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//4)
        self.fc3 = nn.Linear(input_dim//4, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.5)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


