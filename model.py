import torch
import torch.nn as nn

# 从你上传的代码中提取的核心网络结构
class ParaNet(nn.Module):
    def __init__(self, num_emotions=2): # 默认为2类：专注/不专注
        super(ParaNet, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        
        # 第一条卷积支路
        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(p=0.4),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(4, 4), nn.Dropout(p=0.4),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(4, 4), nn.Dropout(p=0.4),
        )
        
        # 第二条卷积支路 (并行结构，这是该模型的亮点)
        self.conv2Dblock2 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(p=0.4),
            nn.Conv2d(32, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(4, 4), nn.Dropout(p=0.4),
            nn.Conv2d(64, 128, 7, 1, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(4, 4), nn.Dropout(p=0.4),
        )

        # 全连接层
        self.fc1_linear = nn.Linear(9408, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        conv2d_embedding1 = self.conv2Dblock1(x)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)

        conv2d_embedding2 = self.conv2Dblock2(x)
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)

        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2], dim=1)
        output_logits = self.fc1_linear(complete_embedding)
        return output_logits