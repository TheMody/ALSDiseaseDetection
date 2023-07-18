import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=2, num_input = 75584):
        super().__init__()
        hidden_dim = 256
        self.dense1 = nn.Linear(num_input, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.gelu(self.dense1(x))
        x = F.gelu(self.dense2(x))
        return F.softmax(self.dense3(x), dim=1)