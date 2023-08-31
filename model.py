import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class MLPModel(nn.Module):
    def __init__(self, num_classes=2, num_input = 75584):
        super().__init__()
        hidden_dim = 4196
        self.dense1 = nn.Linear(num_input, hidden_dim)
     #   self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(8)])
        self.dense3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.gelu(self.dense1(x))
        for i in range(4):
            x2 = x
            x = F.gelu(self.linears[i*2](x))
            x = F.gelu(self.linears[i*2+1](x2)) +x2
    #    x = F.gelu(self.dense2(x))
        return F.softmax(self.dense3(x), dim=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
     #   self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

class EncoderModel(nn.Module):
    #input should be (batchsize, num_pcas, dim_pcas)
    def __init__(self, num_classes=2, input_dim = 8):
        super().__init__()
        hidden_dim = 16
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, batch_first=True, activation='gelu')
        self.PositionalEncoding = PositionalEncoding(hidden_dim, max_len=18279)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
       # 
       # self.dense1 = nn.Linear(num_input, hidden_dim)
       # self.dense2 = nn.Linear(hidden_dim, hidden_dim)
      #  self.dense3 = nn.Linear(64, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
      #  print(x.shape)
      #  print(x.type())
        x = self.dense1(x)
        x = self.PositionalEncoding(x)
       # x = self.encoder_layer(x)[:,0,:]
       # x,_ = torch.max(x, dim=1)#/x.shape[1]
        x = torch.sum(x, dim=1)/x.shape[1]
        x = F.gelu(self.dense2(x))
      #  x = F.gelu(self.dense2(x))
     #   x = F.gelu(self.dense3(x))
        return F.softmax(self.dense3(x), dim=1)

class MultichannelLinear(nn.Module): #maybe this is missing the bias term
    def __init__(self, channels, in_features, out_features,  down_project = 1):
        super(MultichannelLinear, self).__init__()
        self.down_project = down_project
        self.weight_pw = nn.Parameter(torch.empty(int(math.ceil(channels/down_project)), out_features, in_features*down_project))
        self.weight_bias = nn.Parameter(torch.empty(int(math.ceil(channels/down_project)), out_features))
        nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features*down_project), b=1/math.sqrt(in_features*down_project))
        nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features*down_project), b=1/math.sqrt(in_features*down_project))

    def __call__(self, x):
        if not self.down_project ==1:   
            #reshape x to (batchsize, num_pcas/down_project, dim_pcas*down_project)
            if x.shape[1] % self.down_project != 0:
                x = F.pad(x, (0,0,0,self.down_project - x.shape[1] % self.down_project))
            x = x.reshape(x.shape[0], int(x.shape[1]/self.down_project), x.shape[2]*self.down_project)
            
        x = torch.matmul(self.weight_pw.unsqueeze(0),x.unsqueeze(-1)).squeeze(-1) + self.weight_bias.unsqueeze(0)
     #   print(x.shape)
        return x

class IndMLPModel(nn.Module):
    #input should be (batchsize, num_pcas, dim_pcas)
    def __init__(self, num_classes=2, input_dim = 8, num_pcas = 18279):
        super().__init__()
        self.num_pcas = num_pcas
       # self.linears = nn.ModuleList([nn.Linear(input_dim, 1) for i in range(self.num_pcas)])
        self.linears1 = MultichannelLinear(self.num_pcas, input_dim, 32,32)
     #   self.linears2 = MultichannelLinear(int(math.ceil(self.num_pcas/8)), 8, 16,8)
        hidden_dim = 512
        self.dense1 = nn.Linear(18304, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense8 = nn.Linear(hidden_dim, hidden_dim)
        self.dense4 = nn.Linear(hidden_dim, hidden_dim)
        self.dense5 = nn.Linear(hidden_dim, hidden_dim)
        self.dense6 = nn.Linear(hidden_dim, hidden_dim)
        self.dense7 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
    #    print(x.shape)
    #    print(F.gelu(self.linears[0](x[:,0,:])).shape)
       # x = torch.cat([F.gelu(self.linears[i](x[:,i,:])) for i in range(self.num_pcas)], dim=1)
     #   x = F.gelu(self.linears1(x))
        x = F.gelu(self.linears1(x).flatten(1))
     #   print(x.shape)
        x2 = F.gelu(self.dense1(x))
        x = F.gelu(self.dense2(x2))  
        x2 = F.gelu(self.dense8(x)) +x2
        x = F.gelu(self.dense4(x2)) 
        x2 = F.gelu(self.dense5(x)) +x2
        x = F.gelu(self.dense6(x2))
        x = F.gelu(self.dense7(x)) +x2
     #   x = F.gelu(self.dense2(x))
        return F.softmax(self.dense3(x), dim=1)
