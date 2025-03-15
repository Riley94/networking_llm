import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Embedding layers for: 
'bytes_in','bytes_out', 'num_pkts_in', 'num_pkts_out', 'proto'
"""

class Network_Dection_Model(nn.Module):
    def __init__(self, output_dim, 
                 embed_dim_bin, num_bin,
                 embed_dim_bout, num_bout,
                 embed_dim_pin, num_pin,
                 embed_dim_pout, num_pout,
                 embed_dim_proto, num_proto):
        super().__init__()
        self.continuous = nn.Linear(4, 20)
        self.bin_embed = nn.Embedding(num_embeddings=num_bin, embedding_dim=embed_dim_bin)
        self.bout_embed = nn.Embedding(num_embeddings=num_bout, embedding_dim=embed_dim_bout)
        self.pin_embed = nn.Embedding(num_embeddings=num_pin, embedding_dim=embed_dim_pin)
        self.pout_embed = nn.Embedding(num_embeddings=num_pout, embedding_dim=embed_dim_pout)
        self.proto_embed = nn.Embedding(num_embeddings=num_proto, embedding_dim=embed_dim_proto)

        self.input_dim = 20 + embed_dim_bin + embed_dim_bout + embed_dim_pin + embed_dim_pout + embed_dim_proto

        self.hidden_dim = int(math.ceil((self.input_dim + 1) *.67))

        self.linear_1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        #Extract features
        """
        'avg_ipt', 'entropy', 'total_entropy', 'duration', [0: 4]
        'bytes_in', 4
       'bytes_out', 5
       'num_pkts_in', 6
       'num_pkts_out', 7
       'proto', 8
       """
        continuous_features = x[:, 0:4]
        bin = x[:, 4].to(torch.long)
        bout = x[:, 5].to(torch.long)
        pin = x[:, 6].to(torch.long)
        pout = x[:, 7].to(torch.long)
        proto = x[:, 8].to(torch.long)

        #Run through embedding layers
        continuous = F.relu(self.continuous(continuous_features))
        bin = self.bin_embed(bin)
        bout = self.bout_embed(bout)
        pin = self.pin_embed(pin)
        pout  = self.pout_embed(pout)
        proto = self.proto_embed(proto)

        x = torch.cat([continuous, bin, bout, pin, pout, proto], dim=-1)
        
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.out(x)

        return x



