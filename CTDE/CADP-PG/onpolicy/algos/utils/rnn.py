
import torch
import torch.nn as nn


"""
RNN modules.
"""



class RNNLayer(nn.Module):

    def __init__(
            self,
            inputs_dim,
            outputs_dim,
            recurrent_N,
            use_orthogonal):
        
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(
            inputs_dim,
            outputs_dim,
            num_layers=self._recurrent_N
        )

        for name, param in self.rnn.named_parameters():
            
            if 'bias' in name:
                nn.init.constant_(param, 0)
            
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        self.norm = nn.LayerNorm(outputs_dim)
    

    