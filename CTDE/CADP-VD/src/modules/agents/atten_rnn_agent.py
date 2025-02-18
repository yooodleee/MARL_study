
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init


from modules.layers.self_atten import SelfAttention



class ATTRNNAgent(nn.Module):

    def __init__(self, input_shape, args):
        super(ATTRNNAgent, self).__init__()

        self.args = args
        self.use_q_v = False
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.att = SelfAttention(input_shape, args.att_heads, args.att_embed_dim)
        self.fc2 = nn.Linear(args.att_heads * args.att_embed_dim, args.rnn_hidden_dim)
        self.fc_inter = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim), nn.ReLU(inplace=True)
        )
        self.fc_last = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(args.rnn_hidden_dim, args.n_actions),
        )

    
    