
import torch.nn as nn
import torch.nn.functional as F



class RNNAgent(nn.Module):

    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
    

    def init_hidden(self):
        """Make hidden states on same device as model"""
        return self.fc1.weight.new(-1, self.modules.rnn_hidden_dim).zero_()
    

    