import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        activate_fn=None
        if args.activate_fn == "ELU":
            activate_fn = nn.ELU
        elif args.activate_fn == "relu":
            activate_fn = nn.ReLU
        lst_layer_dim = args.net_arch[0]
        encoder = []
        for nxt_layer_dim in args.net_arch[1:]:
            encoder.append(nn.Linear(lst_layer_dim, nxt_layer_dim))
            encoder.append(activate_fn())
            lst_layer_dim = nxt_layer_dim


        self.fc1 = nn.Linear(input_shape, args.net_arch[0])#从输入到mlp的第一个维度，用来做跳跃连接
        self.activate_fn1 = activate_fn()
        self.encoder_mlp = nn.Sequential(*encoder)
        self.rnn = nn.GRUCell(lst_layer_dim , hidden_size=args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + args.net_arch[0], args.n_actions)

    def init_hidden(self):
        # make hidden states on same device and type as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x0 = self.activate_fn1(self.fc1(inputs))
        x = self.encoder_mlp(x0)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(torch.concatenate([h,x0],dim=1))
        return q, h
