from utils import *
from configs import *


class MyRNN(nn.Module):
    def __init__(self, n_neuron, act_func, decay=0.1, train_geometry=False):
        super(MyRNN, self).__init__()
        self.hidden_size = n_neuron
        self.w_hh = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size), requires_grad=True)
        self.d = decay
        self.act_func = act_func
        self.vec_embedding = nn.Embedding(3, self.hidden_size)

        self.use_scale_loss = False
        self.train_geometry = train_geometry

        self.geometry = nn.Linear(in_features=self.hidden_size, out_features=2, bias=True)
        self.readout = nn.Linear(in_features=2, out_features=6, bias=False)
        if not self.train_geometry:
            direction = nn.Parameter(torch.tensor(DIRECTION), requires_grad=False)
            self.readout.weight = direction
            self.readout.weight.requires_grad = False

        self.init_connection_weight()

    def init_rand_hidden(self):
        # initialize hidden state with(time, N)
        return torch.randn(self.hidden_size, requires_grad=True)

    def reset_hidden(self):
        return torch.zeros(self.hidden_size, requires_grad=True)

    def init_connection_weight(self):
        self.w_hh.data.normal_(0, math.sqrt(2 / self.hidden_size))

    def forward(self, hidden_0, batch_seq):
        """
        Arguments:
            hidden_0: the initial state of hidden layer, (hidden_size)
            batch_seq: torch.tensor, (seq_len, batch_size, feature_dim), a batch input to rnn
        Returns:
            hidden_t: state of hidden layer, (seq_len, batch_size, hidden_size)
            geometry_t: 2-dimension output of the batch, training target, (seq_len, batch_size, 2)
            readout_t: 6-dimension output of the batch, possible training target for cross-entropy loss, (seq_len, batch_size, 6)
            embedded_t: state of input after embedded, (seq_len, batch_size, hidden_size)
        """
        # initialize hidden state and hidden state time course
        hidden_t = torch.tensor([])
        batch_size, seq_len = batch_seq.shape[1], batch_seq.shape[0]
        # hidden (batch_size, hidden_size)
        hidden = hidden_0.unsqueeze(dim=0).repeat(batch_size, 1)

        axis = self.vec_embedding(torch.tensor([0, 1, 2]))
        embedded_t = batch_seq @ axis
        for frame in range(seq_len):
            hidden = (1 - self.d) * hidden + self.d * (
                    torch.einsum('ab,cb->ca', [self.w_hh, self.act_func(hidden)]) + embedded_t[frame, :, :])
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=0)), dim=0)
        geometry_t = self.geometry(self.act_func(hidden_t))
        readout_t = self.readout(geometry_t)
        return hidden_t, geometry_t, readout_t, embedded_t

    # # input geometry should be in critical period cause no additional slicing operation in the function
    # # scale determine the radius trained, if 1:normalize,if 0:fixed at zero point
    # def scale_loss(self, geometry, scale, temp):
    #     if not self.use_scale_loss:
    #         return 0
    #     else:
    #         L = len(geometry)
    #         norm = geometry[:, 0] ** 2 + geometry[:, 1] ** 2
    #         abs_norm = temp * sum(abs(norm - scale)) / L
    #         return abs_norm

    # go forward a single step with given input and hidden
    def get_one_step_fun(self):

        def fun(hidden, input):
            hidden_t = (1 - self.d) * hidden + self.d * (self.w_hh @ self.act_func(hidden) + input)
            geometry_t = self.geometry(self.act_func(hidden_t))
            readout_t = self.readout(geometry_t)
            return hidden_t, geometry_t, readout_t

        return fun
