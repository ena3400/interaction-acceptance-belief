import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor


class MLP(nn.Module):
    """
    MLP with one hidden layer
    """

    def __init__(self, in_dim, h_dim, out_dim):
        super(MLP, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        y = self.deterministic_output(x)
        return y


class SimpleRNN(nn.Module):
    """
    A simple GRU used as a baseline
    """

    def __init__(self, input_dim, hidden_dim, output_dim, attention_context=False, bidirectional=False):
        """
        .. note::
        For bidirectional GRUs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.
        :param input_dim:
        :param hidden_dim:
        :param attention_context:
        :param bidirectional:
        """
        super(SimpleRNN, self).__init__()
        print(f"GRU with {input_dim} input dim and {hidden_dim} hidden dim")
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.hidden_dim *= 2
        self.linear = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.attention_context = attention_context
        self.attention = DotProductAttention(self.hidden_dim)

    def forward(self, data):
        """
        :param data: (seq_len, batch_size, input_dim)
        :return: res: batch_size, hidden_dim)
                 If attention is on, a weighted sum of all hidden states.
                 Else, returns the last hidden state of the GRU.
        """
        output, h = self.rnn(data)
        # print(f"shape output: {output.shape}, shape hidden {h.shape}")
        # output: (batch_size, seq_len, hidden_dim). All hidden states -> the last one is the last of seq_len
        # h: (#_layers (1), batch_size, hidden_dim) last hidden state -> output[batch, -1, hidden_dim]

        if self.attention_context:
            res, attn = self.attention(output)
            res = res[:, -1, :]
            # res = torch.squeeze(res, dim=1)
        else:
            # We return the last hidden state of the GRU's last layer
            if self.bidirectional:
                res = output[:, -1, :]
            else:
                h = h.permute(1, 0, 2).mean(dim=1)  # h: (batch_size, 1, hidden_dim)
                res = h.view(-1, self.hidden_dim)  # res: (batch_size, hidden_dim)

        y = self.linear(res)
        # y = torch.squeeze(y, self.output_dim)
        y = torch.sigmoid(y)
        y = y.squeeze(-1)
        return y


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the
    values
    """

    def __init__(self, input_dim):
        super(DotProductAttention, self).__init__()
        self.linear_query = nn.Linear(input_dim, input_dim, bias=False)
        self.linear_value = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param input: (batch_size, seq_len, input_dim) hidden vectors
        :return: contex
        """
        query = self.linear_query(input)
        value = self.linear_value(input)
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn
