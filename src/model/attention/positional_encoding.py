import torch
import math
from torch import Tensor
from torch import nn
from modules.logger import log


class PositionalEncoding(nn.Module):
    """
    This class is made to provide the positional encoding
    originally presented in [Vaswani et al, 2017].

    For the gradient : here it is just a sum. Then there is no problem with the gradient.
    Because it is a constant the gradient will be zero.
    It won't play in the final gradient

    Here we will use the broadcast properties of pytorch to create the positional encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # in our input the batch is at the first position --> we transpose to put it in the middle
        x = torch.transpose(x, dim0=0, dim1=1)
        # add the positional encoding
        x = x + self.pe[:x.size(0)]
        # put the dimension back in the right place
        x = torch.transpose(x, dim0=0, dim1=1)
        return self.dropout(x)
