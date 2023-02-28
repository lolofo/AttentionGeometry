# metrics for the evaluation of the geometry
""" 
This file contains all the functions for the notebook S01EP02.
In this experiment we try to evaluate the geometry of the attention through the entropy regularization.
"""
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from typing import Optional, Tuple


def cosine_sim(W: Tensor, attention_mask: Tensor, normalize: str = "mean") -> Tensor:
    """ Cosine similarity function

    The objective of this function is to calculate the cosine similarity between the rows of a matrix

    Some notations :

    N : the batch size
    T(s) : the number of tokens in the treated sentence
    d : the embedding dimension

    Args:
        W: Tensor. tensor of the shape (N, T(s), d)
        attention_mask: Tensor<bool>. Boolean tensor of shape (N, T(s)) == 1 iff the token is a not padding one
        normalize: str. Indicate the shape of the output. If normalize is mean we proceed a mean over a batch
                        else, we return the tensor of the shape [N]

    Returns:
        float in the range [-1, 1] this float describes the geometry of the row of the tensor W
    """

    assert len(W.shape) == 3, "error : must give a batch tensor of shape [N, T(s), d]"
    assert len(attention_mask.shape) == 2, "error : must give a batch tensor of shape [N, T(s)]"

    N, T, d = W.shape

    # >> dot product between embeddings
    dot_prod = torch.bmm(W, torch.transpose(W, 1, 2))
    v = torch.diagonal(dot_prod, offset=0, dim1=1, dim2=2).unsqueeze(-1)
    nms = torch.sqrt(torch.bmm(v, torch.transpose(v, 1, 2)))
    s_mat = dot_prod / (nms + 1e-16)

    # >> get rid of the padding tokens
    buff = attention_mask.float().unsqueeze(1).repeat(1, T, 1)
    s_mat = s_mat * buff
    s_mat = s_mat * torch.transpose(buff, 1, 2)

    # >> for each sentences s : get the quantity T(s)
    t_s = attention_mask.float().sum(dim=-1)

    # >> proceed the metric calculus
    den = t_s * (t_s - 1)
    num = s_mat.sum(dim=-1).sum(dim=-1) - torch.diagonal(s_mat, offset=0, dim1=1, dim2=2).sum(dim=-1)

    if normalize == "mean":
        # we normalize the quantity over the batch
        res = (num / den).mean()
    else:
        res = num / den

    return res


#
def main(bert_model,data_module, verbose : bool = True):
    """

    The objective

    Args:
        bert_model (_type_): _description_
        data_module (_type_): _description_
        verbose (bool)
    """
    v_print = print if verbose else lambda *a, **k: None
    v_print(">> The main function for S01EP02 session")

    DEVICE = bert_model.device
    test_dataloader = data_module.test_dataloader()

    key_map = np.zeros((12, 12)) # dim 1: layer | dim 2: head
    val_map = np.zeros((12, 12)) # dim 1: layer | dim 2: head

    with torch.no_grad():
        for batch in test_dataloader:

            ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_masks"].bool().to(DEVICE)

            output = bert_model(ids, attention_mask)
            bert_output = output["outputs"]

            # get the keys and the values
            key, val = bert_output["past_key_values"] 
            for l in range(12):
                for h in range(12):
                    # for the keys
                    curr_key = key[l][:, h, :, :]
                    key_map[l, h] += cosine_sim(curr_key, attention_mask).cpu().numpy()

                    # for the values
                    curr_val = val[l][:, h, :, :]
                    val_map[l, h] += cosine_sim(curr_val, attention_mask).cpu().numpy()

            # for each attention head

    return key_map, val_map