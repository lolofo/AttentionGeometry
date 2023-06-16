"""
This file is for the first experience : S01EP01
The experience is to observe if we have one classe for one convergence
"""
import pandas as pd
import numpy as np
import tqdm
import torch
from torch import Tensor



def compute_cos(rep: Tensor, vectors: Tensor) -> Tensor:
    """compute_cos

    This function will compute the cosines between a vector and a 2D tensor.
    Rep for the interpretation will be the representative of a certain class.

    Args:
        rep (Tensor): the main vector we want to make the comparison with
        vectors (Tensor): the vectors we want to now in which class they are.

    Returns:
        a 1D tensor v where v_i = cos(rep, vectors[i, :])
    """
    assert len(rep.shape) == 1, "error : batch is given"
    assert rep.shape[0] == vectors.shape[1], "errors : got unexpected dimension error"

    dot_prod = torch.matmul(vectors, rep)
    nms = torch.norm(vectors, dim=-1)

    # we return the cosine similarity
    return dot_prod / (nms * torch.norm(rep))


def search_rep(model, dm, ft):
    """

    Args:
        model: the model
        dm: the data module (data for the model)

    Returns:

    """
    DEVICE = model.device  # to have no device conflict.
    count = 0

    res = {
        f"class_{i}": None for i in range(dm.num_class)
    }

    for class_rep in range(dm.num_class):
        class_rep_found = False  # indicate if we found the representative of class_rep
        # search for a rep of the class_rep class
        for id_batch, batch in enumerate(dm.test_dataloader()):

            # batch elements
            y_true = batch["labels"].to(DEVICE)
            bs = y_true.shape[0]  # the batch size

            ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_masks"].bool().to(DEVICE) # [B, L]

            # output of the model
            output = model(ids, attention_mask)
            bert_output = output["outputs"]

            # the prediction we make
            cl = output["logits"].argmax(dim=-1)

            mask = attention_mask.float().unsqueeze(-1).repeat(1, 1, 768)
            nb_tokens = attention_mask.sum(dim=-1).unsqueeze(-1).repeat(1, 768)

            for i in range(bs):
                # prediction equals the class and the prediction is good
                # we treat here the element i of the batch.
                if ft:
                    if cl[i] == class_rep and cl[i] == y_true[i]:
                        # we found a rep of the class class_rep
                        rep_emb = []

                        for lay in range(len(bert_output.hidden_states)):
                            # get the [CLS] embedding
                            rep_emb.append((bert_output.hidden_states[lay][i, 0, :]))

                        res[f"class_{class_rep}"] = {
                            "emb": rep_emb
                        }

                        class_rep_found = True
                        count += 1
                        break

            if class_rep_found:
                break
        if count == 3:
            break
    return res


#######################################################################
# the main function to execute in the different cells of our notebook #
#######################################################################
def main(bert_model, data_module, nb_data, batch_size,  ft=True, representative=None, verbose=True):
    """main

    This function is the main function to execute in the notebook S01EP01.
    We will compute all the different cosines possible

    Args:
        bert_model : a bert type model
        data_module : a dataloader which corresponds to the BERT model.
        ft (bool): a boolean parameter to know if the model is a fine-tuned one (True)
        representative (Tensor) : a tensor (can be none type). Give the representative for the raw model
        verbose(bool): classic boolean to print some logs

    Returns:
        This function returns a tuple.
        * the first comp of the tuple is the dataframe of the different cosine values
        * the second component of the tuple is the representative we use for our computation
    """
    v_print = print if verbose else lambda *a, **k: None

    DEVICE = bert_model.device
    test_dataloader = data_module.test_dataloader()

    emb_cos_values = {
        f"cos_class_{c}": np.array([]) for c in range(data_module.num_class)
    }
    predictions = []

    class_id = np.array([])

    with torch.no_grad():
        # search the representative for our embeddings
        if ft:
            v_print(">> Searching the rep")
            class_rep_res = search_rep(bert_model, data_module, ft)
        else:
            v_print(">> Re-use the rep")
            class_rep_res = representative
        v_print(">> rep found !")

        v_print(">> start the loop")
        for batch in test_dataloader:

            ids = batch["input_ids"].to(DEVICE)
            class_id = np.concatenate((class_id, batch["labels"].to(DEVICE).cpu().numpy()))
            attention_mask = batch["attention_masks"].bool().to(DEVICE)

            output = bert_model(ids, attention_mask)
            bert_output = output["outputs"]

            # for every sentence we get the embedding of the CLS token
            emb = bert_output.hidden_states[-1][:, 0, :]
            pred = list(torch.argmax(output["logits"], dim=-1).cpu().numpy())
            predictions += pred

            for c in range(data_module.num_class):
                rep_emb = class_rep_res[f"class_{c}"]["emb"][-1]
                emb_cos_values[f"cos_class_{c}"] = np.concatenate((emb_cos_values[f"cos_class_{c}"],
                                                                    compute_cos(rep_emb, emb).cpu().numpy()))

        res = pd.DataFrame(emb_cos_values)
        res["label"] = class_id
    v_print(">> all done !")
    return res, class_rep_res, predictions



def bert_type_main(bert_model, data_module, nb_data, batch_size,  ft=True, representative=None, verbose=True):
    """bert_type_main

    This function is the main function to execute in the notebook S01EP01.
    We will compute all the different cosines possible

    Args:
        bert_model : a bert type model
        data_module : a dataloader which corresponds to the BERT model.
        ft (bool): a boolean parameter to know if the model is a fine-tuned one (True)
        representative (Tensor) : a tensor (can be none type). Give the representative for the raw model
        verbose(bool): classic boolean to print some logs

    Returns:
        This function returns a tuple.
        * the first comp of the tuple is the dataframe of the different cosine values
        * the second component of the tuple is the representative we use for our computation
    """
    v_print = print if verbose else lambda *a, **k: None

    DEVICE = bert_model.device

    emb_cos_values = {
        f"cos_class_{c}": np.array([]) for c in range(data_module.num_class)
    }
    predictions = []

    class_id = np.array([])

    with torch.no_grad():
        # search the representative for our embeddings
        if ft:
            v_print(">> Searching the rep")
            class_rep_res = search_rep(bert_model, data_module, ft)
        else:
            v_print(">> Re-use the rep")
            class_rep_res = representative
        v_print(">> rep found !")

        v_print(">> start the loop")
        for batch in test_dataloader:

            ids = batch["input_ids"].to(DEVICE)
            class_id = np.concatenate((class_id, batch["labels"].to(DEVICE).cpu().numpy()))
            attention_mask = batch["attention_masks"].bool().to(DEVICE)

            output = bert_model(ids, attention_mask)
            bert_output = output["outputs"]

            # for every sentence we get the embedding of the CLS token
            emb = bert_output.hidden_states[-1][:, 0, :]
            pred = list(torch.argmax(output["logits"], dim=-1).cpu().numpy())
            predictions += pred

            for c in range(data_module.num_class):
                rep_emb = class_rep_res[f"class_{c}"]["emb"][-1]
                emb_cos_values[f"cos_class_{c}"] = np.concatenate((emb_cos_values[f"cos_class_{c}"],
                                                                    compute_cos(rep_emb, emb).cpu().numpy()))

        res = pd.DataFrame(emb_cos_values)
        res["label"] = class_id
    v_print(">> all done !")
    return res, class_rep_res, predictions