import torch
import numpy as np
from notebooks.BERT.utils.sim_mesure import cosine_sim


def main(bert_model, data_module, nb_data):
    res = []
    DEVICE = bert_model.device
    test_dataloader = data_module.test_dataloader()

    with torch.no_grad():
        for batch in test_dataloader:
            ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_masks"].bool().to(DEVICE)

            output = bert_model(ids, attention_mask)
            bert_output = output["outputs"]

            H = bert_output.hidden_states
            temp = []

            for i, h in enumerate(H):
                temp.append(cosine_sim(h, attention_mask, normalize=None).cpu().numpy().reshape(1, ids.shape[0]))
            temp = np.concatenate(temp, axis=0)
            res.append(temp)

    return np.concatenate(res, axis=1)