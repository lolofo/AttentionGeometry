# A study on the geometry around attention.

## ToC

  * [Authors](#authors)
  * [Introduction and background](#introduction-and-background)
  * [How to use this repository](#how-to-use-this-repository)
    + [Organization](#organization)
    + [Command lines](#command-lines)
  * [Principal references](#principal-references)

## Authors

- Loïc FOSSE (INSA Rennes)
- Duc-Hau Nguyen (CNRS - IRISA Rennes)
- Guillaume GRAVIER (CNRS - IRISA Rennes)

## Introduction and background

Transformers as reached state of the art in many NLP tasks. This is thanks to the existence of pre-trained models on very generic tasks (Devlin et al., 2019).
These models use a very complexe attention mecanism.
The arrival of this mechanism has created a lot of enthusiasm to explain the decisions of the model and to understand how the model works. The reality is quite different and a debate has been created in the NLP community. Debate which is sum up in (Bibal et al., 2022)
Here we try to produce an explanation on why all the opinion diverge on this subject.
Here we made the choice to use the framework of the geometry.

This study is the direct following of (Fosse et al., 2022), and is inspired from the previous work of (Ethayarajh, 2019)

## How to use this repository

### Organization

In this repository there is two main folders:
- `src` : which contains the code to load the data and train the models
- `notebook` : which contains the notebook and the graphics of the different experiements.

### Command lines

First start by creating the different folder.

```
mkdir .cache_bert
mkdir .cache_attention
```

To train BERT on both datasets :

```
python src/bert_model.py --dataset esnli -d .cache_bert/datasets/EsnliDataSet -e 50 -b 64 --experiment bert_esnli --version 0
python src/bert_model.py --dataset hatexplain .cache_bert/datasets/hatexplain -e 50 -b 64 --experiment bert_hatexplain --version 0
```

To train the *Pur attention* models with one head and one layer.
```
python src/pur_attention.py --data esnli -e 50 -b 64
python src/put_attention.py --data hatexplain -e 50 -b 64
```

Il you want to add more layers or more heads, use the parsers `--num_layers` or `--num_heads`.
If you want to regularize by the entropy of the attention map, use the `--lambda_entropy` parser.



## Principal references

[Bibal et al., 2022] Bibal, A., Cardon, R., Alfter, D., Wilkens, R., Wang, X., François, T., and Watrin, P.
(2022). Is Attention Explanation? An Introduction to the Debate. In Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3889–3900,
Dublin, Ireland. Association for Computational Linguistics.

[Devlin et al., 2019] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of
Deep Bidirectional Transformers for Language Understanding. Conference of the North {A}merican
Chapter of the Association for Computational Linguistics: Human Language Technologies.

[Ethayarajh, 2019] Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?
Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. Association for Computational
Linguistics, pages 55–65.

[Fosse et al., 2022] Fosse, L., Nguyen, D. H., Sébillot, P., and Gravier, G. (2022). Une étude statistique
des plongements dans les modèles transformers pour le français. TALN (traitement automatique des
langues naturelles, 2022, Avignon France), page 10.

[Nguyen et al., 2022] Nguyen, D. H., Gravier, G., and Sébillot, P. (2022). Filtrage et régularisation pour
améliorer la plausibilité des poids d’attention dans la tâche d’inférence en langue naturelle. TALN
(traitement automatique des langues naturelles, 2022, Avignon France), page 9.
