import os
import shutil

import pandas as pd
from os import path
import pytorch_lightning as pl
import pickle
import spacy
import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import MapDataPipe, DataLoader

import numpy as np
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab.vectors import pretrained_aliases as pretrained
import env

from DataModules.utils import yelp_hat_ham, yelp_hat_token
from logger import log
import torchtext.transforms as T
from transformers import BertTokenizer


class ArgumentError(ValueError):
    pass


DATASET_NAME = 'yelp-hat'

_EXTRACTED_FILES = {
    'yelp': 'yelp.csv',
    'yelp50': 'yelp50.csv',
    'yelp100': 'yelp100.csv',
    'yelp200': 'yelp200.csv',

    'train': 'train.csv',
    'val': 'val.csv',
}

_SUBSET = {
    'ham_part1(50words).csv': 'yelp50.csv',
    'ham_part6(100words).csv': 'yelp100.csv',
    'ham_part8(200words).csv': 'yelp200.csv'
}

_TRAIN_VAL_SPLIT = 0.3

URL = 'https://github.com/cansusen/Human-Attention-for-Text-Classification/archive/205c1552bc7be7ec48623d79d85d4c6fbfe62362.zip'

tk = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_row(row):
    post_tokens = row[0]
    rationale = row[1]

    pos_tk_ids = []
    rat = []

    for id_w, w in enumerate(post_tokens):
        buff = w
        if "<" in buff:
            buff = buff.replace("<", "")
        if ">" in buff:
            buff = buff.replace(">", "")

        t = tk(buff)
        # remove special tokens
        ids = t.input_ids
        _ = ids.pop(0)
        _ = ids.pop(-1)
        pos_tk_ids += ids

        if rationale[id_w] == 1:
            rat += [1] * len(ids)
        else:
            rat += [0] * len(ids)

    return {"bert_ids": pos_tk_ids,
            "bert_post_tokens": tk.convert_ids_to_tokens(pos_tk_ids),
            "bert_rationale": rat}


def get_bert_ids(row):
    return row[3]["bert_ids"]


def get_bert_post_tok(row):
    return row[3]["bert_post_tokens"]


def get_bert_rationale(row):
    return row[3]["bert_rationale"]


def download_format_dataset(root: str, split: str = 'yelp', spacy_model=None):
    """
    Download and reformat dataset of eSNLI
    Args:
        root (str): cache folder where to find the dataset.
        split (str): among train, val, test
        n_data (int): maximum data to load. -1 to load entire data
    """

    if path.basename(root) != DATASET_NAME:
        root = path.join(root, DATASET_NAME)

    zip_path = download_from_url(URL, root=root, path=path.join(root, f'{DATASET_NAME}.zip'))
    extracted_path = path.join(root, 'caching')
    files = extract_archive(from_path=zip_path, to_path=extracted_path)
    files = [f for f in files if f.endswith('.csv')]

    # If path exists already, ignore doing things
    csv_path = path.join(root, _EXTRACTED_FILES[split])
    if path.exists(csv_path):
        return csv_path

    # Place .csv files at the level of dataset
    for f in files: shutil.copy2(f, extracted_path)

    # Special case of part7.csv: contains 2 HAMs and 4 HAMs for some sentences
    df = pd.read_csv(path.join(extracted_path, 'ham_part7.csv'))

    # 1. Duplicate of 2: drop all
    duplicates = df.groupby(df['Input.text'].tolist(), as_index=False).size()
    dupli_2 = duplicates[duplicates['size'] < 3]  # finds dupli_2
    df = df[~df['Input.text'].isin(dupli_2['index'])]

    # 2. Duplicate of 4:
    df = df.groupby('Input.text').head(3).reset_index(drop=True)

    # Check if no other duplicate in part7
    duplicates = df.groupby(df['Input.text'].tolist(), as_index=False).size()
    duplicated_values = duplicates['size'].unique()
    if len(duplicated_values) == 1 and duplicated_values[0] == 3:
        log.debug('Correctly handle part7.csv')
    else:
        log.error(f'Unsuccessfully handle part7.csv. Duplicated HAM: {duplicated_values}')
        raise ArithmeticError

    df.to_csv(path.join(extracted_path, 'ham_part7.csv'), index=False)

    # Special case of part5.csv: drop duplicate 2
    df = pd.read_csv(path.join(extracted_path, 'ham_part5.csv'))
    duplicates = df.groupby(df['Input.text'].tolist(), as_index=False).size()
    dupli_2 = duplicates[duplicates['size'] == 2]  # finds dupli_2
    df = df[~df['Input.text'].isin(dupli_2['index'])]
    df.to_csv(path.join(extracted_path, 'ham_part5.csv'), index=False)

    # Now reformat every files and save them to extracted
    training_sets = []
    files = [path.join(extracted_path, f) for f in os.listdir(extracted_path) if f.endswith('.csv')]
    if spacy_model is None: spacy_model = spacy.load('en_core_web_sm')  # use en_core_web_sm by default if not set
    for f in files:
        log.debug(f'Reformat {f}')
        df = pd.read_csv(f)
        df = _reformat_csv(df, spacy_model=spacy_model)
        csv_path = _SUBSET.get(path.basename(f), False)
        if csv_path:
            csv_path = path.join(root, csv_path)
            df.to_csv(csv_path, index=False)
            log.info(f'Save yelp subset at: {csv_path}')
        else:
            training_sets.append(df)

    training_df = pd.concat(training_sets, ignore_index=True)
    training_df.to_csv(path.join(root, _EXTRACTED_FILES['yelp']), index=False)
    log.info(f'Save clean dataset at {path.join(root, _EXTRACTED_FILES["yelp"])}')

    # Spliting data set into train and val
    train_df, val_df = train_test_split(training_df, test_size=0.3)
    train_df.to_csv(path.join(root, _EXTRACTED_FILES['train']), index=False)
    log.info(f'Save training set at {path.join(root, _EXTRACTED_FILES["train"])}')
    val_df.to_csv(path.join(root, _EXTRACTED_FILES['val']), index=False)
    log.info(f'Save clean dataset at {path.join(root, _EXTRACTED_FILES["val"])}')

    return path.join(root, _EXTRACTED_FILES[split])


def clean_cache(root: str):
    shutil.rmtree(path.join(root, 'caching'), ignore_errors=True)


def _reformat_csv(data: pd.DataFrame, spacy_model):
    # Binarizing human attention map
    data['ham_'] = data[f'Answer.html_output'].apply(lambda x: yelp_hat_ham(x, spacy_model)).apply(
        lambda x: np.array(x))
    # Pre tokenize
    data['text_tokens'] = data['Answer.html_output'].apply(lambda x: yelp_hat_token(x, spacy_model))
    # Rename column
    data = data.rename(columns={'Answer.Q1Answer': 'human_label_', 'Input.text': 'text', 'Input.label': 'label'})

    # Sliding into 3 subset to get 3HAMs
    dfs = [data.loc[0::3, ['text', 'label', 'text_tokens']].reset_index(drop=True)]
    for idx in range(3):
        _data = data.loc[idx::3, ['ham_', 'human_label_']]
        _data = _data.reset_index(drop=True).add_suffix(str(idx))
        dfs += [_data]

    data = pd.concat(dfs, axis=1)

    # Drop incoherent attention maps samples
    data_drop = data[(data['ham_0'].str.len() == data['ham_1'].str.len()) & (
            data['ham_1'].str.len() == data['ham_2'].str.len())].reset_index()
    n_drop = len(data) - len(data_drop)

    if n_drop > 0:
        log.warning(f'Drop {n_drop} samples because HAMs are not compatibles')
        data = data_drop

    data['ham'] = data.apply(lambda row: ((row['ham_0'] + row['ham_1'] + row['ham_2']) / 3 >= 0.5).astype(int), axis=1)
    data['cam'] = data.apply(lambda row: np.logical_and(row['ham_0'], row['ham_1'], row['ham_2']), axis=1)
    data['sam'] = data.apply(lambda row: np.logical_or(row['ham_0'], row['ham_1'], row['ham_2']), axis=1)

    # convert numpy into list:
    for column in ['ham_0', 'ham_1', 'ham_2', 'ham', 'cam', 'sam']:
        data[column] = data[column].apply(lambda x: x.tolist())
    buff = data[["text_tokens", "ham", "label"]]
    buff["buff"] = buff.apply(tokenize_row, axis=1)
    buff["bert_ids"] = buff.apply(get_bert_ids, axis=1)
    buff["bert_text_tokens"] = buff.apply(get_bert_post_tok, axis=1)
    buff["bert_ham"] = buff.apply(get_bert_rationale, axis=1)
    buff = buff.drop(columns=["buff", "text_tokens", "ham", "label"])
    data = pd.concat(data, buff)
    return data


class YelpHat(MapDataPipe):

    def __init__(self, split: str = 'yelp',
                 root: str = path.join(os.getcwd(), '.cache'),
                 n_data: int = -1, spacy_model=None):

        # assert
        if split not in _EXTRACTED_FILES.keys():
            raise ArgumentError(f'split argument {split} doesnt exist for {type(self).__name__}')

        root = self.root(root)
        self.split = split

        # download and prepare csv file if not exist
        self.csv_path = download_format_dataset(root=root, split=split, spacy_model=spacy_model)

        coltype = {'label': 'category', 'text': str, 'human_label_0': 'category', 'human_label_1': 'category',
                   'human_label_2': 'category'}
        col_convert = {'text_tokens': pd.eval, 'ham_0': pd.eval, 'ham_1': pd.eval, 'ham_2': pd.eval, 'ham': pd.eval,
                       'cam': pd.eval, 'sam': pd.eval}

        # load the csv file to data
        self.data = pd.read_csv(self.csv_path, dtype=coltype, converters=col_convert)

        # if n_data activated, reduce the dataset equally for each class
        if n_data > 0:
            _unique_label = self.data['label'].unique()

            subset = [
                self.data[self.data['label'] == label]  # slice at each label
                    .head(n_data // len(_unique_label))  # get the top n_data/3
                for label in _unique_label
            ]
            self.data = pd.concat(subset).reset_index(drop=True)

    def __getitem__(self, index: int):
        """

        Args:
            index ():

        Returns:

        """

        # Load data and get label
        if index >= len(self): raise IndexError  # meet the end of dataset

        sample = self.data.loc[index].to_dict()

        return sample

    def __len__(self):
        """
        Denotes the total number of samples
        Returns: int
        """
        return len(self.data)

    @classmethod
    def root(cls, root):
        return path.join(root, DATASET_NAME)

    @classmethod
    def download_format_dataset(cls, root, split='yelp'):
        return download_format_dataset(root, split)

    @classmethod
    def clean_cache(cls, root):
        return clean_cache(root)


PAD_TOK = '<pad>'
UNK_TOK = '<unk>'


class LitData(pl.LightningDataModule):

    def __init__(self, cache_path, batch_size=8, num_workers=0, n_data=-1):
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        # Dataset already tokenized
        self.n_data = n_data
        self.num_workers = num_workers
        self.spacy_model = spacy.load('en_core_web_sm')

    def prepare_data(self):
        # called only on 1 GPU

        # download_dataset()
        dataset_path = YelpHat.root(self.cache_path)
        vocab_path = path.join(dataset_path, f'vocab.pt')

        YelpHat.download_format_dataset(dataset_path)  # only 1 line, download all dataset
        # build_vocab()
        if not path.exists(vocab_path):

            # return a single list of tokens
            def flatten_token(batch):
                return [token for sentence in batch['text_tokens'] for token in sentence]

            train_set = YelpHat(root=self.cache_path, split='train', n_data=self.n_data)

            # build vocab from train set
            dp = train_set.batch(self.batch_size).map(self.list2dict).map(flatten_token)

            # Build vocabulary from iterator. We don't know yet how long does it take
            iter_tokens = tqdm(iter(dp), desc='Building vocabulary', total=len(dp), unit='sents', file=sys.stdout,
                               disable=env.disable_tqdm)
            if env.disable_tqdm: log.info(f'Building vocabulary')
            vocab = build_vocab_from_iterator(iterator=iter_tokens, specials=[PAD_TOK, UNK_TOK])
            # vocab = build_vocab_from_iterator(iter(doc for doc in train_set['post_tokens']), specials=[PAD_TOK, UNK_TOK])
            vocab.set_default_index(vocab[UNK_TOK])

            # Announce where we save the vocabulary
            torch.save(vocab, vocab_path,
                       pickle_protocol=pickle.HIGHEST_PROTOCOL)  # Use highest protocol to speed things up
            iter_tokens.set_postfix({'path': vocab_path})
            if env.disable_tqdm: log.info(f'Vocabulary is saved at {vocab_path}')
            iter_tokens.close()
            self.vocab = vocab
        else:
            self.vocab = torch.load(vocab_path)
            log.info(f'Loaded vocab at {vocab_path}')

        log.info(f'Vocab size: {len(self.vocab)}')

        # Clean cache
        YelpHat.clean_cache(root=self.cache_path)

        self.ham_transform = T.Sequential(
            T.ToTensor(padding_value=0)
        )

        self.label_transform = T.Sequential(
            T.LabelToIndex(['0', '1']),
            T.ToTensor()
        )

        self.tensor_transform = T.Sequential(
            T.ToTensor(padding_value=0),  # put all the element of the batch to the same size
            T.PadTransform(max_length=150, pad_value=0)
        )

    def setup(self, stage: str = None):
        dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data)

        # called on every GPU
        if stage == 'fit' or stage is None:
            self.train_set = YelpHat(split='train', **dataset_kwargs)
            self.val_set = YelpHat(split='val', **dataset_kwargs)

        if stage == 'test' or stage is None:
            self.test_sets = {key: YelpHat(split=key, **dataset_kwargs) for key in ['yelp50', 'yelp100', 'yelp200']}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        loader_kwargs = dict(batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
                             num_workers=self.num_workers)
        # loaders = {dataset_name : DataLoader(dataset, **loader_kwargs) for dataset_name, dataset in self.test_sets.items()}
        loaders = [DataLoader(dataset, **loader_kwargs) for dataset_name, dataset in self.test_sets.items()]
        # return CombinedLoader(loaders, mode="max_size_cycle") # Run multiple test set in parallel
        return loaders

    ## ======= PRIVATE SECTIONS ======= ##
    def bert_collate(self, batch):
        # a bert collate
        batch = self.list2dict(batch)
        input_ids = self.tensor_transform(batch["bert_ids"])
        attention_masks = (input_ids == 0).type(torch.uint8)
        annotations = self.tensor_transform(batch["bert_rationale"])
        labels = self.label_transform(batch["label"])
        b = {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "labels": labels,
            "annotations": annotations
        }
        return b

    def list2dict(self, batch):
        # convert list of dict to dict of list

        if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}  # handle case where no batch
        return {k: [row[k] for row in batch] for k in batch[0]}
