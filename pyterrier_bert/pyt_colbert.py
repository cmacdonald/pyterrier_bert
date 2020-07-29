from pyterrier.transformer import TransformerBase

import os
import sys
from os.path import dirname
sys.path.append(os.path.join( dirname(__file__), '..', 'ColBERT'))
from multiprocessing import Pool

from . import add_label_column

import torch
import pandas as pd

import random
from src.evaluation.loaders import load_colbert
from src.evaluation.ranking import rerank
from tqdm import tqdm

from collections import defaultdict

class Object(object):
    pass




class ColBERTPipeline(TransformerBase):

    def __init__(self, checkpoint, model_name='bert-base-uncased', tokenizer_name='bert-base-uncased', doc_attr="body", verbose=False):
        args = Object()
        args.query_maxlen = 32
        args.doc_maxlen = 180
        args.dim = 128
        args.bsize = 128
        args.similarity = 'cosine'
        args.checkpoint = checkpoint
        args.pool = Pool(10)
        args.bert = model_name
        args.bert_tokenizer = tokenizer_name
        args.colbert, args.checkpoint = load_colbert(args)
        self.args = args
        self.doc_attr = doc_attr
        self.verbose = verbose

    def transform(self, queries_and_docs):
        groupby = queries_and_docs.groupby("qid")
        rtr=[]
        with torch.no_grad():
            for qid, group in tqdm(groupby, total=len(groupby), unit="q") if self.verbose else groupby:
                query = group["query"].values[0]
                ranking = rerank(self.args, query, group["docno"].values, group[self.doc_attr].values, index=None)
                for rank, (score, pid, passage) in enumerate(ranking):
                        rtr.append([qid, query, pid, score, rank])          
        return pd.DataFrame(rtr, columns=["qid", "query", "docno", "score", "rank"])

    
