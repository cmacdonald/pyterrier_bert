from pyterrier.transformer import TransformerBase

import os
import sys
from os.path import dirname
#sys.path.append(os.path.join( dirname(__file__), '..', 'ColBERT'))
from multiprocessing import Pool

from . import add_label_column

import torch
import pandas as pd

import random
from colbert.evaluation.load_model import load_model
from colbert.modeling.inference import ModelInference
from colbert.evaluation.slow import slow_rerank
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
        args.mask_punctuation = True
        args.amp = True
        args.colbert, args.checkpoint = load_model(args)
        args.inference = ModelInference(args.colbert, amp=args.amp)
        self.args = args
        self.doc_attr = doc_attr
        self.verbose = verbose

    def transform(self, queries_and_docs):
        groupby = queries_and_docs.groupby("qid")
        rtr=[]
        with torch.no_grad():
            for qid, group in tqdm(groupby, total=len(groupby), unit="q") if self.verbose else groupby:
                query = group["query"].values[0]
                ranking = slow_rerank(self.args, query, group["docno"].values, group[self.doc_attr].values)
                for rank, (score, pid, passage) in enumerate(ranking):
                        rtr.append([qid, query, pid, score, rank])          
        return pd.DataFrame(rtr, columns=["qid", "query", "docno", "score", "rank"])

    
