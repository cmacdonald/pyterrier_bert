from pyterrier.transformer import TransformerBase

import os
import sys
from os.path import dirname
sys.path.append(os.path.join( dirname(__file__), '..', 'ColBERT'))
from multiprocessing import Pool

from . import add_label_column

from src.evaluation.loaders import load_colbert
from src.evaluation.ranking import rerank

from collections import defaultdict

class Object(object):
    pass

class ColBERTPipeline(TransformerBase):

    def __init__(self, checkpoint, doc_attr="body"):
        args = Object()
        args.query_maxlen = 32
        args.doc_maxlen = 180
        args.dim = 128
        args.similarity = 'cosine'
        args.checkpoint = checkpoint
        args.pool = Pool(10)
        args.colbert, args.checkpoint = load_colbert(args)
        self.args = args
        self.doc_attr = doc_attr

    def transform(self, queries_and_docs):
        #we may not need this. we could simply do a group by
        args.queries, args.topK_docs, args.topK_pids = load_topK(queries_and_docs)
        keys = sorted(list(queries.keys()))
        #WHY? 
        random.shuffle(keys)
        rtr = []
        for query_idx, qid in enumerate(keys):
            query = queries[qid]
            ranking = rerank(self.args, query, topK_pids[qid], topK_docs[qid], index)
            for i, (score, pid, passage) in enumerate(ranking):
                rtr.append([qid, pid, score, rank])
        return pd.DataFrame(rtr, columns=["qid", "docno", "score", "rank"])

    def make_topK(queries_and_docs):
        queries = {}
        topK_docs = defaultdict(list)
        topK_pids = defaultdict(list)
        for qd in queries_and_docs.itertuples():
            qid = qd.qid
            queries[qid] = qd.query
            body = get_attr(qd, self.doc_attr)
            topK_docs[qid].append(body)
            topK_pids[qid].append(qd.docno)
        return queries, topK_docs, topK_pids
