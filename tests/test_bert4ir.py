
import pandas as pd
import pyterrier as pt
import os
import unittest

from pyterrier_bert.bert4ir import *
from transformers import *

class TestBERT4IR(unittest.TestCase):

    def test_caching_dataset(self):
        df = pd.DataFrame([["q1", "query text", "d1", "doc text", 1]], columns=["qid", "query", "docno", "text", "label"])
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = CachingDFDataset(df, tokenizer, "train")
        self.assertEqual(len(dataset), 1)
        # make sure we can obtain with 0.
        x = dataset[0]
        self.assertEqual(4, len(x))
    
    def test_dataset(self):
        df = pd.DataFrame([["q1", "query text", "d1", "doc text", 1]], columns=["qid", "query", "docno", "text", "label"])
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = CachingDFDataset(df, tokenizer, "train")
        self.assertEqual(len(dataset), 1)
        # make sure we can obtain with 0.
        x = dataset[0]
        self.assertEqual(4, len(x))

    def test_fit(self):
        df = pd.DataFrame([
            ["q1", "query text", "d1", "doc text", 1],
            ["q1", "query text", "d1", "this is an irrelevant document", 0],
            ], columns=["qid", "query", "docno", "text", "label"])

        pipe = BERTPipeline()
        pipe.fit(df, None, df, None)
        rtr = pipe.transform(df)
        self.assertEqual(2, len(rtr))
        self.assertTrue("score" in rtr.columns)
        print(rtr)
