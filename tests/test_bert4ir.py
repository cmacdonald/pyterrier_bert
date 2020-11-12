
import pandas as pd
import pyterrier as pt
import os
import unittest
import shutil
import tempfile
import numpy as np

from pyterrier_bert.bert4ir import *
from transformers import *

class TestBERT4IR(unittest.TestCase):

    def __init__(self):
        if not pt.started():
            pt.init()

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_caching_dataset(self):
        df = pd.DataFrame([["q1", "query text", "d1", "doc text", 1]], columns=["qid", "query", "docno", "text", "label"])
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = CachingDFDataset(df, tokenizer, split="train", directory=self.test_dir, get_doc_fn=lambda row: row["text"])
        self.assertEqual(len(dataset), 1)
        # make sure we can obtain with 0.
        x = dataset[0]
        self.assertEqual(4, len(x))
    
    def test_dataset(self):
        df = pd.DataFrame([["q1", "query text", "d1", "doc text", 1]], columns=["qid", "query", "docno", "text", "label"])
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = DFDataset(df, tokenizer, split="train", get_doc_fn=lambda row: row["text"])
        self.assertEqual(len(dataset), 1)
        # make sure we can obtain with 0.
        x = dataset[0]
        self.assertEqual(4, len(x))

    def test_fit(self):
        longdocstring = "this is an irrelevant document. "
        longdoclist = longdocstring.split(" ")
        longdoc = longdoclist * 105
        self.assertTrue(len(longdoc) > 512)
        df = pd.DataFrame([
            ["q1", "query text", "d1", "doc text", 1],
            ["q1", "query text", "d1", longdocstring, 0],
            ["q1", "query text", "d1", " ".join(longdoc), 0],
            ], columns=["qid", "query", "docno", "text", "label"])

        pipe = BERTPipeline(get_doc_fn=lambda row: row["text"])
        pipe.fit(df, None, df, None)
        pipe.save(self.test_dir + "/model.weights")
        rtr = pipe.transform(df)
        self.assertEqual(3, len(rtr))
        self.assertTrue("score" in rtr.columns)
        pipe.test_batch_size = 1
        rtr = pipe.transform(df)
        self.assertEqual(3, len(rtr))
        self.assertTrue("score" in rtr.columns)
        
        # check we can save and load a model
        pipe = None
        pipe = BERTPipeline(get_doc_fn=lambda row: row["text"])
        pipe.load(self.test_dir + "/model.weights")
        rtr2 = pipe.transform(df)
        self.assertTrue(np.allclose(rtr2["score"], rtr["score"]))

        # check we can fit when using a caching dataset
        pipe = None
        pipe = BERTPipeline(get_doc_fn=lambda row: row["text"], cache_threshold=0)
        pipe.fit(df, None, df, None)
