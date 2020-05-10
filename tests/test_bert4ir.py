
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

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_caching_dataset(self):
        df = pd.DataFrame([["q1", "query text", "d1", "doc text", 1]], columns=["qid", "query", "docno", "text", "label"])
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = CachingDFDataset(df, tokenizer, "train", directory=self.test_dir)
        self.assertEqual(len(dataset), 1)
        # make sure we can obtain with 0.
        x = dataset[0]
        self.assertEqual(4, len(x))
    
    def test_dataset(self):
        df = pd.DataFrame([["q1", "query text", "d1", "doc text", 1]], columns=["qid", "query", "docno", "text", "label"])
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        dataset = DFDataset(df, tokenizer, "train")
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
        pipe.save(self.test_dir + "/model.weights")
        rtr = pipe.transform(df)
        self.assertEqual(2, len(rtr))
        self.assertTrue("score" in rtr.columns)
        
        pipe = None
        pipe = BERTPipeline()
        pipe.load(self.test_dir + "/model.weights")
        rtr2 = pipe.transform(df)
        self.assertTrue(np.allclose(rtr2["score"], rtr["score"]))
