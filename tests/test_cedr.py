
import pandas as pd
import pyterrier as pt
import os
import unittest

from pyterrier_bert.pyt_cedr import CEDRPipeline

class TestCEDR(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not pt.started():
            pt.init()

    def test_fit(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest(reason="No CUDA available")
        qrels = pd.DataFrame([
            ["q1", "d1", 1],
            ["q1", "d2", 0],
        ], columns=["qid", "docno", "label"])
        df = pd.DataFrame([
            ["q1", "query text", "d1", "doc text"],
            ["q1", "query text", "d2", "this is an irrelevant document"],
            ], columns=["qid", "query", "docno", "text"])

        pipe = CEDRPipeline(doc_attr="text")
        pipe.fit(df, qrels, df, qrels)
        rtr = pipe.transform(df)
        self.assertEqual(2, len(rtr))
        self.assertTrue("score" in rtr.columns)
        print(rtr)
