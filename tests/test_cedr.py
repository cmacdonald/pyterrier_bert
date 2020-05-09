
import pandas as pd
import pyterrier as pt
import os
import unittest

from pyterrier_bert.pyt_cedr import CEDRPipeline

class TestCEDR(unittest.TestCase):

    def test_fit(self):
        df = pd.DataFrame([
            ["q1", "query text", "d1", "doc text", 1],
            ["q1", "query text", "d1", "this is an irrelevant document", 0],
            ], columns=["qid", "query", "docno", "text", "label"])

        pipe = CEDRPipeline(doc_attr="text")
        pipe.fit(df, None, df, None)
        rtr = pipe.transform(df)
        self.assertEqual(2, len(rtr))
        self.assertTrue("score" in rtr.columns)
        print(rtr)
