
import pandas as pd
import pyterrier_bert
from pyterrier_bert.passager import SlidingWindowPassager, MaxPassage, FirstPassage, MeanPassage
import unittest
class TestPassager(unittest.TestCase):
    
    def test_passager_title(self):
        dfinput = pd.DataFrame([["q1", "a query", "doc1", "title", "body sentence"]], columns=["qid", "query", "docno", "title", "body"])
        passager = SlidingWindowPassager(passage_length=1, passage_stride=1)
        dfoutput = passager(dfinput)
        self.assertEqual(2, len(dfoutput))
        self.assertEqual("doc1%p0", dfoutput["docno"][0])
        self.assertEqual("doc1%p1", dfoutput["docno"][1])
        
        self.assertEqual("title body", dfoutput["body"][0])
        self.assertEqual("title sentence", dfoutput["body"][1])

    def test_passager(self):
        dfinput = pd.DataFrame([["q1", "a query", "doc1", "body sentence"]], columns=["qid", "query", "docno", "body"])
        passager = SlidingWindowPassager(passage_length=1, passage_stride=1, prepend_title=False)
        dfoutput = passager(dfinput)
        self.assertEqual(2, len(dfoutput))
        self.assertEqual("doc1%p0", dfoutput["docno"][0])
        self.assertEqual("doc1%p1", dfoutput["docno"][1])
        
        self.assertEqual("body", dfoutput["body"][0])
        self.assertEqual("sentence", dfoutput["body"][1])

    def test_depassager(self):
        dfinput = pd.DataFrame([["q1", "a query", "doc1", "title", "body sentence"]], columns=["qid", "query", "docno", "title", "body"])
        passager = SlidingWindowPassager(passage_length=1, passage_stride=1)
        dfpassage = passager(dfinput)
        #     qid    query    docno            body                                         
        # 0  q1  a query  doc1%p0      title body
        # 1  q1  a query  doc1%p1  title sentence
        dfpassage["score"] = [1, 0]

        dfmax = MaxPassage()(dfpassage)
        self.assertEqual(1, len(dfmax))
        self.assertEqual(1, dfmax["score"][0])

        dffirst = FirstPassage()(dfpassage)
        self.assertEqual(1, len(dffirst))
        self.assertEqual(1, dffirst["score"][0])

        dfmean = MeanPassage()(dfpassage)
        self.assertEqual(1, len(dfmean))
        self.assertEqual(0.5, dfmean["score"][0])



