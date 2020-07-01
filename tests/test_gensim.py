import pandas as pd
import pyterrier_bert
from pyterrier_bert.gensim import GensimWordMoverDistance, GensimAverageSimilarity
import unittest
class TestGensim(unittest.TestCase):

    def test_gensim_cos(self):
        df = pd.DataFrame([
            ["q1", 'Obama speaks to the media in Illinois', "d1", 'The president greets the press in Chicago'],
            ["q1", 'Obama speaks to the media in Illinois', "d2", 'Holiday accommodation owners have been deluged with bookings over the last 48 hours.']]
            , columns=["qid", "query", "docno", "body"])
        pipe = GensimAverageSimilarity()
        rtr = pipe(df)
        print(rtr)
        self.assertTrue(df["score"][0] > df["score"][1])

    def test_gensim_wmd(self):
        df = pd.DataFrame([
            ["q1", 'Obama speaks to the media in Illinois', "d1", 'The president greets the press in Chicago'],
            ["q1", 'Obama speaks to the media in Illinois', "d2", 'Holiday accommodation owners have been deluged with bookings over the last 48 hours.']]
            , columns=["qid", "query", "docno", "body"])
        pipe = GensimWordMoverDistance()
        rtr = pipe(df)
        print(rtr)
        self.assertTrue(df["score"][0] > df["score"][1])
