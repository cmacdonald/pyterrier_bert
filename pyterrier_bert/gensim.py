import gensim.downloader as api
from tqdm import tqdm
from pyterrier.transformer import TransformerBase
from gensim.utils import tokenize

class GensimWordMoverDistance(TransformerBase):
    
    def __init__(self, doc_attr="body", verbose=0):
        self.doc_attr = doc_attr
        self.wv = api.load("glove-wiki-gigaword-100")
        self.verbose = verbose

        # When using the wmdistance method, it is beneficial to normalize 
        # the word2vec vectors first, so they all have equal length.
        self.wv.init_sims(replace=True)
        

    # def preprocess(self, sentence):
    #     return [w for w in tokenize(sentence) if w not in self.stop_words]

    def transform(self, topics_res):
        # wmdistance is a _distance_, so we take the negative as our score

        lambda_row = lambda row: self.wv.wmdistance(list(tokenize(row["query"])), list(tokenize(row[self.doc_attr])))

        # could take a while, add a progress bar if asked to
        if self.verbose:
          tqdm.pandas()
          topics_res["score"] = -1 * topics_res.progress_apply(lambda_row, axis=1)
        else:
          topics_res["score"] = -1 * topics_res.apply(lambda_row, axis=1)

        return topics_res
