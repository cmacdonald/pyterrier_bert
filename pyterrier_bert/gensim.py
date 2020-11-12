import gensim.downloader as api
from pyterrier import tqdm
from pyterrier.transformer import TransformerBase
from gensim.utils import tokenize
import numpy as np
from numpy import dot
from numpy.linalg import norm

class GensimBase(TransformerBase):

  def __init__(self, doc_attr="body", modelname="glove-wiki-gigaword-100", verbose=0):
    self.doc_attr = doc_attr
    self.wv = api.load(modelname)
    self.verbose = verbose
    self.oov = np.random.rand(self.wv.vector_size)

  def transform(self, topics_res):
    pass


class GensimAverageSimilarity(GensimBase):
  '''
    A similarity score that computes the cosine similarity on the 
    average of the document and query term embedding vectors
  '''
  def transform(self, topics_res):

    def lambda_row(row):
      q = list(tokenize(row["query"]))
      d = list(tokenize(row[self.doc_attr]))

      qs = np.array([ self.wv[t] if t in self.wv else self.oov for t in q ])
      ds = np.array([ self.wv[t] if t in self.wv else self.oov for t in d ])
      qs_avg = np.average(qs, axis=0)
      ds_avg = np.average(ds, axis=0)   

      return qs_avg @ qs_avg.T/(norm(qs_avg)*norm(ds_avg)) 

    # could take a while, add a progress bar if asked to
    if self.verbose:
      tqdm.pandas()
      topics_res["score"] = topics_res.progress_apply(lambda_row, axis=1)
    else:
      topics_res["score"] = topics_res.apply(lambda_row, axis=1)

    return topics_res
    

class GensimWordMoverDistance(GensimBase):
  '''
    A similarity score that uses WordMoversDistance
  '''
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # When using the wmdistance method, it is beneficial to normalize 
    # the word2vec vectors first, so they all have equal length.
    self.wv.init_sims(replace=True)

  def transform(self, topics_res):
    # wmdistance is a _distance_, so we take the negative as our "similarity" score

    lambda_row = lambda row: self.wv.wmdistance(list(tokenize(row["query"])), list(tokenize(row[self.doc_attr])))

    # could take a while, add a progress bar if asked to
    if self.verbose:
      tqdm.pandas()
      topics_res["score"] = -1 * topics_res.progress_apply(lambda_row, axis=1)
    else:
      topics_res["score"] = -1 * topics_res.apply(lambda_row, axis=1)

    return topics_res
