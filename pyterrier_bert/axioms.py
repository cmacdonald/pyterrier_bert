import pyterrier as pt
from pyterrier.transformer import EstimatorBase
import pandas as pd
import random
from . import add_label_column


class AxiomEstimator(EstimatorBase):

    body_attr = "body"
    parent : EstimatorBase

    def __init__(self, parent : EstimatorBase, body_attr="body", **kwargs):
        super().__init__()
        self.parent = parent
        self.body_attr = body_attr

    def transform(self, topics_or_res):
        return parent.transform(topics_or_res)
    
    def _choose(self):
        return random.choice([0,1])

    def _changeDF(self, resTrain, qrelsTrain):

        def _applyTFC1A(row : pd.DataFrame):
            query = row["query"].lower().split(r'\s+')
            row["docno"] = row["docno"] + "%r" 
            randomQt = random.choice(query)
            doc = row[self.body_attr].split(r'\s+')
            rand_pos = random.randint(0, len(doc)-1)
            doc.insert(rand_pos, randomQt)
            row[self.body_attr] = ' '.join(doc)
            return row

        def _applyTFC3(row : pd.DataFrame):            
            query = row["query"].lower().split(r'\s+')
            row["docno"] = row["docno"] + "%r" 
            randomQt = random.choice(query)
            doc = row[self.body_attr].split(r'\s+')
            if randomQt not in doc:
                rand_pos = random.randint(0, len(doc)-1)
                doc.insert(rand_pos, randomQt)
                row[self.body_attr] = ' '.join(doc)
                return row

        def _apply(row : pd.DataFrame):
            nonlocal self
            if row["label"] == 0:
                return
            if self._choose() == 0:
                return _applyTFC1A(row)
            else:
                return _applyTFC3(row)

        resTrain = add_label_column(resTrain, qrelsTrain)
        resTrain = pd.concat([resTrain, resTrain.apply(_apply, axis=1)])
        qrelsTrain = resTrain[["qid", "docno", "label"]]
        resTrain = resTrain.drop(columns=["label"])
        return (resTrain, qrelsTrain)

    def fit(self, resTrain, qrelsTrain,  resValid, qrelsValid):
        (resTrain, qrelsTrain) = self._changeDF(resTrain, qrelsTrain)
        return parent.fit(resTrain, qrelsTrain,  resValid, qrelsValid) 
