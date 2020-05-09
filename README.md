# BERT Estimators for Pyterrier

This package contains two integrations of BERT that can be used with PyTerrier:
1. [CEDR](https://github.com/Georgetown-IR-Lab/cedr), from the Georgetown IR Lab, by MacAveney et al.
2. [BERT4IR](https://github.com/ArthurCamara/Bert4IR), by Athur Camara, Unviersity of Delft.

## Installation

```
pip install  --upgrade git+https://github.com/cmacdonald/pyterrier_bert.git
```

## Usage

We show experiments using the TREC 2019 Deep Learning track. This assumes you already have a Terrier index that includes the contents of the document recorded as metadata.

This is the common setup using Pyterrier

```python

import pyterrier as pt
if not pt.started():
    pt.init(mem=8000)
trecdl = pt.datasets.get_dataset("trec-deep-learning-docs")

# your index must have the contents of the documents recorded as metadata
indexloc="/path/to/terrier/index/data.properties"

qrelsTest = trecdl.get_qrels("test")
qrelsTrain = trecdl.get_qrels("train")
#take 1000 topics for training
topicsTrain = trecdl.get_topics("train").head(1000)
#take 50 topics for validation
topicsValid = trecdl.get_topics("train").iloc[1001:1050]
#this one-liner removes topics without relevant documents
topicsTest = trecdl.get_topics("test").merge(qrelsTest[qrelsTest["label"] > 0][["qid"]])

# initial retrieval and QE baseline.
index = pt.IndexFactory.of(pt.IndexRef.of(indexloc))
DPH_br = pt.BatchRetrieve(index, controls={"wmodel" : "DPH"}, verbose=True, metadata=["docno", "body"])
DPH_br_qe = pt.BatchRetrieve(index, controls={"wmodel" : "DPH", "qe" : "on"}, verbose=True)

```

### CEDR Usage

```python
from pyterrier_bert.pyt_cedr import CEDRPipeline

cedrpipe = DPH_br >> CEDRPipeline(max_valid_rank=20)
# training, this uses validation set to apply early stopping
cedrpipe.fit(topicsTrain, qrelsTrain, topicsValid, qrelsTrain)

# testing performance
pt.pipelines.Experiment(topicsTest, 
                        [DPH_br, DPH_qe, cedrpipe],
                        ['map', 'ndcg'], 
                        qrelsTest, 
                        names=["DPH", "DPH + CEDR BERT"])
```


### BERT4IR Usage

```python
from pyterrier_bert.bert4ir import *

bertpipe = DPH_br >> BERTPipeline(max_valid_rank=20)
# training, this uses validation set to apply early stopping
bertpipe.fit(topicsTrain, qrelsTrain, topicsValid, qrelsTrain)

# testing performance
pt.pipelines.Experiment(topicsTest, 
                        [DPH_br, DPH_qe, bertpipe],
                        ['map', 'ndcg'], 
                        qrelsTest, 
                        names=["DPH", "DPH + QE", "DPH + BERT4IR"])
```