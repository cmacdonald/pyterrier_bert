# BERT Estimators for Pyterrier

This package contains two integrations of BERT that can be used with [PyTerrier](https://github.com/terrier-org/pyterrier):
1. [CEDR](https://github.com/Georgetown-IR-Lab/cedr), from the Georgetown IR Lab, by MacAveney et al.
2. [BERT4IR](https://github.com/ArthurCamara/Bert4IR), by Arthur Camara, University of Delft.

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
#this one-liner removes topics from the test set that do not have relevant documents
topicsTest = trecdl.get_topics("test").merge(qrelsTest[qrelsTest["label"] > 0][["qid"]].drop_duplicates())

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

### Passaging

Documents can be too long for BERT. It is possible to split them down in to passages, for instance by applying a sliding window (of a given size, in terms of number of tokens), which advances by a given number of tokens (called stride) each time. To ensure that there is some possibly-relevant content in each passage, Dai & Callan [3] proposed to prepend the title of the document to each passage.

You can apply passaging in the same fashion as Dai & Callan by adding two additional transformers to the pipeline:
```python
from pyterrier_bert.passager import SlidingWindowPassager, MaxPassage

DPH_br = pt.BatchRetrieve(index, controls={"wmodel" : "DPH"}, verbose=True, metadata=["docno", "body", "title"])

pipe_max_passage = DPH_br >> \
    SlidingWindowPassager(passage_length=150, passage_stride=175) >> \
    CEDRPipeline(max_valid_rank=20) >> MaxPassage()
```

You an even apply the passaging aggregation as different features, which could then be comined using learning-to-rank, such as xgBoost:
```python

from pyterrier_bert.passager import SlidingWindowPassager, MaxPassage, FirstPassage, MeanPassage
pipe_passage_features = DPH_br \
    >> SlidingWindowPassager() \
    >> CEDRPipeline(max_valid_rank=20) \
    >> ( MaxPassage() ** FirstPassage() ** MeanPassage() )

```

Both CEDRPipeline and BERTPipeline support passaging in this form.

# Credits

 - Craig Macdonald, University of Glasgow
 - Alberto Ueda, UFMG

Code in BERTPipeline is adapted from that by Arthur Camara, University of Delft.

# References

1. Sean MacAvaney, Andrew Yates, Arman Cohan, Nazli Goharian: CEDR: Contextualized Embeddings for Document Ranking. SIGIR 2019: 1101-1104
2. Zhuyun Dai, Jamie Callan: Deeper Text Understanding for IR with Contextual Neural Language Modeling. SIGIR 2019: 985-988
3. Arthur Câmara, Claudia Hauff: Diagnosing BERT with Retrieval Heuristics. ECIR (1) 2020: 605-618
