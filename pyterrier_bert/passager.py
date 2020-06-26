from pyterrier.transformer import TransformerBase
import more_itertools
from collections import defaultdict
import re
from tqdm import tqdm
import pandas as pd

def slidingWindow(sequence, winSize, step):
    return [x for x in list(more_itertools.windowed(sequence,n=winSize, step=step)) if x[-1] is not None]

class DePassager(TransformerBase):

    def __init__(self, agg="max", **kwargs):
        super().__init__(**kwargs)
        self.agg = agg

    def transform(self, topics_and_res):
        scoredict=defaultdict(lambda: defaultdict(dict))
        lastqid=None
        qids=[]
        for i, row in topics_and_res.iterrows():
            qid = row["qid"]
            if qid != lastqid:
                qids.append(qid)
                lastqid = qid
                
            docno, passage = row["docno"].split("%p")
            #print("%s %s" % (docno, passage))
            scoredict[qid][docno][int(passage)] = row["score"]
        rows=[]
        #print(scoredict)
        for qid in qids:
            for docno in scoredict[qid]:
                #print(docno)
                if self.agg == 'first':
                    first_passage_id = min( scoredict[qid][docno].keys() )
                    score = scoredict[qid][docno][first_passage_id]
                if self.agg == 'max':
                    score = max( scoredict[qid][docno].values() )
                if self.agg == 'mean':
                    score = sum( scoredict[qid][docno].values() ) / len(scoredict[qid][docno])
                rows.append([qid, docno, score])
        rtr = pd.DataFrame(rows, columns=["qid", "docno", "score"])
        #add the queries back
        queries = topics_and_res.groupby(["qid"]).count().reset_index()[["qid", "query"]]
        rtr = rtr.merge(queries, on=["qid"])
        return rtr

class MaxPassage(DePassager):
    def __init__(self, **kwargs):
        kwargs["agg"] = "max"
        super().__init__(**kwargs)

class FirstPassage(DePassager):
    def __init__(self, **kwargs):
        kwargs["agg"] = "first"
        super().__init__(**kwargs)

class MeanPassage(DePassager):
    def __init__(self, **kwargs):
        kwargs["agg"] = "mean"
        super().__init__(**kwargs)


class SlidingWindowPassager(TransformerBase):

    def __init__(self, text_attr='body', title_attr='title', passage_length=150, passage_stride=75, join=' ', **kwargs):
        super().__init__(**kwargs)
        self.text_attr=text_attr
        self.title_attr=title_attr
        self.passage_length = passage_length
        self.passage_stride= passage_stride
        self.join = join

    def transform(self, topics_and_res):
        return self.applyPassaging(topics_and_res, labels="label" in topics_and_res.columns)

    def applyPassaging(self, df, labels=True):
        newRows=[]
        labelCount=defaultdict(int)
        p = re.compile(r"\s+")
        currentQid=None
        rank=0
        copy_columns=[]
        for col in ["score", "rank"]:
            if col in df.columns:
                copy_columns.append(col)

        if len(df) == 0:
            return pd.DataFrame(columns=['qid', 'query', 'docno', self.text_attr, 'score', 'rank'])
        with tqdm('passsaging', total=len(df), ncols=80, desc='passaging', leave=False) as pbar:
            for index, row in df.iterrows():
                pbar.update(1)
                qid = row['qid']
                if currentQid is None or currentQid != qid:
                    rank=0
                    currentQid = qid
                rank+=1
                toks = p.split(row[self.text_attr])
                if len(toks) < self.passage_length:
                    newRow = row.drop(labels=[self.title_attr])
                    newRow['docno'] = row['docno'] + "%p0"
                    
                    newRow[self.text_attr] = str(row[self.title_attr]) + self.join + ' '.join(toks)
                    if labels:
                        labelCount[row['label']] += 1
                    for col in copy_columns:
                        newRow[col] = row[col]
                    newRows.append(newRow)
                else:
                    passageCount=0
                    for i, passage in enumerate( slidingWindow(toks, self.passage_length, self.passage_stride)):
                        newRow = row.drop(labels=[self.title_attr])
                        newRow['docno'] = row['docno'] + "%p" + str(i)
                        newRow[self.text_attr] = str(row[self.title_attr]) + self.join + ' '.join(passage)
                        for col in copy_columns:
                            newRow[col] = row[col]
                        if labels:
                            labelCount[row['label']] += 1
                        newRows.append(newRow)
                        passageCount+=1
        newDF = pd.DataFrame(newRows)
        newDF['query'].fillna('',inplace=True)
        newDF[self.text_attr].fillna('',inplace=True)
        newDF['qid'].fillna('',inplace=True)
        newDF.reset_index(inplace=True,drop=True)
        return newDF


