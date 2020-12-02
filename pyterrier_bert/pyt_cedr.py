from pyterrier.transformer import EstimatorBase

from . import add_label_column
from pyterrier.model import add_ranks

class CEDRPipeline(EstimatorBase):
    
    
    
    def __init__(self, modelname='vanilla_bert', doc_attr="body", max_train_rank=None, max_valid_rank=None, model_out_dir=None):
            self.modelname = modelname
            self.doc_attr = doc_attr
            self.max_train_rank = max_train_rank
            self.max_valid_rank = max_valid_rank
            self.model_out_dir = model_out_dir
            
    def _make_cedr_dataset(self, table):
        docs={}
        queries={}
        for index, row in table.iterrows():
            queries[row['qid']] = row['query']
            docs[row['docno']] = row[self.doc_attr]
        dataset=(queries, docs)
        return dataset

    def _make_cedr_run(self, run_df, qrels_df=None):
        final_DF = add_label_column(run_df, qrels_df)
        from collections import defaultdict
        run=defaultdict(dict)
        for index, row in final_DF.iterrows():
            run[row['qid']][row['docno']] = float(1)
        return run

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        import torch
        from cedr import train
        # load the requested model
        self.model = train.MODEL_MAP[self.modelname]()
        # check if gpu enabled
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        # now load model
        self.model.load(filename)
        
    def fit(self, tr, qrelsTrain, va, qrelsValid):
        
        if self.max_train_rank is not None:
            tr = tr[tr["rank"] < self.max_train_rank]
        if self.max_valid_rank is not None:
            va = va[va["rank"] < self.max_valid_rank]
        
        train_run = self._make_cedr_run(tr, qrelsTrain)
        valid_run = self._make_cedr_run(va, qrelsValid)
        dataset = self._make_cedr_dataset(tr.append(va))

        import torch
        from cedr import train
        import pyterrier as pt
        train.tqdm = pt.tqdm
        
        # load the requested model
        model = train.MODEL_MAP[self.modelname]()
        # check if gpu enabled
        model = model.cuda() if torch.cuda.is_available() else model

        train.VALIDATION_METRIC="ndcg"
        self.model, _  = train.main( 
            model,
            dataset,
            train_run,
            pt.Utils.convert_qrels_to_dict(qrelsTrain),
            valid_run,
            pt.Utils.convert_qrels_to_dict(qrelsValid),
            self.model_out_dir
        )
        return self
    
    def transform(self, queries_and_docs):
        
        from cedr import train
        import pyterrier as pt
        train.tqdm = pt.tqdm
        import pandas as pd
        
        test_run = self._make_cedr_run(queries_and_docs, None)
        dataset = self._make_cedr_dataset(queries_and_docs)
        
        
        run_values = train.run_model(self.model, dataset, test_run, desc="CEDR")
        run_df_rows = []
        for q, docs in run_values.items():
            for d in docs:
                run_df_rows.append([q, d, docs[d]])
        run_df = pd.DataFrame(run_df_rows, columns=["qid", "docno", "score"])
        if "score" in queries_and_docs.columns:
            queries_and_docs = queries_and_docs.drop(columns="score")
        
        final_df = run_df.merge(queries_and_docs, on=["qid", "docno"])
        final_df = add_ranks(final_df)
        return final_df
