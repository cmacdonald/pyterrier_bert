from transformers import AdamW, get_linear_schedule_with_warmup
from pyterrier import tqdm
import math
import torch
import pickle
from torch.utils.data import Dataset
from transformers import *
import os
import tempfile

def path(x):
    return os.path.join(".", x)

from pyterrier.transformer import EstimatorBase
from . import add_label_column

'''
Some of code in this file is based on that by Arthur Camara, found in
https://github.com/ArthurCamara/Bert4IR/blob/master/Train%20BERT.ipynb
'''

def train_neg_sampling(res_with_labels, neg_ratio):
    '''
        This implements the negative sampling found in Arthur's ECIR 2020 paper:
        We fine-tuned our BERT10 model with 10 negative samples for each positive
        sample from the training dataset, randomly picked from the top-100 retrieved
        from QL.
    '''
    import pandas as pd
    
    qid_groups = res_with_labels.groupby("qid")
    
    keeping_dfs = []
    for qid, queryDf in tqdm(qid_groups, desc="Negative sampling", total=qid_groups.ngroups, unit="q"):

        pos = queryDf[queryDf["label"] >= 1]
        neg = queryDf[queryDf["label"] < 1]
        num_pos = len(pos)
        num_neg = len(neg)
        num_neg_needed = num_pos * neg_ratio
        #print("qid %s num_pos %d num_neg %d num_neg needed %d" % (qid, num_pos, num_neg, num_neg_needed))

        if num_neg > num_neg_needed:
            neg = neg.sample(num_neg_needed)
        keeping_dfs.append(pos)  
        keeping_dfs.append(neg) 
        
    #we keep all positives
    rtr = pd.concat(keeping_dfs)
    # ensure labels are ints
    rtr["label"] = rtr["label"].astype(int)
    return rtr


    

class BERTPipeline(EstimatorBase):

    def __init__(self, *args,
        get_doc_fn=lambda row: row["body"], 
        max_train_rank=None, 
        max_valid_rank=None, 
        cache_threshold = None, 
        train_neg_sampling = None,
        **kwargs):
        super().__init__(*args, **kwargs)
        #error will happen below on token_type_ids if you change this to distilbert-base-uncased
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        self.max_train_rank = max_train_rank
        self.max_valid_rank = max_valid_rank
        self.get_doc_fn = get_doc_fn
        self.test_batch_size = 32
        self.cache_threshold = cache_threshold
        self.cache_dir = None
        self.train_neg_sampling = train_neg_sampling

    def make_dataset(self, res, *args, **kwargs):
        if self.cache_threshold is not None and len(res) > self.cache_threshold:
            cachedir = self.cache_dir
            cachedir = tempfile.mkdtemp()
            return CachingDFDataset(res, *args, directory=cachedir, **kwargs)
        return DFDataset(res, *args, **kwargs)

    def fit(self, tr, qrels_tr, va, qrels_va):
        print("Adding labels...")
        tr = add_label_column(tr, qrels_tr)
        va = add_label_column(va, qrels_va)
        
        if self.max_train_rank is not None:
            tr = tr[tr["rank"] < self.max_train_rank]
        if self.max_valid_rank is not None:
            va = va[va["rank"] < self.max_valid_rank]
        
        if self.train_neg_sampling is not None:
            assert self.train_neg_sampling > 0
            print("Negative sampling")
            tr = train_neg_sampling(tr, self.train_neg_sampling)

        tr_dataset = self.make_dataset(tr, self.tokenizer, split="train", get_doc_fn=self.get_doc_fn)
        assert len(tr_dataset) > 0
        va_dataset = self.make_dataset(va, self.tokenizer, split="valid", get_doc_fn=self.get_doc_fn)
        assert len(va_dataset) > 0
        self.model = train_bert4ir(self.model, tr_dataset, va_dataset)
        return self
        
    def transform(self, te):
        te_dataset = DFDataset(te, self.tokenizer, split="test", get_doc_fn=self.get_doc_fn)
        # we permit to adjust the batch size to allow better testing
        scores = bert4ir_score(self.model, te_dataset, batch_size=self.test_batch_size)
        assert len(scores) == len(te), "Expected %d scores, but got %d" % (len(tr), len(scores))
        te["score"] = scores
        return te

    def load(self, filename):
        import torch
        self.model.load_state_dict(torch.load(filename), strict=False)

    def save(self, filename):
        state = self.model.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, filename)

class DFDataset(Dataset):
    def __init__(self, df, tokenizer, *args, split, get_doc_fn, tokenizer_batch=8000):
        '''Initialize a Dataset object. 
        Arguments:
            samples: A list of samples. Each sample should be a tuple with (query_id, doc_id, <label>), where label is optional
            tokenizer: A tokenizer object from Hugging Face's Tokenizer lib. (need to implement encode_batch())
            split: a name for this dataset
            get_doc_fn: a function that maps a row into the text of the document 
            tokenizer_batch: How many samples to be tokenized at once by the tokenizer object.
        '''
        self.tokenizer = tokenizer
        tokenizer.padding_side = "right"
        print("Loading and tokenizing %s dataset of %d rows ..." % (split, len(df)))
        assert len(df) > 0
        self.labels_present = "label" in df.columns
        query_batch = []
        doc_batch = []
        sample_ids_batch = []
        labels_batch = []
        self.store={}
        self.processed_samples = 0
        number_of_batches = math.ceil(len(df) / tokenizer_batch)
        assert number_of_batches > 0        
        with tqdm(total=len(df), desc="Tokenizer input", unit="d") as batch_pbar:
            i=0
            for indx, row in df.iterrows():
                query_batch.append(row["query"])
                doc_batch.append(get_doc_fn(row))
                sample_ids_batch.append(row["qid"] + "_" + row["docno"])
                if self.labels_present:
                    labels_batch.append(row["label"])
                else:
                    # we dont have a label, but lets append 0, to get rid of if elsewhere.
                    labels_batch.append(0)
                if len(query_batch) == tokenizer_batch or i == len(df) - 1:
                    self._tokenize_and_dump_batch(doc_batch, query_batch, labels_batch, sample_ids_batch)
                    query_batch = []
                    doc_batch = []
                    sample_ids_batch = []
                    labels_batch = []
                i += 1
                batch_pbar.update()
        

    def _tokenize_and_dump_batch(self, doc_batch, query_batch, labels_batch,
                                 sample_ids_batch):
        '''tokenizes and dumps the samples in the current batch
        It also store the positions from the current file into the samples_offset_dict.
        '''
        # Use the tokenizer object
        batch_tokens = self.tokenizer.batch_encode_plus(list(zip(query_batch, doc_batch)), max_length=512, pad_to_max_length=True)
        for idx, (sample_id, tokens) in enumerate(zip(sample_ids_batch, batch_tokens['input_ids'])):
            assert len(tokens) == 512
            # BERT supports up to 512 tokens. batch_encode_plus will enforce this for us.
            # the original implementation had code to truncate long documents with [SEP]
            # or pad short documents with [0] 
            segment_ids = batch_tokens['token_type_ids'][idx]
            self._store(sample_id, tokens, segment_ids, labels_batch[idx])
            self.processed_samples += 1

    def _store(self, sample_id, token_ids, segment_ids, label):
        self.store[self.processed_samples] = (sample_id, token_ids, segment_ids, label)

    def __getitem__(self, idx):
        '''Returns a sample with index idx
        DistilBERT does not take into account segment_ids. (indicator if the token comes from the query or the document) 
        However, for the sake of completeness, we are including it here, together with the attention mask
        position_ids, with the positional encoder, is not needed. It's created for you inside the model.
        '''
        _, input_ids, token_type_ids, label = self.store[idx]
        input_mask = [1] * 512
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long),
                torch.tensor([label], dtype=torch.long))

    def __len__(self):
        return len(self.store)


# This is the caching Dataset class. This dataset does not yet check if the files are there already
class CachingDFDataset(DFDataset):
    def __init__(self, *args, split, directory=None, **kwargs):
        '''Initialize a Dataset object. 
        Arguments:
            samples: A list of samples. Each sample should be a tuple with (query_id, doc_id, <label>), where label is optional
            tokenizer: A tokenizer object from Hugging Face's Tokenizer lib. (need to implement encode_batch())
            split: a name for this dataset
            tokenizer_batch: How many samples to be tokenized at once by the tokenizer object.
        '''
        self.samples_offset_dict = dict()
        self.index_dict = dict()
        assert directory is not None
        self.directory = directory
        self.samples_file = open(os.path.join(self.directory, f"{split}_msmarco_samples.tsv"),'w',encoding="utf-8")        
        super().__init__(*args, split=split, **kwargs)
        # Dump files in disk, so we don't need to go over it again.
        self.samples_file.close()
        pickle.dump(self.index_dict, open(os.path.join(self.directory, f"{split}_msmarco_index.pkl"), 'wb'))
        pickle.dump(self.samples_offset_dict, open(os.path.join(self.directory, f"{split}_msmarco_offset.pkl"), 'wb'))
        self.split = split
        
    def _store(self, sample_id, token_ids, segment_ids, label):
        # How far in the file are we? This is where we need to go to find the documents later.
        file_location = self.samples_file.tell()
        self.samples_file.write(f"{sample_id}\t{token_ids}\t{segment_ids}\t{label}\n")
        self.samples_offset_dict[sample_id] = file_location
        self.index_dict[self.processed_samples] = sample_id

   

    def __getitem__(self, idx):
        '''Returns a sample with index idx
        DistilBERT does not take into account segment_ids. (indicator if the token comes from the query or the document) 
        However, for the sake of completness, we are including it here, together with the attention mask
        position_ids, with the positional encoder, is not needed. It's created for you inside the model.
        '''
        if isinstance(idx, int):
            idx = self.index_dict[idx]
        with open(os.path.join(self.directory, f"{self.split}_msmarco_samples.tsv"), 'r', encoding="utf-8") as inf:
            inf.seek(self.samples_offset_dict[idx])
            line = inf.readline().split("\t")
            try:
                sample_id = line[0]
                input_ids = eval(line[1])
                token_type_ids = eval(line[2])
                input_mask = [1] * 512
            except:
                print(line, idx)
                raise IndexError
            # If it's a training dataset, we also have a label tag.
            if self.labels_present:
                label = int(line[3])
                return (torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(input_mask, dtype=torch.long),
                        torch.tensor(token_type_ids, dtype=torch.long),
                        torch.tensor([label], dtype=torch.long))
            return (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(input_mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples_offset_dict)


from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from tokenizers import BertWordPieceTokenizer


GPUS_TO_USE = [2,4,5,6,7] # If you have multiple GPUs, pick the ones you want to use.
number_of_cpus = 1 #was 24 # Number of CPUS to use when loading your dataset.
n_epochs = 2 # How may passes over the whole dataset to complete
weight_decay = 0.0 # Some papers define a weight decay, meaning, the weights on some layers will decay slower overtime. By default, we don't do this.
lr = 0.00005 # Learning rate for the fine-tunning.
warmup_proportion = 0.1 # Percentage of training steps to perform before we start to decrease the learning rate.
steps_to_print = 1000 # How many steps to wait before printing loss
steps_to_eval = 2000 # How many steps to wait before running an eval step



# tokenizer = BertWordPieceTokenizer("/ssd2/arthur/bert-axioms/tokenizer/bert-base-uncased-vocab.txt", lowercase=True)

def train_bert4ir(model, train_dataset, dev_dataset):

    if torch.cuda.is_available():
        # Asssign the model to GPUs, specifying to use Data parallelism.
        model = torch.nn.DataParallel(model, device_ids=GPUS_TO_USE)
        parallel = len(GPUS_TO_USE)
        # The main model should be on the first GPU
        device = torch.device(f"cuda:{GPUS_TO_USE[0]}") 
        model.to(device)
        # For a 1080Ti, 16 samples fit on a GPU comfortably. So, the train batch size will be 16*the number of GPUS
        train_batch_size = parallel * 16
        print(f"running on {parallel} GPUS, on {train_batch_size}-sized batches")
    else:
        print("Are you sure about it? We will try to run this in CPU, but it's a BAD idea...")
        device = torch.device("cpu")
        train_batch_size = 16
        model.to(device)
        parallel = number_of_cpus

    # A data loader is a nice device for generating batches for you easily.
    # It receives any object that implements __getitem__(self, idx) and __len__(self)
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=number_of_cpus,shuffle=True)
    dev_data_loader = DataLoader(dev_dataset, batch_size=32, num_workers=number_of_cpus,shuffle=True)

    #how many optimization steps to run, given the NUMBER OF BATCHES. (The len of the dataloader is the number of batches).
    num_train_optimization_steps = len(train_data_loader) * n_epochs

    #which layers will not have a linear weigth decay when training
    no_decay = ['bias', 'LayerNorm.weight']

    #all parameters to be optimized by our fine tunning.
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any( nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any( nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    #We use the AdamW optmizer here.
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8) 

    # How many steps to wait before we start to decrease the learning rate
    warmup_steps = num_train_optimization_steps * warmup_proportion 
    # A scheduler to take care of the above.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
    print(f"*********Total optmization steps: {num_train_optimization_steps}*********")

    import warnings
    import numpy as np
    import datetime

    from sklearn.metrics import f1_score, average_precision_score, accuracy_score, roc_auc_score

    global_step = 0 # Number of steps performed so far
    tr_loss = 0.0 # Training loss
    model.zero_grad() # Initialize gradients to 0

    for _ in tqdm(range(n_epochs), desc="Epochs"):
        for step, batch in tqdm(enumerate(train_data_loader), desc="Batches", total=len(train_data_loader)):
            model.train()
            # get the batch inpute
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'labels': batch[3].to(device)
            }
            # Run through the network.
            
            with warnings.catch_warnings():
                # There is a very annoying warning here when we are using multiple GPUS,
                # As described here: https://github.com/huggingface/transformers/issues/852.
                # We can safely ignore this.
                warnings.simplefilter("ignore")
                outputs = model(**inputs)
            loss = outputs[0]

            loss = loss.sum()/parallel # Average over all GPUs/CPUs.
           
            # Backward pass on the network
            loss.backward()
            tr_loss += loss.item()
            # Clipping gradients. Avoud gradient explosion, if the gradient is too large.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Run the optimizer with the gradients
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if step % steps_to_print == 0:
                # Logits is the actual output from the network. 
                # This is the probability of being relevant or not.
                # You can check its shape (Should be a vector sized 2) with logits.shape()
                logits = outputs[1]
                # Send the logits to the CPU and in numpy form. Easier to check what is going on.
                preds = logits.detach().cpu().numpy()
                
                tqdm.write(f"Training loss: {loss.item()} Learning Rate: {scheduler.get_last_lr()[0]}")
            global_step += 1
            
            # Run an evluation step over the eval dataset. Let's see how we are doing.
            if global_step%steps_to_eval == 0:
                eval_loss = 0.0
                nb_eval_steps = 0
                preds = None
                out_label_ids = None
                for batch in tqdm(dev_data_loader, desc="Valid batch"):
                    model.eval()
                    with torch.no_grad(): # Avoid upgrading gradients here
                        inputs = {'input_ids': batch[0].to(device),
                        'attention_mask': batch[1].to(device),
                        'labels': batch[3].to(device)}
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            outputs = model(**inputs)
                        tmp_eval_loss, logits = outputs[:2] # Logits is the actual output. Probabilities between 0 and 1.
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_steps += 1
                        # Concatenate all outputs to evaluate in the end.
                        if preds is None:
                            preds = logits.detach().cpu().numpy() # PRedictions into numpy mode
                            out_label_ids = inputs['labels'].detach().cpu().numpy().flatten() # Labels assigned by model
                        else:
                            batch_predictions = logits.detach().cpu().numpy()
                            preds = np.append(preds, batch_predictions, axis=0)
                            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy().flatten(), axis=0)
                    eval_loss = eval_loss / nb_eval_steps
                results = {}
                results["ROC Dev"] = roc_auc_score(out_label_ids, preds[:, 1])
                preds = np.argmax(preds, axis=1)
                results["Accuracy Dev"] = accuracy_score(out_label_ids, preds)
                results["F1 Dev"] = f1_score(out_label_ids, preds)
                results["AP Dev"] = average_precision_score(out_label_ids, preds)
                tqdm.write("***** Eval results *****")
                for key in sorted(results.keys()):
                    tqdm.write(f"  {key} = {str(results[key])}")
                save_model(model, global_step)
    model_to_save = save_model(model, global_step)
    return model_to_save

def save_model(model, global_step):
    output_dir = path(f"checkpoints/checkpoint-{global_step}")
    if not os.path.isdir(output_dir):
        os.makedirs(path(output_dir))
    print(f"Saving model checkpoint to {output_dir}")
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    return model_to_save

def bert4ir_score(model, dataset, batch_size=32):
    import warnings
    import numpy as np
    if torch.cuda.is_available():
        # Assign the model to GPUs, specifying to use Data parallelism.
        model = torch.nn.DataParallel(model, device_ids=GPUS_TO_USE)
        parallel = len(GPUS_TO_USE)
        # The main model should be on the first GPU
        device = torch.device(f"cuda:{GPUS_TO_USE[0]}") 
        model.to(device)
        # For a 1080Ti, 16 samples fit on a GPU comfortably. So, the train batch size will be 16*the number of GPUS
        train_batch_size = parallel * 16
        print(f"running on {parallel} GPUS, on {train_batch_size}-sized batches")
    else:
        print("Are you sure about it? We will try to run this in CPU, but it's a BAD idea...")
        device = torch.device("cpu")
        train_batch_size = 16
        model.to(device)
        parallel = number_of_cpus

    preds = None
    nb_eval_steps = 0
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=number_of_cpus,shuffle=False)
    for batch in tqdm(data_loader, desc="Scoring batch"):
        model.eval()
        
        with torch.no_grad(): # Avoid upgrading gradients here
            inputs = {'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device),
            #'labels': batch[3].to(device)
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = model(**inputs)
            #print(outputs)
            logits = outputs[:2][0]
            #logits = outputs[0] # Logits is the actual output. Probabilities between 0 and 1.
            #print(logits)
            # we take the second column(?)
            logits = logits[:,0]
            nb_eval_steps += 1
            # Concatenate all outputs to one big score array.
            if preds is None:
                preds = logits.detach().cpu().numpy().flatten() # Predictions into numpy mode
                #print(preds.shape)
            else:
                batch_predictions = logits.detach().cpu().numpy().flatten()
                preds = np.append(preds, batch_predictions, axis=0)
    #print(preds)
    #print(preds.shape)
    return preds
