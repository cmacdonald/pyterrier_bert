from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import math
import torch
import pickle
from torch.utils.data import Dataset
from transformers import *
import os

def path(x):
    return os.path.join(".", x)

from pyterrier.transformer import EstimatorBase

'''
The code in this class is based on that by Arthur Camara, found in
https://github.com/ArthurCamara/Bert4IR/blob/master/Train%20BERT.ipynb
'''

class BERTPipeline(EstimatorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    def fit(self, tr, qrels_tr, va, qrels_va): 
        #TODO apply cutoffs for validation and training
        if qrels_tr is not None:
            tr = tr.merge(qrels_tr, on=["qid", "docno"], how="left")
        if qrels_va is not None:
            va = va.merge(qrels_va, on=["qid", "docno"], how="left")
        tr_dataset = DFDataset(tr, self.tokenizer, "train")
        va_dataset = DFDataset(tr, self.tokenizer, "valid")
        self.model = train_bert4ir(self.model, tr_dataset, va_dataset)
        return self
        
    def transform(self, tr):
        te_dataset = DFDataset(tr, self.tokenizer, "test")
        scores = bert4ir_score(self.model, te_dataset)
        tr["score"] = scores
        return tr

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
    def __init__(self, df, tokenizer, split, tokenizer_batch=8000):
        '''Initialize a Dataset object. 
        Arguments:
            samples: A list of samples. Each sample should be a tuple with (query_id, doc_id, <label>), where label is optional
            tokenizer: A tokenizer object from Hugging Face's Tokenizer lib. (need to implement encode_batch())
            split: a name for this dataset
            tokenizer_batch: How many samples to be tokenized at once by the tokenizer object.
        '''
        self.tokenizer = tokenizer
        print("Loading and tokenizing dataset of %d rows ..." % len(df))
        self.labels_present = "label" in df.columns
        query_batch = []
        doc_batch = []
        sample_ids_batch = []
        labels_batch = []
        self.store={}
        self.processed_samples = 0
        number_of_batches = math.ceil(len(df) // tokenizer_batch)
        with tqdm(total=number_of_batches, desc="Tokenizer batches") as batch_pbar:
            for i, row in df.iterrows():
                query_batch.append(row["query"])
                doc_batch.append(row["text"])
                sample_ids_batch.append(row["qid"] + "_" + row["docno"])
                if self.labels_present:
                    labels_batch.append(row["label"])
                else:
                    # we dont have a label, but lets append 0, to get rid of if elsewhere.
                    labels_batch.append(0)
                if len(query_batch) == tokenizer_batch or i == len(df) - 1:
                    self._tokenize_and_dump_batch(doc_batch, query_batch, labels_batch, sample_ids_batch)
                    batch_pbar.update()
                    query_batch = []
                    doc_batch = []
                    sample_ids_batch = []
                    labels_batch = []
        

    def _tokenize_and_dump_batch(self, doc_batch, query_batch, labels_batch,
                                 sample_ids_batch):
        '''tokenizes and dumps the samples in the current batch
        It also store the positions from the current file into the samples_offset_dict.
        '''
        # Use the tokenizer object
        #tokens = self.tokenizer.encode_batch(list(zip(query_batch, doc_batch)))
        tokens = self.tokenizer.batch_encode_plus(list(zip(query_batch, doc_batch)))
        for idx, (sample_id, token) in enumerate(zip(sample_ids_batch, tokens['input_ids'])):
            #BERT supports up to 512 tokens. If we have more than that, we need to remove some tokens from the document
            if len(token) >= 512:
                token_ids = token[:511]
                token_ids.append(tokenizer.token_to_id("[SEP]"))
                segment_ids = tokens['token_type_ids'][idx][:512]
            # With less tokens, we need to "pad" the vectors up to 512.
            else:
                padding = [0] * (512 - len(token))
                token_ids = token + padding
                segment_ids = tokens['token_type_ids'][idx] + padding
            self._store(sample_id, token_ids, segment_ids, labels_batch[idx])
            self.processed_samples += 1

    def _store(self, sample_id, token_ids, segment_ids, label):
        self.store[self.processed_samples] = (sample_id, token_ids, segment_ids, label)

    def __getitem__(self, idx):
        '''Returns a sample with index idx
        DistilBERT does not take into account segment_ids. (indicator if the token comes from the query or the document) 
        However, for the sake of completness, we are including it here, together with the attention mask
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
    def __init__(self, df, tokenizer, split, tokenizer_batch=8000):
        '''Initialize a Dataset object. 
        Arguments:
            samples: A list of samples. Each sample should be a tuple with (query_id, doc_id, <label>), where label is optional
            tokenizer: A tokenizer object from Hugging Face's Tokenizer lib. (need to implement encode_batch())
            split: a name for this dataset
            tokenizer_batch: How many samples to be tokenized at once by the tokenizer object.
        '''
        self.samples_offset_dict = dict()
        self.index_dict = dict()
        self.samples_file = open(path(f"{split}_msmarco_samples.tsv"),'w',encoding="utf-8")        
        super().__init__(df, tokenizer, split, tokenizer_batch=8000)
        # Dump files in disk, so we don't need to go over it again.
        self.samples_file.close()
        pickle.dump(self.index_dict, open(path(f"{split}_msmarco_index.pkl"), 'wb'))
        pickle.dump(self.samples_offset_dict, open(path(f"{split}_msmarco_offset.pkl"), 'wb'))
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
        with open(path(f"{self.split}_msmarco_samples.tsv"), 'r', encoding="utf-8") as inf:
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
    # It receives any object that implementes __getitem__(self, idx) and __len__(self)
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
            
            # Run an evluation step over the eval dataset. Let's see how we are going.
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
                results["Acuracy Dev"] = accuracy_score(out_label_ids, preds)
                results["F1 Dev"] = f1_score(out_label_ids, preds)
                results["AP Dev"] = average_precision_score(out_label_ids, preds)
                tqdm.write("***** Eval results *****")
                for key in sorted(results.keys()):
                    tqdm.write(f"  {key} = {str(results[key])}")
    output_dir = path(f"checkpoints/checkpoint-{global_step}")
    if not os.path.isdir(output_dir):
        os.makedirs(path(output_dir))
#             print(f"Saving model checkpoint to {output_dir}")
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    return model_to_save

def bert4ir_score(model, dataset):
    import warnings
    import numpy as np
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

    preds = None
    nb_eval_steps = 0
    data_loader = DataLoader(dataset, batch_size=32, num_workers=number_of_cpus,shuffle=False)
    for batch in tqdm(data_loader, desc="Valid batch"):
        model.eval()
        
        with torch.no_grad(): # Avoid upgrading gradients here
            inputs = {'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device),
            #'labels': batch[3].to(device)
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = model(**inputs)
            print(outputs)
            logits = outputs[:2][0] # Logits is the actual output. Probabilities between 0 and 1.
            print(logits)
            nb_eval_steps += 1
            # Concatenate all outputs to evaluate in the end.
            if preds is None:
                preds = logits.detach().cpu().numpy() # PRedictions into numpy mode
            else:
                batch_predictions = logits.detach().cpu().numpy()
                preds = np.append(preds, batch_predictions, axis=0)
    return preds[0]
