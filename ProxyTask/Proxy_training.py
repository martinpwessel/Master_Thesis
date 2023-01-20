import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler,random_split,SubsetRandomSampler
from transformers import get_scheduler, AdamW
from tqdm.auto import tqdm
from datasets import load_metric
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import copy
import time


class DataLoad:
    """
    Class to read in the BABE Dataset for the ProxyTask
    Uses the final labels SG1 for mbic and final labels SG2 for babe and combines them
    """
    @staticmethod
    def read_babe():
        df = pd.read_excel("../input/babe-media-bias-annotations-by-experts/data/final_labels_SG2.xlsx")
        lst = []
        for index, row in df.iterrows():
            if row['label_bias'] == "No agreement":
                pass
            else:
                sub_dict = {'text': row['text']}
                if row['label_bias'] == "Biased":
                    sub_dict['label'] = 1
                elif row['label_bias'] == "Non-biased":
                    sub_dict['label'] = 0
                lst.append(sub_dict)
        return lst


class ProxyTraining:
    """
    Class to perform the fine-tuning of the models on the babe dataset
    """
    def __init__(self, tokenizer, epochs: int):
        """
        :param tokenizer: transformer tokenizer
        :param model: huggingface model ForSequenceClassfication
        :param epochs: Number of epochs (generative?)
        """
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.data = DataLoad.read_babe()

    def proxy_tokenizing(self) -> list:
        # Tokenizing
        tokenized = []
        for i in range(len(self.data)):
            token = self.tokenizer(self.data[i]["text"], padding="max_length", truncation=True)
            token['labels'] = self.data[i]['label']
            tokenized.append(token)
        ten = []
        for i in range(len(tokenized)):
            x = {}
            for j in tokenized[i].keys():
                x[j] = torch.tensor(tokenized[i][j])
            ten.append(x)
        return ten

    def proxy_train_main(self, transformer_model, batch_size=8, gpu_no=0, model_name = "Unknown"):

        # Short check if GPU is available
        if torch.cuda.is_available():
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')

        data = self.proxy_tokenizing()  # loads tokenized data
        splits = KFold(n_splits=5, shuffle=True, random_state=42)  # k-fold k=5 as in the BABE paper
        overall_f1, info_lst = [], [] # list for all f1 scores of the folds
        for fold, (train_ids, val_ids) in enumerate(splits.split(np.arange(len(data)))):

            print('Fold {}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(train_ids)
            test_sampler = SubsetRandomSampler(val_ids)
            train_dataloader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
            test_dataloader = DataLoader(data, batch_size=batch_size, sampler=test_sampler)
            # assigns the available gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:" + str(gpu_no))
            else:
                device = torch.device("cpu")
            model = copy.deepcopy(transformer_model)
            optimizer = AdamW(model.parameters(), lr=5e-5, no_deprecation_warning=True)  # learning rate also set after the BABE paper
            num_epochs = self.epochs  # max number of epochs
            num_training_steps = num_epochs * len(train_dataloader)  # for the progress bar
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
            )
            model.to(device)
            progress_bar = tqdm(range(num_training_steps))  # displays progress bar
            # Early Stopping criteria (in case loss increases with epochs)
            test_loss_lst, test_f1_lst = [], []
            stopping = False  # stopping criteria if current test loss > previous test loss
            epoch = 0
            while epoch < num_epochs and stopping is False:
                # TRAINING
                loss_lst = []
                model.train()
                for batch in train_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss_lst.append(loss)
                    loss.backward()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                train_loss = sum(loss_lst) / len(loss_lst)  # avg train loss

                # EVALUATION
                metric_f1 = load_metric("f1")
                metric_acc = load_metric("accuracy")
                val_loss_lst = []
                model.eval()
                for batch in test_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    val_loss_lst.append(outputs.loss)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric_f1.add_batch(predictions=predictions, references=batch["labels"])
                    metric_acc.add_batch(predictions=predictions, references=batch["labels"])
                val_loss = sum(val_loss_lst)/len(val_loss_lst)  # avg test loss
                try:
                    if val_loss >= min(test_loss_lst):  # Stopping criteria gets invoked
                        stopping = True
                        print("Early Stopping Criteria")
                except ValueError:
                    pass
                test_loss_lst.append(val_loss)

                test_f1, test_acc = metric_f1.compute(), metric_acc.compute()
                test_f1_lst.append(test_f1['f1'])
                info = f"Fold: {fold} Epoch {epoch}: Avg train loss: {train_loss}, Avg test loss: {val_loss} Avg test acc: {test_acc}, Avg test f1: {test_f1}"
                print(info)
                info_lst.append(info)
                epoch += 1
            overall_f1.append(max(test_f1_lst))

        avg_f1 = sum(overall_f1) / len(overall_f1)
        print(f'Overall avg f1: {avg_f1}')
        info_lst.append(overall_f1)
        info_lst.append(avg_f1)
        with open(str(time.time()) + str(model_name)+ 'results.txt', 'w') as f:
            f.write(str(info_lst))
        return avg_f1
