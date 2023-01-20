# Class that loads the saved balanced data
# Splits it into 5 seeded folds
# Distributes them onto five different GPUs
# Calls the training on each

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from multiprocessing import Process, Queue
import numpy as np
import torch
from ModelSpecifications import ModelSpecifications
from MBTraining import ModelTraining
import copy
import random
import time

class MBWrapper:

    def __init__(self, k, category, model_name, gpu, batch_size=256, model_length=512):
        self.k = k
        self.category = category
        self.model_name = model_name
        self.gpu = gpu
        self.batch_size = batch_size
        self.model_length = model_length

    def load_data(self, category):
        '''
        Loads the data from stored place and returns df
        :return:
        '''
        df = pd.read_pickle("/beegfs/zhukova/mbg_jupyter/input/balanced-datasets/" + str(self.category) + "-balanced_data.pkl")
        data = []
        for index, row in df.iterrows():
            data.append({'text': str(row['text']), 'label': row['label'], 'dataset_id': row['dataset_id']})
        return data

    def run_parallel(self, args: list):
        """
        Method to run multiple functions in parallel
        :param self:
        :param args: list of arguments that are inserted into the modeltrain function
        :return:
        """
        
        training = ModelTraining()
        process = []
        for arg in args:
            p = Process(target=training.training, args=arg) # with queue (queue,) + 
            p.start()
            process.append(p)
        for i in process:
            i.join()

    def check_gpu(self):
        """
        Only check to see if enough GPUs are available
        :return:
        """
        if torch.cuda.is_available():
            gpu_k = torch.cuda.device_count()
            print(f'There are {gpu_k} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(self.gpu))
        else:
            print('No GPU available, using the CPU instead.')

    def tokenize(self, tokenizer, data: list):
        """
        Tokenizer for now takes a list with dictionaries of the shape [{'text': 'sometext','label':0}, ...]
        :param tokenizer: Huggingface Tokenizer
        :param data: data list
        :return:
        """
        tokenized = []
        for i in range(len(data)):
            token = tokenizer(data[i]["text"], padding="max_length", truncation=True)
            token['labels'] = data[i]['label']
            token['dataset_id'] = int(data[i]['dataset_id'])  # Need to input the dataset number in the dataloader class
            tokenized.append(token)
        ten = []
        for i in range(len(tokenized)):
            x = {}
            for j in tokenized[i].keys():
                x[j] = torch.tensor(tokenized[i][j])
            ten.append(x)
        return ten

    def seed_all(self, seed_value):
        """
        Set SEEDS to make Model Training replicable
        :param seed_value:
        :return:
        """
        random.seed(seed_value) # Python
        np.random.seed(seed_value) # cpu vars
        torch.manual_seed(seed_value) # cpu  vars

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value) # gpu vars
            torch.backends.cudnn.deterministic = True  #needed
            torch.backends.cudnn.benchmark = False

    def main(self):
        """
        Main function where data is tokenized, split in Folds and distributed to GPUs
        Maximum number of GPUs 4
        :return:
        """
        self.check_gpu()  # Check GPU availability
        self.seed_all(42)  # Set Seed

        specs = ModelSpecifications()
        model, tokenizer, learning_rate = specs.modelspecifications(self.model_name, self.model_length)  # Get Model Specifications form ModelSpecifications.py
        print('Model Downloaded')

        print('Start Tokenizing')
        df = self.load_data(self.category)
        data = self.tokenize(tokenizer, df)
        print('Finish Tokenizing')
        
        # Split Data into Folds and Input Folds into ModelTraining Method from MBTraining.py
        splits = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        training = ModelTraining(self.category, self.model_name)
        score_lst, time_lst = [], []
        try:
            for fold, (train_ids, val_ids) in enumerate(splits.split(np.arange(len(data)), [ele['dataset_id'] for ele in data])):
                start = time.time()
                score = training.main(fold, train_ids, val_ids, data, copy.deepcopy(model), learning_rate, self.batch_size, "cuda:" + str(self.gpu))
                fold_time = time.time() - start
                print(f'fold_time: {fold_time}')
                score_lst.append(score)
                time_lst.append(fold_time)
            avg_f1 = sum(score_lst)/len(score_lst)
            print(f'Average weighted f1-score: {avg_f1}')
            print(f'Average weighted fold time: {sum(time_lst) / len(time_lst)}')
            with open('./Results_new/' + self.model_name + '-' + str(self.category) + '-fold-time.txt', 'w') as f:
                for line in time_lst:
                    f.write(f"{line}\n")
            score_lst.append(avg_f1)
            with open('./Results_new/' + self.model_name + '-' + str(self.category) + '-final-result.txt', 'w') as f:
                for line in score_lst:
                    f.write(f"{line}\n")
            return sum(score_lst)/len(score_lst)
        
        except Exception as e: 
            with open("ERROR"+ str(self.model_name)+".log", "w") as f:
                f.write('error:'+ str(e))
