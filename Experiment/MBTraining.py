# Build a class that trains the model 
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, SubsetRandomSampler
import random
from sklearn.model_selection import train_test_split
from transformers import get_scheduler, TrainingArguments
from tqdm import trange
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
import random
import wandb
import time
from accelerate import Accelerator


class ModelTraining:
    def __init__(self, category, model_name):
        self.max_epochs = 50 # Set it low for testing
        self.category = category
        self.model_name = model_name

    def training(self, model, optimizer, train_dataloader, dev_dataloader, device, accelerator, lr_scheduler):
        """
        Method for Training loop with Early Stopping based on the DevSet
        :param model: Model Loaded in the Wrapper
        :param optimizer: AdamW
        :param train_dataloader:
        :param dev_dataloader:
        :param device: GPU
        :param accelerator:
        :return: trained model
        """
        # SHOW A PROGRESS BAR
        num_training_steps = self.max_epochs * len(train_dataloader)
 
        progress_bar = tqdm(range(num_training_steps))

        # EARLY STOPPING CRITERIA
        last_loss = 100  # Source of the Early Stopping: https://pythonguides.com/pytorch-early-stopping/
        patience = 1
        trigger = 0

        for epoch in trange(self.max_epochs, desc='Epoch'):
            '''Need to implement Early Stopping Here'''
            print(f'Started Training Epoch {epoch}')
            # Training
            model.train()
            for step, batch in enumerate(train_dataloader, start = 1):
                with accelerator.accumulate(model):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if self.model_name == 'convbert' or self.model_name == 'electra':
                        outputs = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                    else:
                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                    loss = outputs.loss
                    accelerator.backward(loss)
                    #loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    progress_bar.update(1)
                    wandb.log({"batch": step, "time": time.time()})

            # Evaluation on DevSet
            model.eval()
            loss_lst, dev_predictions, dev_actuals = [], [], []
            for batch in dev_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    if self.model_name == 'convbert' or self.model_name == 'electra':
                        outputs = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                    else:
                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                logits = outputs.logits
                loss = outputs.loss
                loss_lst.append(loss)
                dev_actuals.extend(batch['labels'])
                dev_predictions.extend(torch.argmax(logits, dim=-1))

            current_loss = sum(loss_lst) / len(loss_lst)  # Dev Loss
            wandb.log({"loss": current_loss, "epoch": epoch})  # Logging of Dev Loss
            dev_predictions = torch.stack(dev_predictions).cpu()
            dev_actuals = torch.stack(dev_actuals).cpu()
            dev_report = classification_report(dev_actuals, dev_predictions, target_names=['non-biased', 'biased'],
                                               output_dict=True)
            wandb.log({"DEV f-1 score": dev_report['weighted avg']['f1-score'], "epoch": epoch})
            print('The current dev loss:', current_loss)
            if current_loss >= last_loss:
                trigger += 1
                print('trigger times:', trigger)

                if trigger >= patience:
                    print('Early stopping!\n Starting evaluation on test set.')
                    break

            else:
                print('trigger: 0')
                trigger = 0
            last_loss = current_loss
        return model

    def evaluate(self, model, test_dataloader, device, fold):
        """
        Evaluation model on the Test set
        Generates and saves a report on Scores
        :param model:
        :param test_dataloader:
        :param device:
        :return:
        """
        num_test_steps = len(test_dataloader)
        progress_bar = tqdm(range(num_test_steps))

        print(f'Start Evaluation')
        predictions, actuals, datasets = [], [], []
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                if self.model_name == 'convbert' or self.model_name == 'electra':
                    outputs = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                else:
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            logits = outputs.logits
            actuals.extend(batch['labels'])
            predictions.extend(torch.argmax(logits, dim=-1))
            datasets.extend(batch['dataset_id'])
            progress_bar.update(1)

        predictions = torch.stack(predictions).cpu()
        actuals = torch.stack(actuals).cpu()
        datasets = torch.stack(datasets).cpu()
        report = classification_report(actuals, predictions, target_names=['non-biased', 'biased'], output_dict=True)
        f1_score = report['weighted avg']['f1-score']
        wandb.log({"TEST f-1 score": f1_score, "fold": fold})
        df_report = pd.DataFrame(report)
        df_report.to_csv(f'./Results_new/{self.model_name}-{self.category}-fold-{fold}-report.csv')
        df_predictions = pd.DataFrame(data={'predictions': predictions, 'actuals': actuals, 'dataset_id': datasets})
        df_predictions.to_csv(f'./Results_new/{self.model_name}-{self.category}-fold-{fold}-predictions.csv')  # Save the predictions for later analysis
        return f1_score

    def main(self, fold, train_ids, val_ids, data, model, learning_rate, batch_size, gpu_no):
        """
        Main Method calling the training and evaluation, starting wandb, setting the GPU, and initializes e.g. Optimizer and Accelerator
        :param fold:
        :param train_ids:
        :param val_ids:
        :param data:
        :param model:
        :param learning_rate:
        :param batch_size:
        :param gpu_no:
        :return: None
        """
        
        print(f'Training Initialized for fold {fold}')
        # Initialize Weights & Biases
        wandb.login(key = "", relogin = True) # Insert Wandb key here
        wandb.init(project="MBG-Model-testing-" + str(self.category) + str(self.model_name), reinit=True)
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": 20,
            "batch_size": batch_size,
        }
        wandb.run.name = "Fold-" + str(fold)

        # Set the GPU
        device = torch.device(gpu_no)
        #print(device)

        # Create DEV and TEST Set from the K-folds Test Set
        # DEV Set used for early stopping criteria, the test set only for final evaluation
        dev_ids, test_ids = train_test_split(val_ids, test_size=0.75, train_size=0.25, random_state=42, shuffle=True)

        train_sampler = SubsetRandomSampler(train_ids)
        dev_sampler = SubsetRandomSampler(dev_ids)
        test_sampler = SubsetRandomSampler(test_ids)

        train_dataloader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
        dev_dataloader = DataLoader(data, batch_size=batch_size, sampler=dev_sampler)
        test_dataloader = DataLoader(data, batch_size=batch_size, sampler=test_sampler)
        model.to(device)  # Push model to GPU
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Initialize Optimizer
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing to save memory
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=10 * len(train_dataloader)
        )
        # Start Accelerator See https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one
        accelerator = Accelerator(fp16=True, device_placement=False, gradient_accumulation_steps=4)
        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

        # Model Training with Dev Evaluation for Early Stopping
        model = self.training(model, optimizer, train_dataloader, dev_dataloader, device, accelerator, lr_scheduler)

        # Evaluation on TestSet
        score = self.evaluate(model, test_dataloader, device, fold)
        #torch.cuda.empty_cache()
        wandb.finish()
        return score

