import os
import pandas as pd
import re

pwd = os.getcwd()
results = os.listdir(os.path.join(pwd, 'Results'))


def add_df(df, lst):
    """
    Add a list to an existing df.
    List has to be in order of the df columns.

    Returns: concatenated dataframe.
    """
    df_tmp = pd.DataFrame(columns=df.columns, data=[lst])
    df = pd.concat([df, df_tmp], ignore_index=True)
    return df


def file_to_list(filepath):
    """
    Takes a file name (full path) as input.

    Returns: float value for each line, removing trailing linebreaks (`\n`).
    """
    try:
        with open(filepath, "r") as f:
            lst = [float(line[:-2]) for line in f]
            return lst
    except IOError:
        print("Error during file processing")


def compute_predictions(df_predictions):
    """
    Taskes raw df report as input.
    For each model and dataset, computes the accuracy, precision, recall and f1 scores.
    """

    # TODO: category

    columns = ['model', 'dataset_id', 'testset_size', 'true_positives', 'true_negatives', 'false_positives',
               'false_negatives', 'accuracy', 'precision', 'recall', 'f1']
    lst = []

    dfs_dataset_id = {}

    for dataset_id, b in df_predictions.groupby('dataset_id'):
        dfs_dataset_id[dataset_id] = b

    for dataset_id in dfs_dataset_id:
        dfs_model_tmp = {}
        df_dataset = dfs_dataset_id.get(dataset_id)
        for df_model, b in df_dataset.groupby('model'):
            dfs_model_tmp[df_model] = b

        for model in dfs_model_tmp:
            df = dfs_model_tmp.get(model)

            testset_size = len(df)
            true_positives = len(df[(df['predictions'] == 1) & (df['actuals'] == 1)])
            false_positives = len(df[(df['predictions'] == 1) & (df['actuals'] == 0)])
            false_negatives = len(df[(df['predictions'] == 0) & (df['actuals'] == 1)])
            true_negatives = len(df[(df['predictions'] == 0) & (df['actuals'] == 0)])

            accuracy = 1 - len(df[(df['predictions'] != df['actuals'])]) / len(df)
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                #                print(f'{dataset_id} and {model}.\nTrue Positives: {true_positives},\nFalse Positives: {false_positives},\nFalse Negatives: {false_negatives}')
                f1 = float("nan")
            lst.append(
                [model, dataset_id, testset_size, true_positives, true_negatives, false_positives, false_negatives,
                 accuracy, precision, recall, f1])

    return pd.DataFrame(columns=columns, data=lst)


df_results = pd.DataFrame(columns=['model', 'category', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'avg'])
df_time = pd.DataFrame(columns=['model', 'category', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5'])
# df_reports =
df_predictions = pd.DataFrame(columns=['model', 'category', 'fold', 'predictions', 'actuals', 'dataset_id'])


for filename in results:
    file = filename.split('.')[0]
    if str.__contains__(file, '-final-result'):
        if str.__contains__(file, 'roberta-twitter'):
            (model1,model2, category, _, _) = file.split('-')
            model = model1 + '-' + model2
        else:
            (model, category, _, _) = file.split('-')
        #        results = pd.read_csv(os.path.join(pwd, 'results', filename))

        lst = file_to_list(os.path.join(pwd,'Results', filename))
        lst.insert(0, model)
        lst.insert(1, category)

        df_results = add_df(df_results, lst)

    elif str.__contains__(file, '-fold-time'):
        if str.__contains__(file, 'roberta-twitter'):
            (model1,model2, category, _, _) = file.split('-')
            model = model1 + '-' + model2
        else:
            (model, category, _, _) = file.split('-')

        lst = file_to_list(os.path.join(pwd,'Results', filename))
        lst.insert(0, model)
        lst.insert(1, category)

        df_time = add_df(df_time, lst)

    elif str.__contains__(file, '-report'):
        #(model, category, _, fold, _) = file.split('-')

        pass
        # TODO?

    elif str.__contains__(file, '-predictions'):
        if str.__contains__(file, 'roberta-twitter'):
            (model1,model2, category,  _, fold, _) = file.split('-')
            model = model1 + '-' + model2
        else:
            (model, category,_, fold, _) = file.split('-')
        df = pd.read_csv(os.path.join(pwd,'Results', filename))
        df['model'] = model
        df['category'] = category
        df['fold'] = fold
        df_predictions = pd.concat([df_predictions, df], ignore_index=True)
df_predictions.to_csv('Raw_Predictions.csv')
df_predictions = compute_predictions(df_predictions)

df_results.to_csv("Results_Table.csv")
df_predictions.to_csv("Prediction_Table.csv")
