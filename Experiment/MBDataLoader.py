"""
Class to
- combine the datasets for each category,
- and take a balanced sample from each category.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MBDataLoader:
    def __init__(self, category):
        self.category = category
        self.category_ids = {0: ['003', '010', '026', '049', '075', '035', '009', '038'],
                             1: ['066', '072'],
                             2: ['019', '076'],
                             3: ['040', '075', '087', '092'],
                             4: ['075', '105', '106', '107'],
                             5: ['075', '034', '113', '110'],
                             6: ['072', '012', '025'],
                             7: ['049', '066', '029'],
                             8: ['010'],
                             9: ['003','009','010','012','019','025','026','029','035','038','040','049','072','075','076','087','092','105','106','107','110']}[category] # For testing only
#                             9: ['003','009','010','012','019','025','026','029','034','035','038','040','049','066','072','075','076','087','092','105','106','107','110']}[category] # For testing only

    def load_data(self, files: list, file_path: str):
        """
        Loads the data from the internal file structure for now, should change here for the automatic downloading
        Assigns new unique id to every datapoint: Dataset_id-Prior_id
        """
        df = pd.DataFrame(columns=['id', 'text', 'label'])
        for file in files:
            path = os.path.join(file_path, file[1])
            df_sub = pd.DataFrame()


#            if file[0] == '066':
#                df_file = pd.read_csv(path, skiprows= [0], names= ['id', 'text', 'label', 'category'])
#            else:
            df_file = pd.read_csv(path)
            df_file['nr'] = str(file[0])
#            try:
            df_file['new_id'] = df_file['nr'] + '-' + df_file['id'].astype(str)
#            except KeyError: # In case there's no unique ID, takes the csv id
#                df_file['new_id'] = df_file['nr'] + '-' + df_file['Unnamed: 0'].astype(str)
            df_sub['id'], df_sub['text'], df_sub['label'] = df_file['new_id'], df_file['text'], df_file['label']
            df_sub['dataset_id'] = df_file['nr']
            df = pd.concat([df, df_sub], axis=0)
        return df

    def get_data(self):
        """
        Loads the data from the local file path,
        sorts files by category ids and combines them to one df
        """

        ds_raw_path = os.path.join("/beegfs/zhukova/mbg_jupyter/Preprocessed_Datasets")
#        ds_raw_path = os.path.join(os.path.dirname(os.getcwd()) + "/Preprocessed_Datasets")
        files = os.listdir(ds_raw_path)
        category_files = []
        for i in self.category_ids:
            for file in files:
                if str(i) in file:
                    category_files.append((i, file))
        df = self.load_data(category_files, ds_raw_path)
        return df
    
    def check_balance(self, df):
        """
        checks the balance of the dataset and returns the amount of biased/unbiased labels
        """
        biased = len(df[df['label'] == 1])/len(df)
        non_biased = len(df[df['label'] == 0])/len(df)
        print(f'The Category consists of {biased * 100}% biased and {non_biased * 100}% non-biased labels.')
        return len(df[df['label'] == 1]), len(df[df['label'] == 0])

    def load_balanced_sample(self):
        """
        Draws a random sample based on the smaller available label.
        """
        # Had to exclude all 2 labels for now. But can add them again later if we want to use them
        df = self.remove_duplicates(self.get_data())
        df_wo2 = df[df['label'] != 2]
        len_biased, len_unbiased = self.check_balance(df_wo2)
        k = min(len_biased, len_unbiased)
        grouped = df_wo2.groupby('label')
        df_balanced = grouped.apply(lambda x: x.sample(n=k, random_state=42))
        df_balanced.to_pickle("/beegfs/zhukova/mbg_jupyter/input/balanced-datasets/" + str(self.category) + "-balanced_data.pkl")
        return df_balanced

    def load_full_sample(self, df):
        """
        Just returns a random sample of all the data
        """
        return df.sample(n=len(df), random_state=42)

    def remove_duplicates(self, df):
        """
        Removes exact duplicates from df['text'].
        Should be called on combined dataframe for each category.
        """
        len1 = len(df)
        df = df.drop_duplicates('text', keep='first')
        len2 = len(df)
        
        if len1 > len2:
            print(f'Removed {len1 - len2} duplicates')

        return df

    def text_length_analysis(self, df):
        """
        Counts the length of the texts and plots a histogram
        :param df:
        :return:
        """
        df['count'] = df.text.str.split(' ').str.len()
        l_512 = len(df[df['count'] > 512])
        l_256 = len(df[df['count'] > 256])
        print(f'There are {l_512} texts longer than 512. That\'s {l_512 / len(df)}%.')
        print(f'There are {l_256} texts longer than 256. That\'s {l_256 / len(df)}%.')
        counts = list(df['count'])
        print(f'The 99% percentile is at {np.nanpercentile(counts,99)}')
        bin_values = np.arange(start=0, stop=1000, step=20)
        _ = plt.hist(df['count'], bins=bin_values)
        plt.title('Histogram showing the text length')
        plt.xlabel('Word Count')
        plt.show()
        return counts


if __name__ == "__main__":
    if len(sys.argv) == 1:
        data = MBDataLoader(0)
    else:
        data = MBDataLoader(int(sys.argv[1]))
#    df = data.get_data()
#    df = df.remove_duplicates() # should possibly be run during initialisation instead
    df_balanced = data.load_balanced_sample()
    print(len(df_balanced))
    data.check_balance(df_balanced)
    sys.exit(0)
