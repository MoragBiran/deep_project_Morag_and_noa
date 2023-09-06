import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


def predata(args_config, data_plot=False):
    # Load dataset
    data = pd.read_csv(args_config.data_name)
    #Drop y - take only the x's
    features = data.drop(['is_myopia'], axis=1)
    #For the nirmol - we calculated each column its own and therefore axis equals zero
    mean_values = features.mean(axis=0)
    std_values = features.std(axis=0)

    # check unbalance data
    #Shows me how many samples (number) there are for each class - here 0 and 1 how many there are for myopia
    class_counts = data['is_myopia'].value_counts()
    print(class_counts)
    if data_plot:
        plt.bar(class_counts.index, class_counts.values)
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(rotation='vertical')
        plt.show()
    shuffled_data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    # Define the train-test split ratio
    train_ratio = args_config.split_r
    split_index = int(train_ratio * len(shuffled_data))  # Calculate the split index
    # Split the data into train and test sets
    train_df = shuffled_data[:split_index]
    test_df = shuffled_data[split_index:]
    class_counts_train = train_df['is_myopia'].value_counts()
    #A way to deal with an-blas data - give more
    # weight to what has fewer samples and less weight to what has more samples
    class_weights = torch.FloatTensor([
        len(train_df) / (2 * class_counts_train[0]),
        len(train_df) / (2 * class_counts_train[1])
    ]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    return train_df, test_df, class_weights, mean_values, std_values


class MyopiaDataset(Dataset):
    def __init__(self, data, mean, std):
        self.labels = data['is_myopia']
        #Throwing away the column of the label that I will have all the data without
        data.drop('is_myopia', axis=1, inplace=True)
        self.features = data
        self.mean = np.array(mean.values)
        self.std = np.array(std.values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features.iloc[idx].values
        y = self.labels.iloc[idx]
        #לבדוק האם להוריד מביא תוצאה יותר טובה
        # x = (x - self.mean) / self.std
        x = torch.Tensor(x)
        y = torch.Tensor([y])
        return x, y

