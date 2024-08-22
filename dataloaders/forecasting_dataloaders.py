import torch
from torch.utils.data import Dataset, DataLoader

from data_processor.data_preprocessor import DataPreprocessor


class PricePercentageChangeDataset(Dataset):
    def __init__(self, dataframe, sequence_length=16):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the features and one-hot encoded labels.
            sequence_length (int): The length of the time-series sequences.
        """
        self.dataframe = dataframe
        self.sequence_length = sequence_length

        target_columns = ['price_change_percentage']
        feature_columns = list(self.dataframe.columns.difference(target_columns))      

        self.X = torch.tensor(self.dataframe[feature_columns].values, dtype=torch.float32)
        self.y = torch.tensor(self.dataframe[target_columns].values,dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.dataframe) - self.sequence_length

    def __getitem__(self, index): 
        if index >= self.sequence_length - 1:
            i_start = index - self.sequence_length + 1
            x = self.X[i_start:(index + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - index - 1, 1)
            x = self.X[0:(index + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[index]


def create_dataloaders(data_preprocessor: DataPreprocessor, sequence_length: int=8, batch_size: int=16):
    train_dataset = PricePercentageChangeDataset(data_preprocessor.train_data, sequence_length=sequence_length)
    validation_dataset = PricePercentageChangeDataset(data_preprocessor.validation_data, sequence_length=sequence_length)
    test_dataset = PricePercentageChangeDataset(data_preprocessor.test_data, sequence_length=sequence_length)   
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (train_loader, validation_loader, test_loader)