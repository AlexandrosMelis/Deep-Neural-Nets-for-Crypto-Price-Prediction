from matplotlib.pylab import f
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import joblib
import math
import datetime
from config import PathConfig

class DataPreprocessor:
    """
    Data preprocessing class to clean, normalize, and encode the raw data.
    Usage:
    ```python
        preprocessor = DataPreprocessor(raw_data)
        preprocessor.add_features(add_date_features=True, add_indicator_features=True)
        preprocessor.clean_data()
        preprocessor.create_labels()
        preprocessor.split_data(datetime(2022, 3, 1), datetime(2022, 10, 1))
        preprocessor.apply_scalling()
        preprocessor.save_splitted_data()
    ```
    """

    def __init__(self, data: pd.DataFrame):
        self.raw_data = data
        self.preprocessed_data = None
        self.scaler = RobustScaler()
        # self.label_encoder = LabelEncoder()
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def add_features(self, add_date_features: bool = False, add_indicator_features: bool = False) -> pd.DataFrame:
        if self.raw_data is None:
            raise ValueError("Raw data doesn't exist!")
        if add_date_features:
            self.raw_data = self.add_date_features(self.raw_data)
        if add_indicator_features:
            self.raw_data = self.add_technical_indicators(self.raw_data)
        self.raw_data.dropna(inplace=True)
        return self.raw_data

    def clean_data(self, set_time_index: bool = True) -> pd.DataFrame:
        self.preprocessed_data = self.raw_data.copy()
        if set_time_index:
            self.preprocessed_data['time'] = pd.to_datetime(self.preprocessed_data['time'])
            self.preprocessed_data.sort_values('time', inplace=True)
            self.preprocessed_data.set_index('time', inplace=True)
        return self.preprocessed_data
    
    # 1st approach to create labels
    def create_labels(self, horizon=24, buy_threshold=0.04, sell_threshold=-0.04) -> pd.DataFrame:
        # Calculate the future return
        self.preprocessed_data['future_return'] = self.preprocessed_data['close'].shift(-horizon) / self.preprocessed_data['close'] - 1

        # Define labels based on future returns
        conditions = [
            (self.preprocessed_data['future_return'] > buy_threshold),
            (self.preprocessed_data['future_return'] < sell_threshold)
        ]
        choices = ['buy', 'sell']
        self.preprocessed_data['label'] = np.select(conditions, choices, default='hold')
        self.preprocessed_data.drop(['future_return'], axis=1, inplace=True)
        self.preprocessed_data['target'] = self.label_encoder.fit_transform(self.preprocessed_data['label'])
        return self.preprocessed_data
    
    # 2nd approach create percentage change target
    def create_percent_change_target(self) -> pd.DataFrame:
        self.preprocessed_data['price_change_percentage'] = self.preprocessed_data['close'].pct_change() * 100
        self.preprocessed_data = self.preprocessed_data.dropna(subset=['price_change_percentage'])
        self.preprocessed_data['price_change'] = self.preprocessed_data['price_change_percentage'].apply(lambda x: 'increase' if x > 0 else 'decrease')
        return self.preprocessed_data


    # def normalize_datasets(self,):
    #     """
    #     Normalize the feature columns of train, validation, and test datasets using Min-Max scaling.
        
    #     Args:
    #         train_df (pd.DataFrame): Training dataset containing feature and target columns.
    #         val_df (pd.DataFrame): Validation dataset containing feature and target columns.
    #         test_df (pd.DataFrame): Test dataset containing feature and target columns.
        
    #     Returns:
    #         tuple: Normalized train, validation, and test dataframes.
    #     """    
    #     if self.train_data is None or self.validation_data is None or self.test_data is None:
    #         raise ValueError("Data must be splitted correctly before normalization.")

    #     target_columns = ['target', 'label']
    #     if 'target' not in self.train_data.columns:
    #         raise ValueError("Target column not found in the dataset.")
    #     columns_to_normalize = list(self.train_data.columns.difference(target_columns))  
    #     print(f"Columns to normalize: {columns_to_normalize}")
    #     print(f"Target columns: {target_columns}")

    #     self.scaler.fit(self.train_data[columns_to_normalize])
        
    #     self.train_data.loc[:, columns_to_normalize] = self.scaler.transform(self.train_data[columns_to_normalize])
    #     self.validation_data.loc[:, columns_to_normalize] = self.scaler.transform(self.validation_data[columns_to_normalize])
    #     self.test_data.loc[:, columns_to_normalize] = self.scaler.transform(self.test_data[columns_to_normalize])
    #     print("Data normalized successfully!")	

    def apply_scalling(self):

        if self.train_data is None or self.validation_data is None or self.test_data is None:
            raise ValueError("Data must be splitted correctly before normalization.")

        self.scaler.fit(self.train_data)
        
        self.train_data.loc[:, :] = self.scaler.transform(self.train_data)
        self.validation_data.loc[:, :]  = self.scaler.transform(self.validation_data)
        self.test_data.loc[:, :]  = self.scaler.transform(self.test_data)
        print("Data scaled!")

    def save_splitted_data(self, filepath: str = PathConfig.base_data_path.value):
        if self.train_data is None or self.validation_data is None or self.test_data is None:
            raise ValueError("Data must be splitted before saving.")
        # delete categorical label column
        if 'label' in self.train_data.columns:
            self.train_data.drop(columns=['label'], axis=1, inplace=True)
            self.validation_data.drop(columns=['label'], axis=1, inplace=True)
            self.test_data.drop(columns=['label'], axis=1, inplace=True)
        # save data
        self.train_data.to_csv(join(filepath, 'train_data.csv'), index=False)
        self.validation_data.to_csv(join(filepath, 'validation_data.csv'), index=False)
        self.test_data.to_csv(join(filepath, 'test_data.csv'), index=False)
        if self.scaler:
            joblib.dump(self.scaler, join(filepath, 'scaler.pkl'))
        # if self.label_encoder:
        #     joblib.dump(self.label_encoder, join(filepath, 'encoder.pkl'))
        print("Preprocessed data and scaler saved successfully!")

    def load_preprocessed_data(self, filepath: str = PathConfig.base_data_path.value):
        self.train_data = pd.read_csv(join(filepath, 'train_data.csv'))
        self.validation_data = pd.read_csv(join(filepath, 'validation_data.csv'))
        self.test_data = pd.read_csv(join(filepath, 'test_data.csv'))
        self.scaler = joblib.load(join(filepath, 'scaler.pkl'))
        # self.label_encoders = joblib.load(join(filepath, 'encoder.pkl'))
        print("Preprocessed data and transformation objects loaded successfully!")
    
    # date-time features
    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['time'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['hour'] = df['date'].dt.hour
        df.drop(['date'], axis=1, inplace=True)

        df = self.create_trigonometric_columns(df)
        return df
    
    def create_trigonometric_columns(self, df) -> pd.DataFrame:
        df['year_sin'] = df['year'].apply(lambda x: math.sin(2*math.pi*x/2023))
        df['year_cos'] = df['year'].apply(lambda x: math.cos(2*math.pi*x/2023))
        df['month_sin'] = df['month'].apply(lambda x: math.sin(2*math.pi*x/12))
        df['month_cos'] = df['month'].apply(lambda x: math.cos(2*math.pi*x/12))
        df['day_sin'] = df['day'].apply(lambda x: math.sin(2*math.pi*x/31))
        df['day_cos'] = df['day'].apply(lambda x: math.cos(2*math.pi*x/31))
        df['hour_sin'] = df['hour'].apply(lambda x: math.sin(2*math.pi*x/24))
        df['hour_cos'] = df['hour'].apply(lambda x: math.cos(2*math.pi*x/24))
        df = df.drop(['year'], axis=1)
        df = df.drop(['month'], axis=1)
        df = df.drop(['day'], axis=1)
        df = df.drop(['week_of_year'], axis=1)
        df = df.drop(['hour'], axis=1)    
        return df
    
    # techincal indicator features
    def add_technical_indicators(self, df):
        """
        Adds technical indicators to the dataframe: SMA, EMA, RSI, MACD, and Bollinger Bands.

        Args:
        df (pd.DataFrame): Dataframe containing 'open', 'high', 'low', 'close', 'volume'.

        Returns:
        pd.DataFrame: Dataframe with added technical indicator columns.
        """

        # Simple Moving Averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()

        # Exponential Moving Averages
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()

        # RSI (Relative Strength Index)
        # change = df['close'].diff(1)
        # gain = change.where(change > 0, 0)
        # loss = -change.where(change < 0, 0)
        # avg_gain = gain.rolling(window=14).mean()
        # avg_loss = loss.rolling(window=14).mean()
        # rs = avg_gain / avg_loss
        # df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        # ema12 = df['close'].ewm(span=12, adjust=False).mean()
        # ema26 = df['close'].ewm(span=26, adjust=False).mean()
        # df['macd'] = ema12 - ema26
        # df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        rstd = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = sma20 + 2 * rstd
        df['bollinger_lower'] = sma20 - 2 * rstd
        return df
    
    def split_data(self, split_date_1: datetime, split_date_2: datetime):
        """
        Split the data into training, validation, and testing sets based on provided dates.
        **Usage**:  ```preprocessor.split_data(datetime(2022, 3, 1), datetime(2022, 10, 1))```
        """
        self.train_data = self.preprocessed_data.loc[self.preprocessed_data.index < split_date_1]
        self.validation_data = self.preprocessed_data.loc[(split_date_1 <= self.preprocessed_data.index) & (self.preprocessed_data.index <= split_date_2)]
        self.test_data = self.preprocessed_data.loc[self.preprocessed_data.index > split_date_2]
        
        print("Training set:", round((len(self.train_data) / len(self.preprocessed_data)), 2), '%', "- train shape -> ", self.train_data.shape)
        print("Validation set:", round((len(self.validation_data) / len(self.preprocessed_data)), 2), '%', "- valid shape -> ", self.validation_data.shape)
        print("Test set:", round((len(self.test_data) / len(self.preprocessed_data)), 2), '%', "- test shape -> ", self.test_data.shape)
