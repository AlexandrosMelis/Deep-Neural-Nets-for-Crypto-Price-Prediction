from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model_name, model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, experiment_index=0):
        self.model_name = model_name
        self.index_iter = experiment_index
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def evaluate(self, data_loader):
        pass

    @abstractmethod
    def save_metrics(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def plot_metrics(self):
        pass

    @classmethod
    @abstractmethod
    def hyperparameter_tuning(cls, model, data_preprocessor, parameter_grid, num_epochs, model_name):
        pass