from os.path import join
import pandas as pd
import torch
from torch import nn, optim
from config import PathConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

from data_processor.data_preprocessor import DataPreprocessor
from dataloaders.label_dataloaders import create_dataloaders
from pytorch_trainer.base_trainer import BaseTrainer


class CausalCnnModelTrainer(BaseTrainer):

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

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            train_loss = 0
            for inputs, targets in self.train_loader:
                inputs = inputs.permute(0, 2, 1)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            self.metrics['train_loss'].append(train_loss)
            
            val_loss, val_metrics = self.evaluate(self.val_loader)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['accuracy'].append(val_metrics['accuracy'])
            self.metrics['precision'].append(val_metrics['precision'])
            self.metrics['recall'].append(val_metrics['recall'])
            self.metrics['f1'].append(val_metrics['f1'])

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}, '
                  f'Val Prec: {val_metrics["precision"]:.4f}, Val Rec: {val_metrics["recall"]:.4f}, '
                  f'Val F1: {val_metrics["f1"]:.4f}')
        
        self.plot_metrics()
        self.save_model()

    def evaluate(self, data_loader):
        self.model.eval()
        val_loss = 0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.permute(0, 2, 1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        val_loss /= len(data_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        
        return val_loss, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def save_metrics(self):
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(join(PathConfig.experiment_results_path.value, 'causal_convolution_models', f'{self.model_name}_metrics.csv'), index=False)
        self.plot_metrics()
    
    def save_model(self):
        torch.save(self.model.state_dict(), join(PathConfig.model_path.value, f"{self.model_name}.pth"))

    def plot_metrics(self):
        epochs = range(1, self.num_epochs + 1)
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.metrics['train_loss'], label='Train Loss')
        plt.plot(epochs, self.metrics['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.metrics['accuracy'], label='Accuracy')
        plt.plot(epochs, self.metrics['precision'], label='Precision')
        plt.plot(epochs, self.metrics['recall'], label='Recall')
        plt.plot(epochs, self.metrics['f1'], label='F1 Score')
        plt.title('Evaluation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()

        plt.tight_layout()
        plt.savefig(join(PathConfig.metric_plots_path.value, 'causal_convolution_models', f'{self.model_name}_{self.index_iter}_plot.png'))
        plt.show()


    @classmethod
    def hyperparameter_tuning(cls, model, data_preprocessor: DataPreprocessor, parameter_grid: dict, num_epochs: int, model_name: str = "causal_cnn"):
        validation_results = []
        test_results = []

        # Iterate through all combinations of hyperparameters
        for exp_index, params in tqdm(enumerate(itertools.product(*parameter_grid.values()), start=1)):
            param_dict = dict(zip(parameter_grid.keys(), params))

            print(f"Testing combination: {param_dict}")

            # Create datasets and dataloaders
            train_loader, validation_loader, test_loader = create_dataloaders(data_preprocessor=data_preprocessor, sequence_length=param_dict['sequence_length'], batch_size=param_dict['batch_size'])

            # Define the model, criterion, optimizer
            model = model(
                in_channels=param_dict['in_channels'],
                num_classes=param_dict['num_classes'],
                kernel_size=param_dict['kernel_size'],
                num_filters=param_dict['num_filters'],
                num_layers=param_dict['num_layers']
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=param_dict['learning_rate'])

            # Train the model
            trainer = cls(model_name, model, train_loader, validation_loader, test_loader, criterion, optimizer, num_epochs, experiment_index=exp_index)
            trainer.train()

            # Save the validation results
            validation_result = {
                'index': exp_index,
                **param_dict,
                'train_loss': trainer.metrics['train_loss'][-1],
                'val_loss': trainer.metrics['val_loss'][-1],
                'accuracy': trainer.metrics['accuracy'][-1],
                'precision': trainer.metrics['precision'][-1],
                'recall': trainer.metrics['recall'][-1],
                'f1': trainer.metrics['f1'][-1]
            }
            validation_results.append(validation_result)

            # Evaluate the model on the test set
            test_loss, test_metrics = trainer.evaluate(test_loader)

            # Save the test results
            test_result = {
                'index': exp_index,
                **param_dict,
                'test_loss': test_loss,
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1']
            }
            test_results.append(test_result)    
            # Print the test results
            print(f"Test Results - Loss: {test_loss:.4f}, Acc: {test_metrics['accuracy']:.4f}, "
                  f"Prec: {test_metrics['precision']:.4f}, Rec: {test_metrics['recall']:.4f}, "
                  f"F1: {test_metrics['f1']:.4f}")

        # Save results
        val_results_df = pd.DataFrame(validation_results)
        val_results_df.to_csv(join(PathConfig.experiment_results_path.value, 'causal_convolution_models', f'{model_name}_validation_metrics.csv'), index=False)
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(join(PathConfig.experiment_results_path.value, 'causal_convolution_models', f'{model_name}__test_metrics.csv'), index=False)

        # Find the best combination based on validation chosen metric
        best_result = val_results_df.loc[val_results_df['accuracy'].idxmax()]
        print(f"Best Parameters - index: {best_result['index']}")
        print(f"Best Accuracy: {best_result['accuracy']:.4f}")

        return best_result