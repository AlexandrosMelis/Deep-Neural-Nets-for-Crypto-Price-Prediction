from cgi import test
from os.path import join
from pyexpat import model
from unittest.mock import Base
import pandas as pd
import torch
from torch import nn, optim
from config import PathConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib

from data_processor.data_preprocessor import DataPreprocessor
from dataloaders.forecasting_dataloaders import create_dataloaders
from pytorch_trainer.base_trainer import BaseTrainer


class RnnModelTrainer(BaseTrainer):

    def __init__(self, model_name, model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, experiment_index=0):
        self.model_name = model_name
        self.index_iter = experiment_index
        self.model = model
        self.trained_model = None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'mae': [],
            'mse': [],
            'r2': []
        }

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            train_loss = 0
            for inputs, targets in self.train_loader:                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                targets = targets.squeeze()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            self.metrics['train_loss'].append(train_loss)
            
            val_loss, val_metrics = self.evaluate(self.val_loader)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['mae'].append(val_metrics['mae'])
            self.metrics['mse'].append(val_metrics['mse'])
            self.metrics['r2'].append(val_metrics['r2'])

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val MAE: {val_metrics["mae"]:.4f}, '
                  f'Val MSE: {val_metrics["mse"]:.4f}, Val R2: {val_metrics["r2"]:.4f}')
        
        # self.plot_metrics()
        self.save_model()
        

    def evaluate(self, data_loader):
        self.model.eval()
        val_loss = 0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).squeeze()
                targets = targets.squeeze()

                # Ensure outputs and targets are 1-dimensional
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)

                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
        
        val_loss /= len(data_loader)
        mae = mean_absolute_error(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        return val_loss, {'mae': mae, 'mse': mse, 'r2': r2}
    
    def plot_metrics(self):
        epochs = range(1, self.num_epochs + 1)
        
        plt.figure(figsize=(12, 8))

        # Plotting Train and Validation Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.metrics['train_loss'], label='Train Loss')
        plt.plot(epochs, self.metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and Validation Loss')
        
        # Plotting MAE, MSE, and R²
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.metrics['mae'], label='MAE')
        plt.plot(epochs, self.metrics['mse'], label='MSE')
        plt.plot(epochs, self.metrics['r2'], label='R²')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.title('MAE, MSE, and R²')

        plt.tight_layout()
        plt.savefig(join(PathConfig.metric_plots_path.value, 'rnn_models', f'{self.model_name}_{self.index_iter}_plot.png'))
        plt.show()
    
    def save_model(self,):
        torch.save(self.model.state_dict(), join(PathConfig.model_path.value, f"{self.model_name}.pth"))
        print(f"Model instance saved successfully in file: {self.model_name}.pth")

    def load_model(self, model_class, input_size, hidden_size, num_layers, output_size, dropout):
        path = join(PathConfig.model_path.value, f"{model_class.__name__}.pth")
        model = model_class(input_size, hidden_size, num_layers, output_size, dropout)
        model.load_state_dict(torch.load(path))
        model.eval()
        self.trained_model = model
        print("Trained model loaded successfully!")
        return model
    
    def save_metrics(self):
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(join(PathConfig.experiment_results_path.value, 'rnn_models', f'{self.model_name}_metrics.csv'), index=False)
        print(f"Metrics saved successfully in file: f'{self.model_name}_metrics.csv'")

    @classmethod
    def hyperparameter_tuning(cls, model_class, data_preprocessor: DataPreprocessor, parameter_grid: dict, num_epochs: int):
        validation_results = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for exp_index, params in tqdm(enumerate(itertools.product(*parameter_grid.values()), start=1)):
            param_dict = dict(zip(parameter_grid.keys(), params))

            print(f"Testing combination: {param_dict}")

            train_loader, validation_loader, test_loader = create_dataloaders(
                data_preprocessor=data_preprocessor,
                sequence_length=param_dict['sequence_length'],
                batch_size=param_dict['batch_size']
            )

            print(model_class.__name__)
            model = model_class(
                input_size=param_dict['input_size'],
                hidden_size=param_dict['hidden_size'],
                num_layers=param_dict['num_layers'],
                output_size=1,
                dropout=param_dict['dropout']
            ).to(device)


            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=param_dict['learning_rate'])

            trainer = cls(model_class.__name__, model, train_loader, validation_loader, test_loader, criterion, optimizer, num_epochs, experiment_index=exp_index)
            trainer.train()

            validation_result = {
                'index': exp_index,
                **param_dict,
                'train_loss': trainer.metrics['train_loss'][-1],
                'val_loss': trainer.metrics['val_loss'][-1],
                'mae': trainer.metrics['mae'][-1],
                'mse': trainer.metrics['mse'][-1],
                'r2': trainer.metrics['r2'][-1]
            }
            validation_results.append(validation_result)

        val_results_df = pd.DataFrame(validation_results)
        # val_results_df.to_csv(join(PathConfig.experiment_results_path.value, 'rnn_models', f'{model_class.__name__}_validation_metrics.csv'), index=False)

        best_result = val_results_df.loc[val_results_df['mse'].idxmin()]
        print(f"Best Parameters - index: {best_result['index']}")
        print(f"Best MSE: {best_result['mse']:.4f}")
        print(f"Best result: {best_result}")
        return best_result
    
    @classmethod
    def evaluate_best_model_on_test_set(cls, model_class, best_params: dict, data_preprocessor: DataPreprocessor, num_epochs: int):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_loader, validation_loader, test_loader = create_dataloaders(
            data_preprocessor=data_preprocessor,
            sequence_length=int(best_params['sequence_length']),
            batch_size=int(best_params['batch_size'])
        )

        model = model_class(
            input_size=int(best_params['input_size']),
            hidden_size=int(best_params['hidden_size']),
            num_layers=int(best_params['num_layers']),
            output_size=1,
            dropout=best_params['dropout']
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

        trainer = cls(model_class.__name__, model, train_loader, validation_loader, test_loader, criterion, optimizer, num_epochs, experiment_index=best_params['index'])
        trainer.train()

        test_loss, test_metrics = trainer.evaluate(test_loader)

        print(f"Test Results - Loss: {test_loss:.4f}, MAE: {test_metrics['mae']:.4f}, "
            f"MSE: {test_metrics['mse']:.4f}, R²: {test_metrics['r2']:.4f}")
        
        print(test_metrics)
        test_metrics = {**best_params, **test_metrics}
        print(test_metrics)
        test_metrics_df = pd.DataFrame([test_metrics])
        # test_metrics_df.to_csv(join(PathConfig.experiment_results_path.value, 'rnn_models', f'{model_class.__name__}_test_metrics.csv'), index=False)
        print(f"Metrics saved successfully in file: f'{model_class.__name__}_test_metrics.csv'")
        return test_loss, test_metrics
    
    def evaluate_model_trends(self, test_loader):
        self.trained_model.to(self.device)
        self.trained_model.eval()
        
        all_targets = []
        all_predictions = []
        all_targets_real = []
        all_outputs_real = []

        # Load the scaler
        scaler_path = join(PathConfig.base_data_path.value, 'scaler.pkl')
        scaler = joblib.load(scaler_path)

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.trained_model(inputs)
                
                # Create a dummy array to inverse transform correctly
                dummy_array = np.zeros((targets.size(0), scaler.n_features_in_))
                dummy_array[:, -1] = targets.cpu().numpy().flatten()
                targets_real = scaler.inverse_transform(dummy_array)[:, -1]

                dummy_array[:, -1] = outputs.cpu().numpy().flatten()
                outputs_real = scaler.inverse_transform(dummy_array)[:, -1]

                targets_real_direction = (targets_real > 0).astype(float)
                predicted_direction = (outputs_real > 0).astype(float)

                all_predictions.extend(predicted_direction)
                all_targets.extend(targets_real_direction)
                all_targets_real.extend(targets_real)
                all_outputs_real.extend(outputs_real)
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_targets_real = np.array(all_targets_real)
        all_outputs_real = np.array(all_outputs_real)

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='binary')
        recall = recall_score(all_targets, all_predictions, average='binary')
        f1 = f1_score(all_targets, all_predictions, average='binary')
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix of Price Movement Prediction')
        plt.show()

        # Plot the real vs predicted trends
        plt.figure(figsize=(14, 7))
        plt.plot(all_targets_real, label='Actual Trends', color='blue')
        plt.plot(all_outputs_real, label='Predicted Trends', color='red', alpha=0.6)
        plt.title('Real vs Predicted Trends')
        plt.xlabel('Time Steps')
        plt.ylabel('Percentage Change')
        plt.legend()
        plt.show()
        
        print(f"Trend Evaluation Results:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")
        return accuracy, precision, recall, f1
