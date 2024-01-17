"""
========================================================================
© 2023 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)

from .TCN import TCNModel

class Models:
    def __init__(self, task='C'):
        """
        Args:
            task (str): the type of machine learning task, either `C` for 
                classifcation or `R` for regression
        """
        self.task = task
        self.types = {
            'LR': Regression,
            'RF': RandomForest,
            'XGB': ExtremeGradientBoosting,
            'NN': NeuralNetwork,
            'RNN': RecurrentNeuralNetwork
        }
        self._experimentally_supported_types = {
            'TCN': TemporalConvolutionalNetwork
        }

    def __iter__(self):
        return iter(self.types)
    
    def get_model(self, alg, *args, **params):
        params['task'] = self.task
        if alg not in self.types:
            return self._experimentally_supported_types[alg](*args, **params)
        return self.types[alg](*args, **params)

###############################################################################
# Machine Learning Models
###############################################################################
class MLModel:
    def __init__(self, task='C', random_state=42, n_jobs=32):
        self.task = task
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.MultiOutput = {
            'C': MultiOutputClassifier, 
            'R': MultiOutputRegressor
        }[task]
        self.model = None
    
    def predict(self, X):
        if self.task == 'R': return self.model.predict(X)
        pred = self.model.predict_proba(X)
        return np.array(pred)[:, :, 1].T
    
    def fit(self, X, Y):
        self.model.fit(X, Y)
    
class Regression(MLModel):
    def __init__(
        self, 
        inv_reg_strength=1.0, 
        max_iter=100,
        tol=1e-3, 
        penalty='l2',
        **kwargs
    ):
        super().__init__(**kwargs)
        model_type = {'C': LogisticRegression, 'R': Ridge}[self.task]
        params = {
            'solver': 'saga',
            'max_iter': max_iter,
            'tol': tol,
            'random_state': self.random_state,
        }
        if self.task == 'C':
            params['C'] = inv_reg_strength
            params['class_weight'] = 'balanced'
            params['penalty'] = penalty
        elif self.task == 'R':
            params['alpha'] = inv_reg_strength / 1    
        model = model_type(**params)
        model = self.MultiOutput(model, n_jobs=9)
        self.model = model
        
class RandomForest(MLModel):
    def __init__(
        self, 
        n_estimators=100,
        max_depth=None, 
        max_features='sqrt',
        min_samples_leaf=6,
        min_impurity_decrease=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        model_types = {'C': RandomForestClassifier, 'R': RandomForestRegressor}
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'min_impurity_decrease': min_impurity_decrease,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        if self.task == 'C':
            params['class_weight'] = 'balanced_subsample'
        model = model_types[self.task](**params)
        model = self.MultiOutput(model)
        self.model = model
    
class ExtremeGradientBoosting(MLModel):
    def __init__(
        self, 
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.3, 
        min_split_loss=0, 
        min_child_weight=6,
        reg_lambda=1,
        reg_alpha=0,
        verbosity=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        model_types = {'C': XGBClassifier, 'R': XGBRegressor}
        params = {
            'n_estimators': n_estimators, 
            'max_depth': max_depth,
            'learning_rate': learning_rate, 
            'min_split_loss': min_split_loss, 
            'min_child_weight': min_child_weight,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'verbosity': verbosity,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        model = model_types[self.task](**params)
        model = self.MultiOutput(model)
        self.model = model
        
###############################################################################
# Deep Learning Models
###############################################################################
class DLModel:
    def __init__(self, task='C'):
        self.task = task
        # NOTE: BCEWithLogitLoss COMBINES Sigmoid and BCELoss. This is more 
        # numerically stable than using Sigmoid followed by BCELoss. As a 
        # result, the model output during training should NOT use a Simgoid 
        # layer at the end
        self.loss_type = {'C': nn.BCEWithLogitsLoss, 'R': nn.MSELoss}[task]
        self.use_gpu = torch.cuda.is_available()
        self.model = None
    
    def eval(self):
        # enters evaluation mode - deactivates dropout
        self.model.eval()
        
    def train(self):
        # enters training mode - activates dropout
        self.model.train()
        
    def state_dict(self):
        # return the model weights
        return self.model.state_dict()
            
    def load_weights(self, state_dict):
        if isinstance(state_dict, str):
            map_location = None if self.use_gpu else torch.device('cpu')
            state_dict = torch.load(state_dict, map_location=map_location)
        self.model.load_state_dict(state_dict)

    def clip_gradients(self, clip_value=1):
        nn.utils.clip_grad_value_(self.model.parameters(), clip_value=clip_value)
        
class NeuralNetwork(DLModel):
    def __init__(
        self, 
        n_features,
        n_targets,
        hidden_size1=128,
        hidden_size2=64,
        dropout=0,
        optimizer='adam',
        learning_rate=1e-2,
        weight_decay=0,
        momentum=0,
        beta1=0.9,
        beta2=0.999,
        **kwargs
    ):
        super().__init__(**kwargs)
        optimizer_types = {'adam': optim.Adam, 'sgd': optim.SGD}
        params = {
            'input_size': n_features,
            'output_size': n_targets,
            'hidden_size1': hidden_size1,
            'hidden_size2': hidden_size2,
            'dropout': dropout,
        }
        self.model = NN(**params)
        self.criterion = self.loss_type(reduction='none')
        if optimizer == 'sgd':
            params = {'momentum': momentum}
        elif optimizer == 'adam':
            params = {'betas': (beta1, beta2)}
        self.optimizer = optimizer_types[optimizer](
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay, 
            **params
        )
        if self.use_gpu:
            self.model.cuda()
            
    def predict(self, X, grad=False, bound_pred=True):
        """
        Args: 
            grad (bool): If True, enable gradients for backward pass
            bound_pred (bool): If True, bound the predictions by using Sigmoid 
                over the model output
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X.astype(float).to_numpy())
        
        if self.use_gpu: 
            X = X.cuda()
            
        with torch.set_grad_enabled(grad):
            # NOTE: need to call .detach() is True
            pred = self.model(X)
            
        if bound_pred: 
            pred = torch.sigmoid(pred)
            
        return pred
    
class NN(nn.Module):
    def __init__(
        self, 
        input_size,
        hidden_size1,
        hidden_size2,
        output_size,
        dropout=0,
    ):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X):
        X = self.dropout(self.relu(self.layer1(X)))
        X = self.dropout(self.relu(self.layer2(X)))
        return self.output(X)
    
class RecurrentNeuralNetwork(DLModel):
    def __init__(
        self, 
        n_features,
        n_targets,
        hidden_size=128,
        hidden_layers=1,
        dropout=0,
        learning_rate=1e-2,
        weight_decay=0,
        beta1=0.9,
        beta2=0.999,
        pad_value=-999, 
        model='GRU',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pad_value = pad_value
        self.n_targets = n_targets
        params = {
            'input_size': n_features,
            'output_size': n_targets,
            'hidden_size': hidden_size,
            'hidden_layers': hidden_layers,
            'dropout': dropout,
            'pad_value': pad_value,
            'model': model
        }
        self.model = RNN(**params)
        self.criterion = self.loss_type(reduction='none')
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay, 
            betas=(beta1, beta2)
        )
        if self.use_gpu:
            self.model.cuda()
            
    def predict(self, X, grad=False, bound_pred=True):
        """
        Args:
            X: batched data from train.SeqData consisting of feature tensors, 
                target tensors, and index tensors of a batch of patients
            grad (bool): If True, enable gradients for backward pass
            bound_pred (bool): If True, bound the predictions applying Sigmoid
                over the model output
        """
        # each is a tuple of tensors
        inputs, targets, indices = tuple(zip(*X))

        # format sequences
        formatted_seqs = self.format_sequences(inputs, targets)
        packed_padded_inputs, padded_targets = formatted_seqs
        targets = torch.cat(targets).float() # concatenate the tensors
        indices = torch.cat(indices).float()
        if self.use_gpu:
            packed_padded_inputs = packed_padded_inputs.cuda()
            targets = targets.cuda()
            
        # make predictions
        # for each patient, for each timestep, a prediction was made given the
        # prev sequence history of that time step
        with torch.set_grad_enabled(grad):
            preds = self.model(packed_padded_inputs)
        
        # unpad predictions based on target lengths
        preds = preds[padded_targets != self.pad_value] 
        
        # ensure preds shape matches targets shape
        preds = preds.reshape(-1, self.n_targets)
        
        # bound the model prediction
        if bound_pred: 
            preds = torch.sigmoid(preds)
            
        return preds, targets, indices
            
    def format_sequences(self, inputs, targets):
        """Format the variable length sequences
        
        1. Pad the variable length sequences
        2. Pack the padded sequences to optimize computations 
        
        If one of the sequence is significantly longer than other sequences, 
        and we pad all the sequences to the same length, we will be doing a lot
        of unnecessary computations with the padded values

        Code Example:
        a = [torch.Tensor([1,2,3]), torch.Tensor([3,4])]
        b = pad_sequence(a, batch_first=True, padding_value=-1)
        >>> tensor([[1,2,3],
                    [3,4,-1]])
        c = pack_padded_sequence(b, batch_first=True, lengths=[3,2])
        >>> PacekedSequence(data=tensor([1,3,2,4,3]), batch_sizes=tensor([2,2,1]))

        data = all the tensors concatenated along the "time" axis. 
        batch_sizes = array of batch sizes at each "time" step. For example, 
        [2,2,1] represent the grouping [1,3], [2,4], [3]
        """
        padded_inputs = pad_sequence(
            inputs, batch_first=True, padding_value=self.pad_value
        )
        padded_targets = pad_sequence(
            targets, batch_first=True, padding_value=self.pad_value
        )
        seq_lengths = list(map(len, inputs))
        packed_padded_inputs = pack_padded_sequence(
            padded_inputs, seq_lengths, batch_first=True, enforce_sorted=False
        ).float()
        return packed_padded_inputs, padded_targets

class RNN(nn.Module):
    """Recurrent Neural Network Model
    Supports GRU (Gated Recurrent Unit) or LSTM (Long Short-Term Memory)
    """
    def __init__(
        self, 
        input_size,
        output_size,
        hidden_size, 
        hidden_layers, 
        dropout=0, 
        pad_value=-999, 
        model='GRU'
    ):
        super(RNN, self).__init__()
        self.pad_value = pad_value
        network_types = {'GRU': nn.GRU, 'LSTM': nn.LSTM}
        self.rnn = network_types[model](
            input_size=input_size, hidden_size=hidden_size, 
            num_layers=hidden_layers, dropout=dropout, batch_first=True
        )
        # fc = fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, packed_padded_inputs):
        packed_outputs, _ = self.rnn(packed_padded_inputs)
        # padded_packed_outputs.shape = [Batch Size x Longest Sequence Length x Hidden Size]
        padded_packed_outputs, lengths = pad_packed_sequence(
            packed_outputs, batch_first=True, padding_value=self.pad_value
        )
        # outputs.shape = [Batch Size x Longest Sequence Length x Number of Targets]
        output = self.fc(padded_packed_outputs)
        return output


class TemporalConvolutionalNetwork(DLModel):
    def __init__(
        self,
        n_features,
        n_targets,
        num_channel1,
        num_channel2,
        num_channel3,
        kernel_size=3,
        dropout=0,
        learning_rate=1e-3,
        weight_decay=0,
        beta1=0.9,
        beta2=0.999,
        pad_value=-999,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pad_value = pad_value
        self.n_targets = n_targets
        params = {
            'input_size': n_features,
            'output_size': n_targets,
            'num_channels': [num_channel1, num_channel2, num_channel3],
            'kernel_size': kernel_size,
            'dropout': dropout
        }
        self.model = TCNModel(**params)
        self.criterion = self.loss_type(reduction='none')
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay, 
            betas=(beta1, beta2)
        )
        if self.use_gpu:
            self.model.cuda()

    def predict(self, X, grad=False, bound_pred=True):
        """
        Args:
            X: batched data from train.SeqData consisting of feature tensors, 
                target tensors, and index tensors of a batch of patients
            grad (bool): If True, enable gradients for backward pass
            bound_pred (bool): If True, bound the predictions applying Sigmoid
                over the model output
        """
        # each is a tuple of tensors
        inputs, targets, indices = tuple(zip(*X))

        # format sequences
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_value)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_value)
        targets, indices = torch.cat(targets).float(), torch.cat(indices).float()
        if self.use_gpu: padded_inputs, targets = padded_inputs.cuda(), targets.cuda()
            
        # make predictions
        with torch.set_grad_enabled(grad):
            preds = self.model(padded_inputs)
        
        # unpad predictions based on target lengths
        preds = preds[padded_targets != self.pad_value] 
        
        # ensure preds shape matches targets shape
        preds = preds.reshape(-1, self.n_targets)
        
        # bound the model prediction
        if bound_pred: 
            preds = torch.sigmoid(preds)
            
        return preds, targets, indices