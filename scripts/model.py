"""
========================================================================
© 2018 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from skmisc.loess import loess
from bayes_opt import BayesianOptimization

from scripts.config import (nn_solvers, nn_activations, calib_param, calib_param_logistic)
from scripts.utility import (load_ml_model)

class MLModels:
    """Machine Learning Models
    """
    def __init__(self, n_jobs=16):
        self.n_jobs = n_jobs
        self.models = {"LR": LogisticRegression, # L2 Regularized Logistic Regression
                       "XGB": XGBClassifier, # Extreme Gradient Boostring
                       "RF": RandomForestClassifier,
                       "NN": MLPClassifier} # Multilayer perceptron (aka neural network)
        
        self.model_tuning_config = {'LR': [{'init_points': 3, 'n_iter': 10}, 
                                           {'C': (0.0001, 1)}],
                                    'XGB': [{'init_points': 5, 'n_iter': 25}, 
                                            {'learning_rate': (0.001, 0.1),
                                             'n_estimators': (50, 200),
                                             'max_depth': (3, 7),
                                             'gamma': (0, 1),
                                             'reg_lambda': (0, 1)}],
                                    'RF': [{'init_points': 3, 'n_iter': 20}, 
                                           {'n_estimators': (50, 200),
                                            'max_depth': (3, 7),
                                            'max_features': (0.01, 1)}],
                                    # NN is actually deep learning, but whatever lol
                                    'NN': [{'init_points': 5, 'n_iter': 50}, 
                                           {'learning_rate_init': (0.0001, 0.1),
                                            'batch_size': (64, 512),
                                            'momentum': (0,1),
                                            'alpha': (0,1),
                                            'first_layer_size': (16, 256),
                                            'second_layer_size': (16, 256),
                                            'solver': (0, len(nn_solvers)-0.0001),
                                            'activation': (0, len(nn_activations)-0.0001)}]}
    
    def get_LR_model(self, C=1.0, max_iter=100):
        params = {'C': C, 
                  'class_weight': 'balanced',
                  'max_iter': max_iter,
                  'solver': 'sag',
                  'tol': 1e-3,
                  'random_state': 42}
        model = MultiOutputClassifier(CalibratedClassifierCV(self.models['LR'](**params), n_jobs=9, **calib_param_logistic), n_jobs=9)
        return model
    
    # weight for positive examples to account for imbalanced dataset
    # scale_pos_weight = [neg_count/pos_count for index, (neg_count, pos_count) in Y_distribution['Train'].iterrows()]
    # min_child_weight = max(scale_pos_weight) * 6 # can't have less than 6 samples in a leaf node
    def get_XGB_model(self, learning_rate=0.3, n_estimators=100, max_depth=6, gamma=0, reg_lambda=1):
        params = {'learning_rate': learning_rate, 
                  'n_estimators': n_estimators, 
                  'max_depth': max_depth,
                  'gamma': gamma, 
                  'reg_lambda': reg_lambda,
                  # 'scale_pos_weight': scale_pos_weight,
                  # 'min_child_weight': min_child_weight, # set to 6 if not using scale_pos_weight
                  'min_child_weight': 6,
                  'verbosity': 0,
                  'use_label_encoder': False,
                  'random_state': 42,
                  'n_jobs': self.n_jobs,
                 }
        model = MultiOutputClassifier(CalibratedClassifierCV(self.models['XGB'](**params), **calib_param))
        return model
    
    def get_RF_model(self, n_estimators=100, max_depth=6, max_features=0.5):
        params = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'max_features': max_features,
                  'min_samples_leaf': 6, # can't allow leaf node to have less than 6 samples
                  'class_weight': 'balanced_subsample',
                  'random_state': 42,
                  'n_jobs': self.n_jobs}
        model = MultiOutputClassifier(CalibratedClassifierCV(self.models['RF'](**params), **calib_param))
        return model
    
    def get_NN_model(self, learning_rate_init=0.001, batch_size=200, momentum=0.9, alpha=0.0001, 
                     first_layer_size=100, second_layer_size=100, solver=0, activation=0, max_iter=20, verbose=False):
        params = {'learning_rate_init': learning_rate_init,
                  'batch_size': batch_size,
                  'momentum': momentum,
                  'alpha': alpha,
                  'hidden_layer_sizes': (first_layer_size, second_layer_size),
                  'solver': nn_solvers[int(np.floor(solver))],
                  'activation': nn_activations[int(np.floor(activation))],
                  'max_iter': max_iter,
                  'verbose': verbose,
                  'tol': 1e-3,
                  'random_state': 42}
        # model = MultiOutputClassifier(CalibratedClassifierCV(ml_models['NN'](**params), **calib_param)) # 3 MLP, each outputs 1 value
        model = self.models['NN'](**params) # 1 MLP, outputs 3 values
        return model
    
class RNN(nn.Module):
    """Recurrent Neural Network Model - supports GRU (Gated Recurrent Unit) or LSTM (Long Short-Term Memory)
    """
    def __init__(self, n_features, n_targets, hidden_size, hidden_layers, batch_size, dropout, pad_value, model='GRU'):
        super().__init__()
        self.name = model
        if self.name not in {'GRU', 'LSTM'}: 
            raise ValueError('model must be either GRU or LSTM')
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.dropout = dropout
        network = getattr(nn, self.name)
        # stack(s) of (bidirectional) LSTM/GRU layers, followed by a linear layer head for the final classification
        # use np.bool_ or else bayesopt throws a fit
        self.rnn_layers = network(input_size=self.n_features, hidden_size=self.hidden_size, num_layers=self.hidden_layers, 
                                  dropout=self.dropout, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size, 
                                out_features=self.n_targets)

    def forward(self, packed_padded_inputs):
        packed_outputs, _ = self.rnn_layers(packed_padded_inputs)
        # padded_packed_outputs.shape = [Batch Size x Longest Sequence Length x Hidden Size]
        padded_packed_outputs, lengths = pad_packed_sequence(packed_outputs, batch_first=True, padding_value=self.pad_value)
        # outputs.shape = [Batch Size x Longest Sequence Length x Number of Targets]
        output = self.linear(padded_packed_outputs)
        return output
    
    def init_hidden(self):
        return torch.zeroes(self.hidden_layers, self.batch_size, self.hidden_size)
    
class IsotonicCalibrator:
    """Calibrator - matches predicted probability with empirical probability via isotonic regression
    """
    def __init__(self, target_types):
        self.target_types = target_types
        self.regressor = {target_type: IsotonicRegression(out_of_bounds='clip') for target_type in self.target_types}
    
    def load_model(self, model_dir, algorithm):
        self.regressor = load_ml_model(model_dir, algorithm, name='calibrator') 
        assert all(target_type in self.regressor for target_type in self.target_types)
        
    def save_model(self, model_dir, algorithm):
        model_filename = f'{model_dir}/{algorithm}_calibrator.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(self.regressor, file)
        
    def calibrate(self, pred, label):
        for target_type in self.target_types:
            # Reference: github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py
            self.regressor[target_type].fit(pred[target_type], label[target_type])
        
    def predict(self, pred_prob):
        result = []
        for target_type, predictions in pred_prob.iteritems():
            try:
                result.append(self.regressor[target_type].predict(predictions))
            except AttributeError as e:
                if str(e) == "'IsotonicRegression' object has no attribute 'X_min_'":
                    raise ValueError('Model has not been calibrated. Please calibrate the model first')
                else: 
                    raise e
        result = np.array(result).T
        return pd.DataFrame(result, columns=pred_prob.columns, index=pred_prob.index)
    
class LOESS:
    """Locally estimated scatterplot smoothing model
    Used as a baseline model in which a single predictor/variable/column is used to predict a target
    """
    def __init__(self, **params):
        self.params = params
        self.models = {}
        self.target_types = []
        
    def fit(self, x, Y):
        for target_type, y in Y.iteritems():
            self.models[target_type] = loess(x, y, **self.params)
            self.models[target_type].fit()
        self.target_types = Y.columns.tolist()
        
    def predict(self, x):
        preds = {target_type: self.models[target_type].predict(x).values for target_type in self.target_types}
        return preds
    
    def predict_with_confidence_interval(self, x):
        preds, ci_lower, ci_upper = {}, {}, {}
        for target_type in self.target_types:
            pred = self.models[target_type].predict(x, stderror=True)
            ci = pred.confidence(0.05)
            preds[target_type], ci_lower[target_type], ci_upper[target_type] = pred.values, ci.lower, ci.upper
        return preds, ci_lower, ci_upper
