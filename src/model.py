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

from bayes_opt import BayesianOptimization
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from skmisc.loess import loess
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import (nn_solvers, nn_activations)
from src.config import (calib_param, calib_param_logistic)
from src.utility import (load_ml_model, bootstrap_sample)

class MLModels:
    """Machine Learning Models
    """
    def __init__(self, n_jobs=16):
        self.n_jobs = n_jobs
        self.models = {
            "LR": LogisticRegression, # L2 Regularized Logistic Regression
            "XGB": XGBClassifier, # Extreme Gradient Boostring
            "RF": RandomForestClassifier,
            "NN": MLPClassifier  # Multilayer perceptron (aka neural network)
        }
    
    def get_LR_model(self, C=1.0, max_iter=100):
        params = {
            'C': C, 
            'class_weight': 'balanced',
            'max_iter': max_iter,
            'solver': 'sag',
            'tol': 1e-3,
            'random_state': 42
        }
        model = MultiOutputClassifier(
            CalibratedClassifierCV(
                self.models['LR'](**params), 
                n_jobs=9, 
                **calib_param_logistic
            ), 
            n_jobs=9
        )
        return model
    
    def get_XGB_model(
        self, 
        learning_rate=0.3, 
        n_estimators=100, 
        max_depth=6, 
        gamma=0, 
        reg_lambda=1
    ):
        params = {
            'learning_rate': learning_rate, 
            'n_estimators': n_estimators, 
            'max_depth': max_depth,
            'gamma': gamma, 
            'reg_lambda': reg_lambda,
            'min_child_weight': 6,
            'verbosity': 0,
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': self.n_jobs
        }
        model = MultiOutputClassifier(
            CalibratedClassifierCV(
                self.models['XGB'](**params), 
                **calib_param
            )
        )
        return model
    
    def get_RF_model(self, n_estimators=100, max_depth=6, max_features=0.5):
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_leaf': 6, # can't allow leaf node to have less than 6 samples
            'class_weight': 'balanced_subsample',
            'random_state': 42,
            'n_jobs': self.n_jobs
        }
        model = MultiOutputClassifier(
            CalibratedClassifierCV(
                self.models['RF'](**params), 
                **calib_param
            )
        )
        return model
    
    def get_NN_model(
        self, 
        learning_rate_init=0.001, 
        batch_size=200, 
        momentum=0.9, 
        alpha=0.0001, 
        first_layer_size=100, 
        second_layer_size=100, 
        solver=0, 
        activation=0, 
        max_iter=20, 
        verbose=False
    ):
        params = {
            'learning_rate_init': learning_rate_init,
            'batch_size': batch_size,
            'momentum': momentum,
            'alpha': alpha,
            'hidden_layer_sizes': (first_layer_size, second_layer_size),
            'solver': nn_solvers[int(np.floor(solver))],
            'activation': nn_activations[int(np.floor(activation))],
            'max_iter': max_iter,
            'verbose': verbose,
            'tol': 1e-3,
            'random_state': 42
        }
        model = self.models['NN'](**params)
        return model
    
class RNN(nn.Module):
    """Recurrent Neural Network Model
    Supports GRU (Gated Recurrent Unit) or LSTM (Long Short-Term Memory)
    """
    def __init__(
        self, 
        n_features, 
        n_targets, 
        hidden_size, 
        hidden_layers, 
        batch_size, 
        dropout, 
        pad_value, 
        model='GRU'
    ):
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
        self.rnn_layers = network(
            input_size=self.n_features, hidden_size=self.hidden_size, 
            num_layers=self.hidden_layers, dropout=self.dropout, 
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_targets
        )

    def forward(self, packed_padded_inputs):
        packed_outputs, _ = self.rnn_layers(packed_padded_inputs)
        # padded_packed_outputs.shape = [Batch Size x Longest Sequence Length x Hidden Size]
        padded_packed_outputs, lengths = pad_packed_sequence(
            packed_outputs, batch_first=True, padding_value=self.pad_value
        )
        # outputs.shape = [Batch Size x Longest Sequence Length x Number of Targets]
        output = self.linear(padded_packed_outputs)
        return output
    
    def init_hidden(self):
        return torch.zeroes(self.hidden_layers, self.batch_size, self.hidden_size)
    
class IsotonicCalibrator:
    """Calibrator. Matches predicted probability with empirical probability via 
    isotonic regression
    """
    def __init__(self, target_events):
        self.target_events = target_events
        self.regressor = {target_event: IsotonicRegression(out_of_bounds='clip')
                          for target_event in self.target_events}
    
    def load_model(self, model_dir, algorithm):
        self.regressor = load_ml_model(model_dir, algorithm, name='calibrator') 
        assert all(target_event in self.regressor 
                   for target_event in self.target_events)
        
    def save_model(self, model_dir, algorithm):
        model_filename = f'{model_dir}/{algorithm}_calibrator.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(self.regressor, file)
        
    def calibrate(self, pred, label):
        # Reference: github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py
        for target_event in self.target_events:
            self.regressor[target_event].fit(
                pred[target_event], label[target_event]
            )
        
    def predict(self, pred_prob):
        result = []
        for target_event, predictions in pred_prob.iteritems():
            try:
                result.append(self.regressor[target_event].predict(predictions))
            except AttributeError as e:
                if str(e) == "'IsotonicRegression' object has no attribute 'X_min_'":
                    raise ValueError('Model has not been calibrated. '
                                     'Please calibrate the model first')
                else: 
                    raise e
        result = np.array(result).T
        result = pd.DataFrame(
            result, 
            columns=pred_prob.columns, 
            index=pred_prob.index
        )
        return result

###############################################################################
# Baseline Models
###############################################################################
class BaselineModel:
    """Baseline model in which a single predictor/variable/column is used to 
    predict a target
    """
    def __init__(self, **params):
        self.params = params
        self.models = {}
        self.target_events = []
        
    def _fit(self, x, y):
        raise NotImplementedError
        
    def _predict(self, model, x):
        raise NotImplementedError
        
    def _predict_with_confidence_interval(self, model, x, **kwargs):
        raise NotImplementedError
        
    def fit(self, x, Y):
        for target_event, y in Y.iteritems():
            self.models[target_event] = self._fit(x, y)
        self.target_events = Y.columns.tolist()
        
    def predict(self, x):
        preds = {target_event: self._predict(self.models[target_event], x)
                 for target_event in self.target_events}
        return preds
    
    def predict_with_confidence_interval(self, x, **kwargs):
        preds, ci_lower, ci_upper = {}, {}, {}
        for tt in self.target_events:
            output = self._predict_with_confidence_interval(tt, x, **kwargs)
            preds[tt], ci_lower[tt], ci_upper[tt] = output
        return preds, ci_lower, ci_upper
    
class LOESSModel(BaselineModel):
    """Locally estimated scatterplot smoothing (LOESS) model
    """
    def _fit(self, x, y):
        model = loess(x, y, **self.params)
        model.fit()
        return model

    def _predict(self, model, x):
        return model.predict(x).values
    
    def _predict_with_confidence_interval(self, target_event, x):
        model = self.models[target_event]
        pred = model.predict(x, stderror=True)
        ci = pred.confidence(0.05)
        return pred.values, ci.lower, ci.upper
    
class PolynomialModel(BaselineModel):
    """Polynomial Model
    
    Supported algorithms include:
    - SPLINE: B-spline basis functions (piecewise polynomials)
    - POLY: pure polynomials
    """
    def __init__(self, algorithm, task_type, **params):
        super().__init__(**params)
        poly_models = {'SPLINE': SplineTransformer, 'POLY': PolynomialFeatures}
        reg_models = {'regression': Ridge, 'classification': LogisticRegression}
        self.poly_model = poly_models[algorithm]
        self.reg_model = reg_models[task_type]
        self.task_type = task_type
        
    def _format(self, x):
        if len(x.shape) == 1: x = np.expand_dims(x, axis=1)
        return x
        
    def _fit(self, x, y):
        model = make_pipeline(
            self.poly_model(**self.params['PLY']), 
            self.reg_model(**self.params['REG'])
        )
        model.fit(self._format(x), y)
        return model
    
    def _predict(self, model, x):
        if self.task_type == 'regression':
            return model.predict(self._format(x))
        else:
            return model.predict_proba(self._format(x))[:, 1]
    
    def _compute_confidence_interval(self, x, x_train, y_train, n_bootstraps=1000):
        ci_preds = []
        for random_seed in tqdm(range(n_bootstraps)):
            Y = bootstrap_sample(y_train, random_seed)
            X = x_train.loc[Y.index]
            model = self._fit(X, Y)
            pred = self._predict(model, x)
            ci_preds.append(pred)
        return np.array(ci_preds)
    
    def _predict_with_confidence_interval(
        self, 
        target_event, 
        x, 
        x_train=None,
        y_train=None
    ):
        """For spline and poly algorithm, confidence interval is calculated by 
        permuting the data used to originally train the model, retraining the 
        model, recomputing the prediction for x, and taking the 2.5% and 97.5% 
        quantiles of each prediction for sample n in x
        """
        model = self.models[target_event]
        pred =  self._predict(model, x)
        ci_preds = self._compute_confidence_interval(
            x, x_train, y_train[target_event]
        )
        lower, upper = np.percentile(ci_preds, [2.5, 97.5], axis=0).round(3)
        return pred, lower, upper
