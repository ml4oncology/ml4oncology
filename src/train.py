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
"""
Module for training models

Current inheritance heirarchy:
Train -> TrainML -> TrainLASSO
      -> TrainRNN
      -> TrainSingleFeatureBaselineModel -> TrainLOESSModel
                                         -> TrainPolynomialModel
PLEASE TRY TO KEEP INHEIRTANCE LEVELS AT MAXIMUM OF 3
"""
from functools import partial
import os
import pickle

from bayes_opt import BayesianOptimization
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_squared_error)
from torch.nn.utils.rnn import (pad_sequence, pack_padded_sequence)
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

from src import logging
from src.conf_int import AUCConfidenceInterval
from src.config import (model_tuning_param, bayesopt_param)
from src.model import (
    IsotonicCalibrator, 
    LOESSModel, 
    MLModels, 
    PolynomialModel, 
    RNN
)
from src.utility import (
    load_ensemble_weights, 
    load_ml_model, 
    load_predictions, 
    save_predictions,
)

torch.manual_seed(0)
np.random.seed(0)

###############################################################################
# Base Class
###############################################################################
class Train:
    def __init__(self, X, Y, tag, output_path):    
        self.n_features = X.shape[1]
        self.n_targets = Y.shape[1]
        self.target_events = Y.columns.tolist()
        self.output_path = output_path
        self.datasets = {split: x for split, x in X.groupby(tag['split'])}
        self.labels = {split: y for split, y in Y.groupby(tag['split'])}
        self.splits = list(self.labels.keys())
        
    def convert_param_types(self, best_param):
        """You can overwrite this to convert the hyperparmeter data types as 
        desired
        """
        return best_param
    
    def _bayesopt(
        self, 
        alg, 
        eval_func, 
        hyperparam_config, 
        optim_config,        
        filename='', 
        **kwargs
    ):
        """Conduct bayesian optimization, a sequential search framework 
        for finding optimal hyperparameters using bayes theorem
        """
        bo = BayesianOptimization(eval_func, hyperparam_config, **kwargs)
        bo.maximize(acq='ei', **optim_config)
        best_param = bo.max['params']
        best_param = self.convert_param_types(best_param)
        logging.info(f'Finished finding best hyperparameters for {alg}')
        logging.info(f'Best param: {best_param}')

        # Save the best hyperparameters
        if not filename: filename = f'{alg}_best_param'
        savepath = f'{self.output_path}/best_params/{filename}.pkl'
        with open(savepath, 'wb') as file:    
            pickle.dump(best_param, file)

        return best_param

###############################################################################
# Main Machine Learning Models
###############################################################################
class TrainML(Train):
    """Train machine learning models
    Employ model calibration and Bayesian Optimization
    """
    def __init__(self, X, Y, tag, output_path, **kwargs):
        """
        Args:
            X (pd.DataFrame): table of input features
            Y (pd.DataFrame): table of target labels
            tag (pd.DataFrame): table containing partition names (e.g. Train, 
                Valid, etc) associated with each sample
            **kwargs (dict): the parameters of MLModels
        """
        super().__init__(X, Y, tag, output_path)
        self.ml = MLModels(**kwargs)
        # memory / cache to store the model predictions
        self.preds = {split: {alg: None for alg in self.ml.models} 
                      for split in self.labels}
       
    def _predict(self, model, alg, X):
        pred = model.predict_proba(X)
        if alg != 'NN': 
            # format it to be row=treatment sessions, columns=targets
            # [:, :, 1] - first column is prob of false, second column is prob 
            # of true
            pred = np.array(pred)[:, :, 1].T
        elif len(self.target_events) == 1: 
            # seems for single target, NN outputs prob for both false and true,
            # instead of just true
            pred = pred[:, 1]
        return pred
        
    def predict(self, model, split, alg, target_event=None, store=True):
        X, Y = self.datasets[split], self.labels[split]
        if target_event is not None: Y = Y[[target_event]]
        
        # Check the cache
        if self.preds[split][alg] is None:
            pred = self._predict(model, alg, X)
            # make your life easier by ensuring pred and Y have same data format
            pred = pd.DataFrame(pred, index=Y.index, columns=Y.columns)
            if store: self.preds[split][alg] = pred
        else:
            pred = self.preds[split][alg]
            
        return pred
    
    def train_model(
        self, 
        alg, 
        target_event=None, 
        save=True, 
        filename='', 
        **kwargs
    ):
        X, Y = self.datasets['Train'], self.labels['Train']
        if target_event is not None: Y = Y[[target_event]]
        get_model = getattr(self.ml, f'get_{alg}_model')
        model = get_model(**kwargs)
        model.fit(X, Y)

        if save:
            if not filename: filename = f'{alg}_classifier'
            filepath = f'{self.output_path}/{filename}.pkl'
            with open(filepath, 'wb') as file:
                pickle.dump(model, file)

        return model
    
    def convert_param_types(self, params):
        int_params = [
            'max_depth', 'batch_size', 'n_estimators', 'first_layer_size', 
            'second_layer_size'
        ]
        for param, value in params.items():
            if param in int_params:
                params[param] = int(value)
        return params

    def bo_evaluate(self, alg, split='Valid', **kwargs):
        """Evaluation function for bayesian optimization
        
        Returns:
            The mean (macro-mean) of auroc scores of all target types
        """
        kwargs = self.convert_param_types(kwargs)
        model = self.train_model(alg, save=False, **kwargs)
        pred_prob = self.predict(model, split, alg, store=False)
        return roc_auc_score(self.labels[split], pred_prob)

    def bayesopt(self, alg, random_state=42):
        hyperparam_config = model_tuning_param[alg]
        optim_config = bayesopt_param[alg]
        evaluate_function = partial(self.bo_evaluate, alg=alg)
        return self._bayesopt(
            alg, evaluate_function, hyperparam_config, optim_config, 
            random_state=random_state
        )

    def tune_and_train(
        self, 
        run_bayesopt=True, 
        run_training=True, 
        save_preds=True, 
        algs=None, 
        random_state=42
    ):
        """
        Args:
            algs (list): A sequence of algorithms (str) to train/tune. If None,
                train/tune all algorithms
        """
        if algs is None: algs = self.ml.models
        for alg in algs:
            # Tune Hyperparameters
            if run_bayesopt:
                best_param = self.bayesopt(alg, random_state=random_state)
            else:
                filepath = f'{self.output_path}/best_params/{alg}_best_param.pkl'
                if not os.path.exists(filepath):
                    raise ValueError('Please run bayesian optimization for '
                                     f'{alg} to obtain best hyperparameters')
                with open(filepath, 'rb') as file: 
                    best_param = pickle.load(file)

            # Train Models
            if run_training:
                if alg == 'NN': 
                    best_param['max_iter'] = 100
                    best_param['verbose'] = True
                elif alg == 'LR': 
                    best_param['max_iter'] = 1000
                self.train_model(alg, **best_param)
                logging.info(f'{alg} training completed!')
            model = load_ml_model(self.output_path, alg)

            # Store Predictions
            for split in self.splits:
                self.predict(model, split, alg, store=True)
                
        if save_preds: 
            save_predictions(
                self.preds, save_dir=f'{self.output_path}/predictions'
            )

###############################################################################
# Recurrent Neural Network Model
###############################################################################
class SeqData(TensorDataset):
    def __init__(self, X_mapping, Y_mapping, ids):
        self.X_mapping = X_mapping
        self.Y_mapping = Y_mapping
        self.ids = ids
                
    def __getitem__(self, index):
        sample = self.ids[index]
        X, Y = self.X_mapping[sample], self.Y_mapping[sample]
        features_tensor = torch.Tensor(X.values)
        target_tensor = torch.Tensor(Y.values)
        indices_tensor = torch.Tensor(Y.index)
        return features_tensor, target_tensor, indices_tensor
    
    def __len__(self):
        return(len(self.ids))
    
class TrainRNN(Train):
    def __init__(self, X, Y, tag, output_path):
        """
        Args:
            X (pd.DataFrame): table of input features
            Y (pd.DataFrame): table of target labels
            tag (pd.DataFrame): table containing partition names (e.g. Train, 
                Valid, etc) associated with each sample
        """
        super().__init__(X, Y, tag, output_path)
        self.ikns = tag['ikn']
        
        to_tensor = lambda x, y: self.transform_to_tensor_dataset(x, y)
        self.tensor_dataset, self.preds = {}, {} 
        for split, Y in self.labels.items():
            self.tensor_dataset[split] = to_tensor(self.datasets[split], Y)
            self.preds[split] = pd.DataFrame(index=Y.index)
        
        self.calibrator = IsotonicCalibrator(self.target_events)
        # the padding value for padding variable length sequences
        self.pad_value = -999 
        
    def transform_to_tensor_dataset(self, X, Y):
        ikns = self.ikns[X.index]
        X = X.astype(float)
        X_mapping = {ikn: group for ikn, group in X.groupby(ikns)}
        Y_mapping = {ikn: group for ikn, group in Y.groupby(ikns)}
        return SeqData(
            X_mapping=X_mapping, Y_mapping=Y_mapping, ids=ikns.unique()
        )

    def get_model(self, load_saved_weights=False, **model_param):
        model = RNN(
            n_features=self.n_features, n_targets=self.n_targets, 
            pad_value=self.pad_value, **model_param
        )
        
        if torch.cuda.is_available():
            model.cuda()
        if load_saved_weights:
            save_path = f'{self.output_path}/{model.name}_classifier'
            map_location = None if torch.cuda.is_available() else torch.device('cpu')
            model.load_state_dict(torch.load(save_path, map_location=map_location))
        return model
    
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
    
    def validate(self, model, loader, criterion):
        """Refer to train_classification for detailed comments
        """
        total_loss = 0
        total_score = 0
        for i, batch in enumerate(loader):
            inputs, targets, _ = tuple(zip(*batch))
            formatted_seqs = self.format_sequences(inputs, targets)
            packed_padded_inputs, padded_targets = formatted_seqs
            targets = torch.cat(targets).float()
            if torch.cuda.is_available():
                packed_padded_inputs = packed_padded_inputs.cuda()
                targets = targets.cuda()
            preds = model(packed_padded_inputs)
            mask = padded_targets != self.pad_value
            preds = preds[mask].reshape(-1, self.n_targets)
            loss = criterion(preds, targets)
            loss = loss.mean(dim=0)
            total_loss += loss
            preds = torch.sigmoid(preds)
            preds = preds > 0.5
            if torch.cuda.is_available():
                preds = preds.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
            total_score += np.array([accuracy_score(targets[:, i], preds[:, i])
                                     for i in range(self.n_targets)])
        return total_loss.cpu().detach().numpy() / (i+1), total_score/(i+1)
    
    def train_model(
        self, 
        epochs=200, 
        batch_size=512, 
        learning_rate=0.001, 
        decay=0,        
        hidden_size=20, 
        hidden_layers=3, 
        dropout=0.5, 
        model='GRU',           
        early_stopping=30, 
        pred_threshold=0.5, 
        save=False
    ):
        model = self.get_model(
            load_saved_weights=False, model=model, batch_size=batch_size,
            hidden_size=hidden_size, hidden_layers=hidden_layers, 
            dropout=dropout
        )

        train_loader = DataLoader(
            dataset=self.tensor_dataset['Train'], batch_size=batch_size, 
            shuffle=False, collate_fn=lambda x:x
        )
        valid_loader = DataLoader(
            dataset=self.tensor_dataset['Valid'], batch_size=batch_size, 
            shuffle=False, collate_fn=lambda x:x
        )

        best_val_loss = np.inf
        best_model_param = None
        torch.manual_seed(42)

        # This loss criterion COMBINES Sigmoid and BCELoss. This is more 
        # numerically stable than using Sigmoid followed by BCELoss. As a 
        # result, the model does not use a Simgoid layer at the end. The model 
        # prediction output will not be bounded from (0, 1). In order to bound 
        # the model prediction, you must Sigmoid the model output.
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=decay
        )

        train_losses = np.zeros((epochs, self.n_targets))
        valid_losses = np.zeros((epochs, self.n_targets))
        train_scores = np.zeros((epochs, self.n_targets)) # acc score
        valid_scores = np.zeros((epochs, self.n_targets)) # acc score
        counter = 0 # for early stopping

        for epoch in range(epochs):
            train_loss = 0
            train_score = 0
            for i, batch in enumerate(train_loader):
                # each is a tuple of tensors
                inputs, targets, _ = tuple(zip(*batch))
                
                # format sequences
                formatted_seqs = self.format_sequences(inputs, targets)
                packed_padded_inputs, padded_targets = formatted_seqs
                targets = torch.cat(targets).float() # concatenate the tensors
                if torch.cuda.is_available():
                    packed_padded_inputs = packed_padded_inputs.cuda()
                    targets = targets.cuda()

                # Make predictions
                # for each patient, for each timestep, a prediction was made 
                # given the prev sequence history of that time step
                preds = model(packed_padded_inputs) 
                
                # unpad predictions based on target lengths
                preds = preds[padded_targets != self.pad_value] 
                
                # ensure preds shape matches targets shape
                preds = preds.reshape(-1, self.n_targets)

                # Calculate loss
                loss = criterion(preds, targets)
                loss = loss.mean(dim=0)
                train_loss += loss
                
                # Bound the model prediction
                preds = torch.sigmoid(preds)

                preds = preds > pred_threshold
                if torch.cuda.is_available():
                    preds = preds.cpu().detach().numpy()
                    targets = targets.cpu().detach().numpy()
                train_score += np.array([accuracy_score(targets[:, i], preds[:, i])
                                         for i in range(self.n_targets)])
                
                loss = loss.mean()
                loss.backward() # back propagation, compute gradients
                optimizer.step() # apply gradients
                optimizer.zero_grad() # clear gradients for next train

            train_losses[epoch] = train_loss.cpu().detach().numpy()/(i+1)
            train_scores[epoch] = train_score/(i+1)
            valid_losses[epoch], valid_scores[epoch] = self.validate(
                model, valid_loader, criterion
            )
            logging.info(f"Epoch: {epoch+1}, "
                         f"Train Loss: {train_losses[epoch].mean():.4f}, "
                         f"Valid Loss: {valid_losses[epoch].mean():.4f}, "
                         f"Train Acc: {train_scores[epoch].mean():.4f}, "
                         f"Valid Acc: {valid_scores[epoch].mean():.4f}")

            if valid_losses[epoch].mean() < best_val_loss:
                logging.info('Saving Best Model')
                best_val_loss = valid_losses[epoch].mean()
                best_model_param = model.state_dict()
                counter = 0

            # early stopping
            if counter > early_stopping: 
                train_losses = train_losses[:epoch+1]
                valid_losses = valid_losses[:epoch+1]
                train_scores = train_scores[:epoch+1]
                valid_scores = valid_scores[:epoch+1]
                break
            counter += 1

        if save:
            save_path = f"{self.output_path}/figures/rnn_train_performance"
            np.save(f"{save_path}/train_losses.npy", train_losses)
            np.save(f"{save_path}/valid_losses.npy", valid_losses)
            np.save(f"{save_path}/train_scores.npy", train_scores)
            np.save(f"{save_path}/valid_scores.npy", valid_scores)
            
            save_path = f'{self.output_path}/{model.name}_classifier'
            logging.info(f'Writing best model to {save_path}')
            torch.save(best_model_param, save_path)

        return model
        
    def _get_model_predictions(self, model, split, bound_pred=True):
        """
        Args: 
            bound_pred (bool): If True, bound the predictions by using Sigmoid 
                over the model output
        """
        loader = DataLoader(
            dataset=self.tensor_dataset[split], batch_size=10000, 
            shuffle=False, collate_fn=lambda x:x
        )
        pred_arr = np.empty([0, self.n_targets])
        index_arr = np.empty(0)
        for i, batch in enumerate(loader):
            inputs, targets, indices = tuple(zip(*batch))

            formatted_seqs = self.format_sequences(inputs, targets)
            packed_padded_inputs, padded_targets = formatted_seqs
            indices = torch.cat(indices).float()
            if torch.cuda.is_available():
                packed_padded_inputs = packed_padded_inputs.cuda()

            with torch.no_grad():
                preds = model(packed_padded_inputs)
            
            mask = padded_targets != self.pad_value
            preds = preds[mask].reshape(-1, self.n_targets)
            if bound_pred: 
                preds = torch.sigmoid(preds)

            if torch.cuda.is_available():
                preds = preds.cpu().detach().numpy()

            pred_arr = np.concatenate([pred_arr, preds])
            index_arr = np.concatenate([index_arr, indices])

        # garbage collection
        del preds, targets, packed_padded_inputs
        torch.cuda.empty_cache()
        
        return pred_arr, index_arr
        
    def get_model_predictions(self, model, split, calibrated=False, **kwargs):
        if self.preds[split].empty:
            pred_arr, index_arr = self._get_model_predictions(
                model, split, **kwargs
            )
            self.preds[split].loc[index_arr, self.target_events] = pred_arr
            if calibrated: # get the calibrated predictions
                self.preds[split] = self.calibrator.predict(self.preds[split])
                
        return self.preds[split]
    
    def convert_param_types(self, best_param):
        for param in ['batch_size', 'hidden_size', 'hidden_layers']:
            best_param[param] = int(best_param[param])
        best_param['model'] = 'LSTM' if best_param['model'] > 0.5 else 'GRU'
        return best_param

    def bo_evaluate(self, split='Valid', **params):
        params['epochs'] = 15
        params = self.convert_param_types(params)
        logging.info(f"Evaluating parameters: {params}")
        model = self.train_model(**params)
        
        # compute total mean auc score for all target types on the valid split
        pred_arr, index_arr = self._get_model_predictions(model, split)
        target = self.labels[split].loc[index_arr]
        auc_score = [roc_auc_score(target[target_event], pred_arr[:, i]) 
                     for i, target_event in enumerate(self.target_events)]
        return np.mean(auc_score)
    
    def bayesopt(self, alg='RNN', random_state=42):
        hyperparam_config = model_tuning_param[alg]
        optim_config = bayesopt_param[alg]
        return self._bayesopt(
            alg, self.bo_evaluate, hyperparam_config, optim_config, 
            random_state=random_state
        )
    
    def tune_and_train(
        self, 
        alg='RNN', 
        run_bayesopt=True, 
        run_training=True, 
        run_calibration=True,          
        calibrate_pred=True, 
        save_preds=True
    ):
        if run_bayesopt:
            best_param = self.bayesopt()
        else:
            filepath = f'{self.output_path}/best_params/{alg}_best_param.pkl'
            if not os.path.exists(filepath): 
                raise ValueError('Please run bayesian optimization for '
                                 f'{alg} to obtain best hyperparameters')
            with open(filepath, 'rb') as file: 
                best_param = pickle.load(file)

        if run_training:
            model = self.train_model(save=True, **best_param)
        else:
            del best_param['learning_rate']
            model = self.get_model(load_saved_weights=True, **best_param)
        
        if run_calibration:
            pred = pd.DataFrame(
                *self._get_model_predictions(model, 'Valid'), 
                columns=self.target_events
            )
            self.calibrator.calibrate(pred, self.labels['Valid'])
            self.calibrator.save_model(self.output_path, alg)
        else:
            self.calibrator.load_model(self.output_path, alg)
            
        # Get predictions
        for split in self.splits:
            self.get_model_predictions(model, split, calibrated=calibrate_pred)
        if save_preds: 
            save_predictions(
                self.preds, save_dir=f'{self.output_path}/predictions', 
                filename='rnn_predictions'
            )

###############################################################################
# Ensemble Model
###############################################################################
class TrainENS(Train):
    """Train the ensemble model 
    Find optimal weights via bayesopt
    """
    def __init__(self, X, Y, tag, output_path, preds=None):
        """
        Args:
            X (pd.DataFrame): table of input features
            Y (pd.DataFrame): table of target labels
            tag (pd.DataFrame): table containing partition names (e.g. Train, 
                Valid, etc) associated with each sample
            preds (dict): mapping of partition names (str) and their samples'  
                predictions by each algorithm (dict of str: pd.DataFrame) to be
                used by the ensemble model.
                e.g. {'Valid': {'LR': pred, 'XGB': pred, 'GRU': pred}
                      'Test': {'LR': pred, 'XGB': pred, 'GRU': pred}}
                If None, will train LR and XGB model using default hyperparameters
                and use their predictions for the ensemble model.
        """
        super().__init__(X, Y, tag, output_path)
        self.preds = preds
        if self.preds is None:
            logging.info('Predictions not provided. Training LR and XGB model '
                         'to be used for ENS model')
            train_ml = TrainML(X, Y, tag, output_path)
            for alg in ['LR', 'XGB']:
                model = train_ml.train_model(alg)
                for split in train_ml.splits: 
                    train_ml.predict(model, split, alg)
            self.preds = train_ml.preds
        else:
            self.models = list(preds['Test'].keys())
            # make sure each split in preds contains the same models
            assert all(list(preds[split].keys()) == self.models 
                       for split in self.splits)
        self.calibrator = IsotonicCalibrator(self.target_events)
        
    def bo_evaluate(self, split='Valid', **kwargs):
        ensemble_weights = [kwargs[alg] for alg in self.models]
        if not np.any(ensemble_weights): return 0 # weights are all zeros
        pred_prob = self.predict(split, ensemble_weights)
        # compute mean (macro-mean) of auroc scores of all target types
        return roc_auc_score(self.labels[split], pred_prob)
    
    def bayesopt(self, alg='ENS', random_state=42):
        hyperparam_config = model_tuning_param[alg]
        hyperparam_config = {alg: hyperparam_config[alg] for alg in self.models}
        optim_config = bayesopt_param[alg]
        return self._bayesopt(
            alg, self.bo_evaluate, hyperparam_config, optim_config, 
            random_state=random_state
        )
    
    def predict(self, split, ensemble_weights=None):
        Y = self.labels[split]
        # compute ensemble predictions by soft vote
        if ensemble_weights is None: ensemble_weights = [1,]*len(self.models)
        pred = [self.preds[split][alg] for alg in self.models]
        pred = np.average(pred, axis=0, weights=ensemble_weights)
        pred = pd.DataFrame(pred, index=Y.index, columns=Y.columns)
        return pred
    
    def store_prediction(self, calibrated=False):
        ensemble_weights = load_ensemble_weights(f'{self.output_path}/best_params')
        ensemble_weights = [ensemble_weights[alg] for alg in self.models]
        for split in self.splits:
            pred_prob = self.predict(split, ensemble_weights)
            if calibrated:
                pred_prob = self.calibrator.predict(pred_prob)
            self.preds[split]['ENS'] = pred_prob
            
    def tune_and_train(
        self, 
        alg='ENS', 
        run_bayesopt=True, 
        run_calibration=True, 
        calibrate_pred=True,
        random_state=42,
    ):
        if run_bayesopt:
            best_param = self.bayesopt(random_state=random_state)
        else:
            filepath = f'{self.output_path}/best_params/{alg}_best_param.pkl'
            if not os.path.exists(filepath): 
                raise ValueError('Please run bayesian optimization for '
                                 f'{alg} to obtain best hyperparameters')
            with open(filepath, 'rb') as file: 
                best_param = pickle.load(file)
        ensemble_weights = [best_param[alg] for alg in self.models]
        
        if run_calibration:
            self.calibrator.calibrate(
                self.predict('Valid', ensemble_weights), self.labels['Valid']
            )
            self.calibrator.save_model(self.output_path, alg)
        else:
            self.calibrator.load_model(self.output_path, alg)

        self.store_prediction(calibrated=calibrate_pred)

###############################################################################
# Baseline Models
###############################################################################
class TrainLASSO(TrainML):
    """Train LASSO model
    """
    def __init__(self, X, Y, tag, output_path, n_jobs=-1, target_event=None):
        """
        Args:
            X (pd.DataFrame): table of input features
            Y (pd.DataFrame): table of target labels
            tag (pd.DataFrame): table containing partition names (e.g. Train, 
        """
        self.target_event = target_event
        self.alg = 'LR'
        self.ci = AUCConfidenceInterval(output_path)
        super().__init__(
            X, Y, tag, output_path, n_jobs=n_jobs,
            custom_params={'LR': {'penalty': 'l1'}}
        )
        
    def grid_search(self, C_search_space=None):
        if self.target_event is None:
            err_msg = ('We currently only support tuning a LASSO model for '
                       'one target event. A target_event must be provided on '
                       'initialization.')
            raise NotImplementedError(err_msg)
            
        if C_search_space is None: 
            C_search_space = np.geomspace(0.000001, 1, 100)
        
        results = []
        for C in C_search_space:
            score, lower, upper, coef = self.gs_evaluate(C=C)
            n_features = len(coef)
            top_ten = coef[:10].round(3).to_dict()
            
            logging.info(f'Parameter C: {C:.2E}. '
                         f'Number of non-zero weighted features: {n_features}. '
                         f'AUROC Score: {score:.3f} ({lower:.3f}-{upper:.3f})\n'
                         f'Top 10 Features: {top_ten}')
            
            results.append([C, n_features, score, lower, upper])
        
        cols = ['C', 'n_feats', 'AUROC', 'AUROC_lower', 'AUROC_upper']
        results = pd.DataFrame(results, columns=cols)
        return results
            
    def gs_evaluate(self, split='Valid', **kwargs):
        model = self.train_model(
            alg=self.alg, save=False, target_event=self.target_event, **kwargs
        )
        pred_prob = self.predict(
            model, split=split, alg=self.alg, target_event=self.target_event, 
            store=False
        )
        
        # Get score
        score = roc_auc_score(self.labels[split][self.target_event], pred_prob)
        
        # Get 95% CI
        ci = self.ci.get_auc_confidence_interval(
            self.labels[split][self.target_event], pred_prob[self.target_event],
            store=False, verbose=False
        )
        lower, upper = ci['AUROC']
        
        # Get coefficients
        coef = self.get_coefficients(model, est_idx=0)
        return score, lower, upper, coef
            
    def get_coefficients(self, model, est_idx=None, non_zero=True):
        """Get coefficients of the features
        
        Args:
            est_idx (int): for multioutput model, which estimator's coefficient
                to extract. If None, defaults to the estimator corresponding to
                self.target_event if provided or the first estimator if not
            non_zero (bool): if True, get non-zero weighted coefficients only
        """
        if est_idx is None:
            est_idx = (0 if self.target_event is None
                       else self.target_events.index(self.target_event))
        estimator = model.estimators_[est_idx].calibrated_classifiers_[0].estimator
        coef = pd.Series(estimator.coef_[0], estimator.feature_names_in_)
        mask = coef != 0
        coef = coef[mask].sort_values(ascending=False)
        return coef
    
    def select_param(self, gs_results, verbose=True):
        """Select C that resulted in the least complex model (minimum number
        of features) whose upper CI AUROC is equal to or greater than the
        max AUROC score achieved
        """
        max_score = gs_results['AUROC'].max()
        mask = gs_results['AUROC_upper'] >= max_score
        min_feats = gs_results.loc[mask, 'n_feats'].min()
        best = gs_results.query('n_feats == @min_feats').iloc[-1]
        if verbose: 
            logging.info(f'Max AUROC Score = {max_score:.3f}. Selected:\n{best}')
        return best
    
    def tune_and_train(
        self,
        run_grid_search=True, 
        run_training=True,
        save=True,
        **gs_kwargs
    ):
        filepath = f'{self.output_path}/tables/grid_search.csv'
        if run_grid_search:
            gs_results = self.grid_search(**gs_kwargs)
            if save: gs_results.to_csv(filepath, index=False)
        else:
            gs_results = pd.read_csv(filepath)
            
        if run_training:
            best = self.select_param(gs_results)
            model = self.train_model(
                self.alg, save=save, filename='LASSO_classifier', C=best['C']
            )
            if save:
                filename = f'{self.output_path}/best_params/LASSO_best_param.pkl'
                with open(filename, 'wb') as file: 
                    pickle.dump({'C': best['C']}, file)

                for split in self.splits: 
                    self.predict(model, split, self.alg, store=True)
                filename = f'{self.output_path}/predictions/LASSO_predictions.pkl'
                with open(filename, 'wb') as file: 
                    pickle.dump(self.preds, file)
                    
class TrainSingleFeatureBaselineModel(Train):
    def __init__(
        self, 
        X,
        Y,
        tag,
        output_path, 
        base_col, 
        alg, 
        task_type='classification'
    ):
        super().__init__(X, Y, tag, output_path)
        bo_eval_funcs = {
            'classification': roc_auc_score, 
            'regression': mean_squared_error
        }
        self.bo_eval_func = bo_eval_funcs[task_type]
        self.col = base_col
        self.alg = alg
        self.task_type = task_type
        self.datasets = {split: X[self.col] for split, X in self.datasets.items()}

    def bo_evaluate(self, split='Valid', **kwargs):
        kwargs = self.convert_param_types(kwargs)
        model = self.train_model(**kwargs)
        if model is None: return 0
        pred = self.predict(model, split=split, ci=False)
        return self.bo_eval_func(self.labels[split], pred)

    def bayesopt(self, filename='', random_state=42, verbose=2):
        hyperparam_config = model_tuning_param[self.alg]
        optim_config = bayesopt_param[self.alg]
        return self._bayesopt(
            self.alg, self.bo_evaluate, hyperparam_config, optim_config, 
            filename=filename, random_state=random_state, verbose=verbose
        )
    
    def train_model(self, **param):
        raise NotImplementedError

    def predict(self, model, split='Test', ci=True, **kwargs):
        X, Y = self.datasets[split], self.labels[split]
        if ci:
            x_train, y_train = self.datasets['Train'], self.labels['Train']
             # output = (preds, ci_lower, ci_upper)
            output = model.predict_with_confidence_interval(
                X, x_train=x_train, y_train=y_train
            )
            return (pd.DataFrame(item, index=Y.index) for item in output)
        else:
            pred = model.predict(X)
            return pd.DataFrame(pred, index=Y.index)
        
class TrainLOESSModel(TrainSingleFeatureBaselineModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LOESS can only handle 10000 max observations before crashing during
        # prediction
        self.max_observations = 10000
        
    def train_model(self, **param):
        X, Y = self.datasets['Train'], self.labels['Train']
        if len(Y) > self.max_observations:
            Y = Y.sample(n=self.max_observations, random_state=42)
            X = X.loc[Y.index]
            
        model = LOESSModel(**param)
        try:
            model.fit(X, Y)
        except ValueError as e:
            if str(e) == "b'svddc failed in l2fit.'":
                logging.info('Caught error svddc failed in l2fit. Returning None')
                return None
            elif 'There are other near singularities as well' in str(e):
                logging.info('Encountered near singularities. Returning None')
                return None
            else: 
                raise e
                
        return model
    
class TrainPolynomialModel(TrainSingleFeatureBaselineModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_observations = 13000
        
    def convert_param_types(self, best_param):
        for param in ['degree', 'n_knots']:
            if param in best_param:
                best_param[param] = int(best_param[param])

        best_param = {
            'PLY': {k: v for k,v in best_param.items() 
                    if k not in model_tuning_param['LR']},
            'REG': {k: v for k,v in best_param.items() 
                    if k in model_tuning_param['LR']}
        }
        if self.task_type == 'regression': 
            best_param['REG']['alpha'] = best_param['REG'].pop('C')

        return best_param
        
    def train_model(self, **param):
        X, Y = self.datasets['Train'], self.labels['Train']
        model = PolynomialModel(
            alg=self.alg, task_type=self.task_type, **param
        )
        model.fit(X, Y)
        return model
