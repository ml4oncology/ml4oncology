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
"""
Module for training and tuning models

Tuner -> Trainer -> LASSOTrainer
      -> Ensembler
      -> BaselineTrainer -> LOESSModelTrainer
                         -> PolynomialModelTrainer

PLEASE TRY TO KEEP INHEIRTANCE LEVELS AT MAXIMUM OF 3
"""
from functools import partial
import copy

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch

from src import logger
from src.conf_int import ScoreConfidenceInterval
from src.config import model_tuning_param, bayesopt_param
from src.model import (
    IsotonicCalibrator, 
    LOESSModel, 
    Models, 
    PolynomialModel, 
)
from src.utility import load_pickle, save_pickle

torch.manual_seed(0)
np.random.seed(0)

###############################################################################
# Tune Models
###############################################################################
class Tuner:
    def __init__(self, X, Y, tag, output_path, task_type='C'):
        """
        Args:
            task_type (str): the type of machine learning task, either `C` for 
                classifcation or `R` for regression
        """
        self.n_features = X.shape[1]
        self.n_targets = Y.shape[1]
        self.target_events = Y.columns.tolist()
        
        self.ikns = tag['ikn']
        self.datasets = {split: x for split, x in X.groupby(tag['split'])}
        self.labels = {split: y for split, y in Y.groupby(tag['split'])}
        self.splits = list(self.labels.keys())
        
        self.task_type = task_type
        score_funcs = {
            'C': roc_auc_score, # greater is better
             # bayes-opt only maximizes the score, make it negative to minimize
            'R': lambda t, y: -mean_squared_error(t, y, squared=False) # smaller is better
        }
        self.score_func = score_funcs[self.task_type]
        
        self.output_path = output_path
        
        self.model_tuning_param = copy.deepcopy(model_tuning_param)
        self.bayesopt_param = copy.deepcopy(bayesopt_param)
    
    def bayesopt(
        self, 
        alg, 
        filename='', 
        random_state=42, 
        eval_kwargs=None,
        bopt_kwargs=None
    ):
        """Conduct bayesian optimization, a sequential search framework 
        for finding optimal hyperparameters using bayes theorem
        """
        if not filename: filename = f'{alg}_params'
        if eval_kwargs is None: eval_kwargs = {}
        if bopt_kwargs is None: bopt_kwargs = {}

        # set up
        hyperparam_config = self.model_tuning_param[alg]
        optim_config = self.bayesopt_param[alg]
        eval_func = partial(self._eval_func, alg=alg, **eval_kwargs)
        bo = BayesianOptimization(
            f=eval_func, 
            pbounds=hyperparam_config, 
            verbose=2,
            random_state=random_state, 
            **bopt_kwargs
        )
        
        # log the progress
        logger1 = JSONLogger(path=f'{self.output_path}/logs/{filename}-bayesopt.log')
        logger2 = ScreenLogger(verbose=2, is_constrained=False)
        for bo_logger in [logger1, logger2]:
            bo.subscribe(Events.OPTIMIZATION_START, bo_logger)
            bo.subscribe(Events.OPTIMIZATION_STEP, bo_logger)
            bo.subscribe(Events.OPTIMIZATION_END, bo_logger)
        
        # find the best hyperparameters
        bo.maximize(acq='ei', **optim_config)
        best_param = bo.max['params']
        best_param = self.convert_hyperparams(best_param)
        logger.info(f'Finished finding best hyperparameters for {alg}')
        logger.info(f'Best param: {best_param}')

        # save the best hyperparameters
        save_pickle(best_param, f'{self.output_path}/best_params', filename)

        return best_param
    
    def _eval_func(self, *args, **kwargs):
        """Evaluation function for bayesian optimization"""
        raise NotImplementedError
    
    def convert_hyperparams(self, best_param):
        """You can overwrite this to convert the hyperparmeters as desired
        """
        return best_param
    
###############################################################################
# Train Models
###############################################################################
class Trainer(Tuner):
    def __init__(self, X, Y, tag, output_path, task_type='C', **kwargs):
        """
        Args:
            X (pd.DataFrame): table of input features
            Y (pd.DataFrame): table of target labels
            tag (pd.DataFrame): table containing partition names (e.g. Train, 
                Valid, etc) associated with each sample
            task_type (str): the type of machine learning task, either `C` for 
                classifcation or `R` for regression
            **kwargs (dict): the parameters of MLModels
        """
        super().__init__(X, Y, tag, output_path, task_type=task_type)
        self.models = Models(task=task_type, **kwargs)
        self.preds = {}
        
    def run(
        self, 
        bayesopt=True, 
        train=True, 
        calibrate=True,
        save_preds=True,
        pred_filename=None,
        algs=None, 
        train_kwargs=None,
        bayes_kwargs=None,
        **kwargs
    ):
        """
        Args:
            algs (list): A sequence of algorithms (str) to train/tune. If None,
                train/tune all algorithms
            train_kwargs: keyword arguments fed into Trainer.train_model
            bayes_kwargs: keyword arguments fed into BayesianOptimization
        """
        if algs is None: algs = self.models
        if train_kwargs is None: train_kwargs = {}
        if bayes_kwargs is None: bayes_kwargs = {}

        for alg in algs:
            # Hyperparameter Tuning
            if bayesopt:
                best_param = self.bayesopt(alg, eval_kwargs=train_kwargs, **bayes_kwargs)
            else:
                filename = bayes_kwargs.get('filename', f'{alg}_params')
                best_param = load_pickle(
                    f'{self.output_path}/best_params', filename,
                    err_msg=(f'Please tune hyperparameters for {alg}')
                )
                logger.info(best_param)
            # NOTE: train_kwargs takes precedence if there are duplicate keys
            # for name, param in best_param.items():
            #     train_kwargs[name] = train_kwargs.get(name, param)

            # Model Training
            if train: 
                # NOTE: train_kwargs takes precedence if there are duplicate keys
                for key in best_param:
                    if key in train_kwargs:
                        del best_param[key]
                model = self.train_model(alg, calibrate=calibrate, **train_kwargs, **best_param)
                logger.info(f'{alg} training completed!')
            else:
                model = load_pickle(self.output_path, train_kwargs.get('filename', alg))

            # Prediction
            self.preds[alg] = {split: self.predict(model, split, alg, calibrated=calibrate) 
                               for split in self.splits}
                
        if save_preds:
            if pred_filename is None: pred_filename = 'all_preds'
            save_pickle(self.preds, f'{self.output_path}/preds', pred_filename)
            
    def train_model(self, alg, calibrate=True, save=True, filename='', **kwargs):
        if alg in ['RNN', 'NN', 'TCN']:
            model = self.train_dl_model(alg, **kwargs)
            if self.task_type == 'C' and calibrate:
                model = self.calibrate_dl_model(model, alg)
        elif alg in ['LR', 'RF', 'XGB', 'LGBM']:
            model = self.train_ml_model(alg, **kwargs)
            if self.task_type == 'C' and calibrate:
                model = self.calibrate_ml_model(model)
        
        if save:
            if not filename: filename = alg
            save_pickle(model, self.output_path, filename)
        return model
        
    def train_ml_model(self, alg, **kwargs):
        """Train machine learning models"""
        model = self.models.get_model(alg, **kwargs)
        X, Y = self.datasets['Train'], self.labels['Train']
        model.fit(X, Y)        
        return model
    
    def train_dl_model(
        self, 
        alg, 
        epochs=200, 
        batch_size=128, 
        early_stop_count=10, 
        early_stop_tol=1e-4,
        clip_gradients=False,
        save=False,
        save_checkpoints=False,
        **kwargs
    ):
        """Train deep learning models"""
        model = self.models.get_model(
            alg, self.n_features, self.n_targets, **kwargs
        )
        
        X_train, Y_train = self.datasets['Train'], self.labels['Train']
        X_valid, Y_valid = self.datasets['Valid'], self.labels['Valid']
        if alg in ['RNN', 'TCN']:
            train_dataset = self.transform_to_seq_dataset(X_train, Y_train)
            valid_dataset = self.transform_to_seq_dataset(X_valid, Y_valid)
            collate_fn = lambda x:x
        elif alg == 'NN':
            train_dataset = self.transform_to_tensor_dataset(X_train, Y_train)
            valid_dataset = self.transform_to_tensor_dataset(X_valid, Y_valid)
            def collate_fn(batch):
                feats, targets = zip(*batch)
                feats, targets = torch.stack(feats), torch.stack(targets)
                if model.use_gpu: targets = targets.cuda()
                return feats, targets
            
        loader_params = dict(batch_size=batch_size, collate_fn=collate_fn)
        train_loader = DataLoader(dataset=train_dataset, **loader_params)
        valid_loader = DataLoader(dataset=valid_dataset, **loader_params)
        
        best_val_loss = prev_val_loss = np.inf
        best_model_weights = None
        early_stop_counter = 0 
        perf = {} # performance scores
        
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(model.optimizer, T_0=10)
        for epoch in range(epochs):
            model.train() # activate dropout
            train_loss = 0
            for i, batch in enumerate(train_loader):
                model.optimizer.zero_grad() # clear gradients
                if alg in ['RNN', 'TCN']:
                    preds, targets, _ = model.predict(
                        batch, grad=True, bound_pred=False
                    )
                elif alg == 'NN':
                    feats, targets = batch
                    preds = model.predict(feats, grad=True, bound_pred=False)

                loss = model.criterion(preds, targets)
                loss = loss.mean(dim=0)
                train_loss += loss
                
                if self.task_type == 'C':
                    preds = torch.sigmoid(preds) # bound the model prediction
                
                loss = loss.mean()
                loss.backward() # back propagation, compute gradients
                if clip_gradients: model.clip_gradients()
                model.optimizer.step() # apply gradients
            # lr_scheduler.step()
            
            model.eval() # deactivates dropout
            valid_loss = self._validate_dl_model(model, alg, valid_loader)
            if model.use_gpu: train_loss = train_loss.cpu().detach()
            perf[epoch] = {
                'Train Loss': train_loss / (i + 1),
                'Valid Loss': valid_loss
            }
            msg = [f'{k}: {v.mean():.4f}' for k, v in perf[epoch].items()]
            logger.info(f"Epoch {epoch}, {(', ').join(msg)}")
            
            # save best model so far
            cur_val_loss = valid_loss.mean()
            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                best_model_weights = model.state_dict()
                early_stop_counter = 0
                
                if save_checkpoints:
                    save_path = f"{self.output_path}/train_perf/{alg}-checkpoint"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_weights,
                        'optimizer_state_dict': model.optimizer.state_dict()
                    }, save_path)

            # early stopping
            if ((early_stop_counter > early_stop_count) or 
                (prev_val_loss - cur_val_loss < early_stop_tol)): 
                break
            early_stop_counter += 1
            prev_val_loss = cur_val_loss
            
        if save:
            if self.n_targets == 1: alg += f'_{self.target_events[0]}'
            save_path = f"{self.output_path}/train_perf"
            save_pickle(perf, save_path, f"{alg}_perf")
            save_path = f'{self.output_path}/{alg}'
            torch.save(best_model_weights, save_path)

        model.load_weights(best_model_weights)
        return model
    
    def calibrate_ml_model(self, model):
        X, Y = self.datasets['Valid'], self.labels['Valid']
        calib_clfs = []
        for i, clf in enumerate(model.model.estimators_):
            clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
            clf.fit(X, Y.iloc[:, i])
            calib_clfs.append(clf)
        model.model.estimators_ = calib_clfs
        return model
    
    def calibrate_dl_model(self, model, alg):
        calibrator = IsotonicCalibrator(self.target_events)
        pred = self.predict(model, 'Valid', alg, calibrated=False)
        calibrator.calibrate(pred, self.labels['Valid'])
        calibrator.save_model(self.output_path, alg)
        return model
        
    def predict(self, model, split, alg, calibrated=True):
        if self.task_type == 'R': calibrated = False

        X, Y = self.datasets[split], self.labels[split]
        if alg in ['RNN', 'TCN']:
            pred = self._seq_predict(model, split)
        elif alg == 'NN': 
            pred = self._nn_predict(model, X)
        else: 
            pred = model.predict(X)
            
        # make your life easier by ensuring pred and Y have same data format
        pred = pd.DataFrame(pred, index=Y.index, columns=Y.columns)
        
        if alg in ['RNN', 'NN', 'TCN'] and calibrated:
            calibrator = IsotonicCalibrator(self.target_events)
            calibrator.load_model(self.output_path, alg)
            pred = calibrator.predict(pred)
            
        return pred
    
    def _seq_predict(self, model, split):
        X, Y = self.datasets[split], self.labels[split]
        dataset = self.transform_to_seq_dataset(X, Y)
        
        # deactivates dropout
        model.eval()
        
        loader = DataLoader(
            dataset=dataset, batch_size=10000, collate_fn=lambda x:x
        )
        pred_arr, index_arr = np.empty([0, self.n_targets]), np.empty(0)
        bound_pred = self.task_type == 'C'
        for i, batch in enumerate(loader):
            preds, targets, indices = model.predict(batch, grad=False, bound_pred=bound_pred)
            if model.use_gpu: preds = preds.cpu()
            pred_arr = np.concatenate([pred_arr, preds])
            index_arr = np.concatenate([index_arr, indices])
        
        # format to match with the other algorithm's prediction outputs
        pred = pd.DataFrame(pred_arr, index=index_arr)
        pred = pred.loc[self.labels[split].index].to_numpy()
        
        # garbage collection
        torch.cuda.empty_cache()
        return pred
    
    def _nn_predict(self, model, X):
        model.eval() # deactivates dropout
        bound_pred = self.task_type == 'C'
        pred = model.predict(X, grad=False, bound_pred=bound_pred)
        if model.use_gpu: pred = pred.cpu()
        return pred
    
    def _validate_dl_model(self, model, alg, loader):
        total_loss = 0
        for i, batch in enumerate(loader):
            if alg in ['RNN', 'TCN']:
                preds, targets, _ = model.predict(
                    batch, grad=False, bound_pred=False
                )
            elif alg == 'NN':
                feats, targets = batch
                preds = model.predict(feats, grad=False, bound_pred=False)
            loss = model.criterion(preds, targets)
            loss = loss.mean(dim=0)
            total_loss += loss
        if model.use_gpu: total_loss = total_loss.cpu().detach()
        return total_loss / (i + 1)
    
    def _eval_func(self, alg, split='Valid', **kwargs):
        """Evaluation function for bayesian optimization
        
        Returns:
            Either the mean (macro-mean) of 
                1. auroc scores
                2. root mean squared error
            of all target types
        """
        kwargs = self.convert_hyperparams(kwargs)
        try:
            model = self.train_model(alg, calibrate=False, save=False, **kwargs)
        except Exception as e:
            raise e
            logger.warning(e)
            return -1e9
        pred = self.predict(model, split, alg, calibrated=False)
        if pred.isnull().any().any():
            # TODO: figure out how to prevent this
            logger.warning('Invalid prediction - contains NaNs')
            return -1e9
        return self.score_func(self.labels[split], pred)
    
    def convert_hyperparams(self, params):
        cat_param_choices = np.geomspace(start=16, stop=4096, num=9)
        int_params = [
            'bagging_freq', 'hidden_layers', 'kernel_size', 'max_depth',
            'min_child_weights', 'min_data_in_leaf', 'n_estimators', 'num_leaves'
        ]
        for param, value in params.items():
            if param in int_params: params[param] = int(value)
            if param == 'model': params[param] = 'LSTM' if value > 0.5 else 'GRU'
            if param == 'optimizer': params[param] = 'adam' if value > 0.5 else 'sgd'
            if (param == 'batch_size' or 
                param.startswith('hidden_size') or 
                params.startswith('num_channel')):
                idx = abs(cat_param_choices - value).argmin()
                params[param] = round(cat_param_choices[idx])
                
        return params
    
    def transform_to_seq_dataset(self, X, Y):
        ikns = self.ikns[X.index]
        try:
            X = X.astype(float)
        except Exception as e:
            logger.warning(f'Could not convert to float. Please check your input X')
            return None
        X_mapping = {ikn: group for ikn, group in X.groupby(ikns)}
        Y_mapping = {ikn: group for ikn, group in Y.groupby(ikns)}
        return SeqData(
            X_mapping=X_mapping, Y_mapping=Y_mapping, ids=ikns.unique()
        )
    
    def transform_to_tensor_dataset(self, X, Y):
        try:
            X = X.astype(float)
        except Exception as e:
            logger.warning(f'Could not convert to float. Please check your input X')
            return None
        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        Y = torch.tensor(Y.to_numpy(), dtype=torch.float32)
        return TensorDataset(X, Y)
    
class SeqData(TensorDataset):
    def __init__(self, X_mapping, Y_mapping, ids):
        self.X_mapping = X_mapping
        self.Y_mapping = Y_mapping
        self.ids = ids
                
    def __getitem__(self, index):
        sample = self.ids[index]
        X, Y = self.X_mapping[sample], self.Y_mapping[sample]
        features_tensor = torch.Tensor(X.to_numpy())
        target_tensor = torch.Tensor(Y.to_numpy())
        indices_tensor = torch.Tensor(Y.index)
        return features_tensor, target_tensor, indices_tensor
    
    def __len__(self):
        return len(self.ids)
    
###############################################################################
# Ensemble Models
###############################################################################
class Ensembler(Tuner):
    """Ensemble the model predictions
    Find optimal weights via bayesopt
    """
    def __init__(self, X, Y, tag, output_path, preds=None, task_type='C'):
        """
        Args:
            X (pd.DataFrame): table of input features
            Y (pd.DataFrame): table of target labels
            tag (pd.DataFrame): table containing partition names (e.g. Train, 
                Valid, etc) associated with each sample
            preds (dict): mapping of algorithms (str) and their predictions for 
                each partition (dict of str: pd.DataFrame) to be used by the 
                ensemble model.
                e.g. {'LR': {'Valid': pred, 'Test': pred}
                      'RNN': {'Valid': pred, 'Test': pred}}
            task_type (str): the type of machine learning task, either `C` for 
                classifcation or `R` for regression
        """
        super().__init__(X, Y, tag, output_path, task_type=task_type)
        self.preds = preds
        if self.preds is None: 
            raise ValueError('Please provide the predictions')
        self.models = list(preds.keys())
        self.calibrator = IsotonicCalibrator(self.target_events)
        for alg in list(self.model_tuning_param['ENS']): 
            if alg not in self.models: del self.model_tuning_param['ENS'][alg]
            
    def run(self, bayesopt=True, calibrate=True, random_state=42):
        if bayesopt:
            best_param = self.bayesopt(alg='ENS', random_state=random_state)
        else:
            best_param = load_pickle(
                f'{self.output_path}/best_params', 'ENS_params',
                err_msg='Please tune hyperparameters for the ensemble model'
            )
        ensemble_weights = [best_param[alg] for alg in self.models]
        
        if self.task_type == 'C':
            if calibrate:
                pred = self.predict('Valid', ensemble_weights)
                self.calibrator.calibrate(pred, self.labels['Valid'])
                self.calibrator.save_model(self.output_path, 'ENS')
            else:
                self.calibrator.load_model(self.output_path, 'ENS')

        self.store_prediction(ensemble_weights)
        
    def predict(self, split, ensemble_weights=None):
        Y = self.labels[split]
        # compute ensemble predictions by soft vote
        if ensemble_weights is None: ensemble_weights = [1,]*len(self.models)
        pred = [self.preds[alg][split] for alg in self.models]
        pred = np.average(pred, axis=0, weights=ensemble_weights)
        pred = pd.DataFrame(pred, index=Y.index, columns=Y.columns)
        return pred
    
    def store_prediction(self, ensemble_weights):
        self.preds['ENS'] = {}
        for split in self.splits:
            pred_prob = self.predict(split, ensemble_weights)
            if self.task_type == 'C':
                pred_prob = self.calibrator.predict(pred_prob)
            self.preds['ENS'][split] = pred_prob
        
    def _eval_func(self, alg='ENS', split='Valid', **kwargs):
        """Evaluation function for bayesian optimization"""
        ensemble_weights = [kwargs[alg] for alg in self.models]
        if not np.any(ensemble_weights): return 0 # weights are all zeros
        pred = self.predict(split, ensemble_weights)
        return self.score_func(self.labels[split], pred)
    
###############################################################################
# Baseline Models
###############################################################################
class LASSOTrainer(Trainer):
    """Trains LASSO model
    NOTE: only classification is currently supported
    """
    def __init__(self, X, Y, tag, output_path, target_event=None):
        """
        Args:
            X (pd.DataFrame): table of input features
            Y (pd.DataFrame): table of target labels
            tag (pd.DataFrame): table containing partition names
        """
        self.target_event = target_event
        self.ci = ScoreConfidenceInterval(
            output_path, score_funcs={'AUROC': roc_auc_score}
        )
        # To be used to train the final model 
        self._labels = {split: y for split, y in Y.groupby(tag['split'])}
        super().__init__(X, Y[[target_event]], tag, output_path, task_type='C')
        
    def run(
        self,
        grid_search=True, 
        train=True,
        save=True,
        **gs_kwargs
    ):
        filepath = f'{self.output_path}/tables/grid_search.csv'
        if grid_search:
            gs_results = self.grid_search(**gs_kwargs)
            if save: gs_results.to_csv(filepath, index=False)
        else:
            gs_results = pd.read_csv(filepath)
            
        best = self.select_param(gs_results)
        params = {'inv_reg_strength': best['C']}
        self.labels = self._labels
        
        if train:
            model = self.train_model('LR', save=save, filename='LASSO', **params)
        else:
            model = load_pickle(self.output_path, 'LASSO')
            
        self.preds['LASSO'] = {split: self.predict(model, split, 'LR') 
                               for split in self.splits}
        if save:
            save_pickle(params, f'{self.output_path}/best_params', 'LASSO_params')
            save_pickle(self.preds, f'{self.output_path}/preds', 'LASSO_preds')
    
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
            output = self.gs_evaluate(inv_reg_strength=C)
            score, lower, upper, coef, intercept = output
            n_features = len(coef)
            top_ten = coef[:10].round(3).to_dict()
            top_ten['intercept'] = intercept
            
            msg = (f'Parameter C: {C:.2E}. '
                   f'Number of non-zero weighted features: {n_features}. '
                   f'AUROC Score: {score:.3f} ({lower:.3f}-{upper:.3f})\n'
                   f'Top 10 Features: {top_ten}')
            logger.info(msg)
            
            results.append([C, n_features, score, lower, upper])
        
        cols = ['C', 'n_feats', 'AUROC', 'AUROC_lower', 'AUROC_upper']
        results = pd.DataFrame(results, columns=cols)
        return results
            
    def gs_evaluate(self, split='Valid', **kwargs):
        model = self.train_model(
            alg='LR', save=False, penalty='l1', n_jobs=-1, **kwargs
        )
        pred_prob = self.predict(model, split=split, alg='LR')
        
        # Get score
        score = roc_auc_score(self.labels[split][self.target_event], pred_prob)
        
        # Get 95% CI
        ci = self.ci.get_score_confidence_interval(
            self.labels[split][self.target_event], pred_prob[self.target_event],
            store=False, verbose=False
        )
        lower, upper = ci['AUROC']
        
        # Get coefficients and intercept
        coef = self.get_coefficients(model)
        intercept = self.get_intercept(model)
        return score, lower, upper, coef, intercept
            
    def get_coefficients(self, model, estimator_idx=None):
        """Get coefficients of the features

        Args:
            estimator_idx (int): which estimator's coefficients to retrieve for
                a multioutput model trained on different target events
        """
        i = 0 if estimator_idx is None else estimator_idx
        estimator = model.model.estimators_[i].calibrated_classifiers_[0].estimator
        coef = pd.Series(estimator.coef_[0], estimator.feature_names_in_)
        mask = coef != 0
        coef = coef[mask].sort_values(ascending=False)
        return coef
    
    def get_intercepts(self, model, estimator_idx=None):
        """Get the intercept of the linear model

        Args:
            estimator_idx (int): which estimator's coefficients to retrieve for
                a multioutput model trained on different target events
        """
        i = 0 if estimator_idx is None else estimator_idx
        estimator = model.model.estimators_[i].calibrated_classifiers_[0].estimator
        return estimator.intercept_[0]
    
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
            logger.info(f'Max AUROC Score = {max_score:.3f}. Selected:\n{best}')
        return best
            
class BaselineTrainer(Tuner):
    def __init__(self, X, Y, tag, output_path, base_col, alg, **kwargs):
        super().__init__(X, Y, tag, output_path, **kwargs)
        self.col = base_col
        self.datasets = {split: X[self.col] for split, X in self.datasets.items()}
        self.alg = alg
        self.preds = {alg: {}}
        self.preds_ci = {alg: {}}

    def run(self, bayesopt=True, train=True, save=True):
        if bayesopt:
            best_param = self.bayesopt(alg=self.alg)
        else:
            best_param = load_pickle(
                f'{self.output_path}/best_params', f'{self.alg}_params',
                err_msg=f'Please tune hyperparameters for the {self.alg} model'
            )
            logger.info(best_param)
        
        if train:
            model = self.train_model(save=save, **best_param)
        else:
            model = load_pickle(self.output_path, self.alg)

        for split in self.splits:
            Y_preds, Y_preds_min, Y_preds_max = self.predict(model, split=split)
            self.preds[self.alg][split] = Y_preds
            self.preds_ci[self.alg][split] = (Y_preds_min, Y_preds_max)

        if save:
            save_dir = f'{self.output_path}/preds'
            save_pickle(self.preds, save_dir, f'{self.alg}_preds')
            save_pickle(self.preds_ci, save_dir, f'{self.alg}_preds_ci')

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
    
    def _eval_func(self, alg, split='Valid', **kwargs):
        """Evaluation function for bayesian optimization"""
        kwargs = self.convert_hyperparams(kwargs)
        model = self.train_model(**kwargs)
        if model is None: return {'C': 0, 'R': -1e10}[self.task_type]
        pred = self.predict(model, split=split, ci=False)
        return self.score_func(self.labels[split], pred)
    
    def model_to_table(self, model, base_vals, extra_info, split='Test'):
        """Save the model as a threshold table so it's transferrable from ICES
        
        Args:
            base_vals (pd.Series): the unnormalized baseline values with the 
                `self.base_col` name
        """
        preds = self.predict(model, split=split, ci=False)
        idxs = preds.index
        ikns = self.ikns.loc[idxs]
        base_vals = base_vals.loc[idxs].round(1)
        extra_info = extra_info.loc[idxs]
        df = pd.concat([base_vals, preds, ikns, extra_info], axis=1)

        # Assign bins to the baseline values
        # Combine bins with less than 10 unique patients
        tmp = df.groupby(self.base_col)['ikn'].unique()
        assert all(tmp.index == sorted(tmp.index))
        bins, seen = list(), set()
        for base_val, ikns in tmp.items():
            seen.update(ikns)
            if len(seen) > 10:
                bins.append(base_val)
                seen = set()
        df[self.base_col] = pd.cut(df[self.base_col], bins=bins)

        # Aggregate stats in each bin
        cols = extra_info.columns.tolist() + preds.columns.tolist()
        df = df.groupby(self.base_col).agg({
            'ikn': 'nunique',
            **{col: 'mean' for col in cols}
        }).round(3)

        return df
    
    def train_model(self, **param):
        raise NotImplementedError
        

class LOESSModelTrainer(BaselineTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LOESS can only handle 10000 max observations before crashing during
        # prediction
        self.max_observations = 10000
        
    def train_model(self, save=False, **param):
        X, Y = self.datasets['Train'], self.labels['Train']
        if len(Y) > self.max_observations:
            Y = Y.sample(n=self.max_observations, random_state=42)
            X = X.loc[Y.index]
            
        model = LOESSModel(**param)
        try:
            model.fit(X, Y)
        except ValueError as e:
            # Ref: https://github.com/has2k1/scikit-misc/issues/9
            if str(e) == "b'svddc failed in l2fit.'":
                logger.info('Caught error svddc failed in l2fit. Returning None')
                return None
            elif 'There are other near singularities as well' in str(e):
                logger.info('Encountered near singularities. Returning None')
                return None
            else: 
                raise e

        if save: save_pickle(model, self.output_path, self.alg)
        return model
    

class PolynomialModelTrainer(BaselineTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_model(self, save=False, **param):
        X, Y = self.datasets['Train'], self.labels['Train']
        model = PolynomialModel(
            alg=self.alg, task_type=self.task_type, **param
        )
        model.fit(X, Y)
        if save: save_pickle(model, self.output_path, self.alg)
        return model
        
    def convert_hyperparams(self, best_param):
        for param in ['degree', 'n_knots']:
            if param in best_param:
                best_param[param] = int(best_param[param])

        best_param = {
            'PLY': {k: v for k,v in best_param.items() 
                    if k not in self.model_tuning_param['LR']},
            'REG': {k: v for k,v in best_param.items() 
                    if k in self.model_tuning_param['LR']}
        }
        inv_reg_strength = best_param['REG'].pop('inv_reg_strength')
        if self.task_type == 'C':
            best_param['REG']['C'] = inv_reg_strength
        elif self.task_type == 'R': 
            best_param['REG']['alpha'] = 1 / inv_reg_strength

        return best_param
