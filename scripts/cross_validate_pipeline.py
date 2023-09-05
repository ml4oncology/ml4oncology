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
Script to run an alternate training pipeline 
- 5-fold cross validation is incorporated in the bayes search
"""
import argparse
import os
import sys
sys.path.append(os.getcwd())

from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

from src import logger
from src.config import root_path, death_folder, split_date
from src.evaluate import EvaluateClf
from src.prep_data import PrepDataEDHD
from src.train import TrainML, TrainENS, TrainRNN
from src.utility import initialize_folders, load_pickle, save_pickle

class TrainMLCV(TrainML):
    def __init__(self, X, Y, tag, prep_data, output_path, task_type='C'):
        super().__init__(X, Y, tag, output_path, task_type=task_type)
        self.ikns = tag['ikn']
        self.prep_data = prep_data
        
        self.orig_datasets = self.datasets.copy()
        self.orig_labels = self.labels.copy()
        self.orig_datasets['Development'] = X[tag['cohort'] == 'Development']
        self.orig_labels['Development'] = Y[tag['cohort'] == 'Development']
        
        self.datasets = {}
        self.labels = {}
        
    def tune_and_train(self, run_bayesopt=True, algs=None, bayes_kwargs=None, **kwargs):
        if run_bayesopt: 
            if bayes_kwargs is None: bayes_kwargs = {}
            if algs is None: algs = self.ml.models
            for alg in algs:
                best_param = self.bayesopt(alg, **bayes_kwargs)
        
        self.datasets = self.orig_datasets
        self.labels = self.orig_labels
        prep = self.prep_data()
        prep_kwargs = dict(verbose=False, ohe_kwargs={'verbose': False})
        self.datasets['Train'] = prep.transform_data(self.datasets['Train'].copy(), **prep_kwargs)
        cols = self.datasets['Train'].columns
        self.datasets['Valid'] = prep.transform_data(self.datasets['Valid'].copy(), **prep_kwargs)[cols]
        self.datasets['Test'] = prep.transform_data(self.datasets['Test'].copy(), **prep_kwargs)[cols]
        super().tune_and_train(run_bayesopt=False, algs=algs, **kwargs)
        
    def _eval_func(self, alg, **kwargs):
        kwargs = self.convert_param_types(kwargs)
        X, Y = self.orig_datasets['Development'], self.orig_labels['Development']
        ikns = self.ikns[X.index]
        
        scores = []
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idxs, valid_idxs) in enumerate(kf.split(X, Y.any(axis=1), ikns)):
            X_train, X_valid = X.iloc[train_idxs], X.iloc[valid_idxs]
            Y_train, Y_valid = Y.iloc[train_idxs], Y.iloc[valid_idxs]
            
            prep = self.prep_data()
            X_train = prep.transform_data(X_train.copy(), verbose=False, ohe_kwargs={'verbose': False})
            X_valid = prep.transform_data(X_valid.copy(), verbose=False, ohe_kwargs={'verbose': False})
            X_valid = X_valid[X_train.columns] # make sure columns match
            
            self.datasets['Train'], self.datasets['Valid'] = X_train, X_valid
            self.labels['Train'], self.labels['Valid'] = Y_train, Y_valid
            
            model = self.train_model(alg, save=False, **kwargs)
            pred = self.predict(model, 'Valid', alg, store=False)
            scores.append(self.score_func(self.labels['Valid'], pred))
            
        return np.mean(scores)
    
class TrainRNNCV(TrainRNN):
    def __init__(self, X, Y, tag, prep_data, output_path, task_type='C'):
        super().__init__(X, Y, tag, output_path, task_type=task_type)
        self.prep_data = prep_data
        
        self.orig_datasets = self.datasets.copy()
        self.orig_labels = self.labels.copy()
        self.orig_datasets['Development'] = X[tag['cohort'] == 'Development']
        self.orig_labels['Development'] = Y[tag['cohort'] == 'Development']
        
        self.tensor_datasets = {}
        self.labels = {}
        
    def tune_and_train(self, run_bayesopt=True, **kwargs):
        if run_bayesopt: 
            best_param = self.bayesopt(alg='RNN')
        
        self.datasets = self.orig_datasets
        self.labels = self.orig_labels
        prep = self.prep_data()
        prep_kwargs = dict(verbose=False, ohe_kwargs={'verbose': False})
        X_train = prep.transform_data(self.datasets['Train'].copy(), **prep_kwargs)
        cols = X_train.columns
        X_valid = prep.transform_data(self.datasets['Valid'].copy(), **prep_kwargs)[cols]
        X_test = prep.transform_data(self.datasets['Test'].copy(), **prep_kwargs)[cols]
        self.n_features = len(cols)
        
        self.tensor_datasets['Train'] = self.to_tensor(X_train, self.labels['Train'])
        self.tensor_datasets['Valid'] = self.to_tensor(X_valid, self.labels['Valid'])
        self.tensor_datasets['Test'] = self.to_tensor(X_test, self.labels['Test'])
        super().tune_and_train(run_bayesopt=False, **kwargs)
        
    def _eval_func(self, alg='RNN', **params):
        params['epochs'] = 15
        params = self.convert_param_types(params)
        X, Y = self.orig_datasets['Development'], self.orig_labels['Development']
        ikns = self.ikns[X.index]
        
        result = []
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idxs, valid_idxs) in enumerate(kf.split(X, Y.any(axis=1), ikns)):
            X_train, X_valid = X.iloc[train_idxs], X.iloc[valid_idxs]
            Y_train, Y_valid = Y.iloc[train_idxs], Y.iloc[valid_idxs]
            
            prep = self.prep_data()
            X_train = prep.transform_data(X_train.copy(), verbose=False, ohe_kwargs={'verbose': False})
            X_valid = prep.transform_data(X_valid.copy(), verbose=False, ohe_kwargs={'verbose': False})
            X_valid = X_valid[X_train.columns] # make sure columns match
            self.tensor_datasets['Train'] = self.to_tensor(X_train, Y_train)
            self.tensor_datasets['Valid'] = self.to_tensor(X_valid, Y_valid)
            self.labels['Train'], self.labels['Valid'] = Y_train, Y_valid
            self.n_features = X_train.shape[1]
        
            model = self.train_model(**params)
            # compute total mean score for all target types on the valid split
            pred_arr, index_arr = self._get_model_predictions(model, 'Valid')
            target = self.labels['Valid'].loc[index_arr]
            scores = [self.score_func(target[target_event], pred_arr[:, i])
                      for i, target_event in enumerate(self.target_events)]
            result.append(np.mean(scores))
            
        return np.mean(result)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adverse-event', type=str, required=True, choices=['ACU', 'CAN', 'CYTO', 'DEATH'])
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--train-ml', action='store_true')
    parser.add_argument('--train-rnn', action='store_true')
    parser.add_argument('--train-ensemble', action='store_true')
    parser.add_argument('--run-bayesopt', action='store_true')
    msg = 'within number of days an event to occur after a treatment session to be a target. Only for ACU.'
    parser.add_argument('--days', type=int, required=False, default=30, help=msg)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    adverse_event = args.adverse_event
    output_path = args.output_path
    evaluate = args.evaluate
    train_ml = args.train_ml
    train_rnn = args.train_rnn
    train_ensemble = args.train_ensemble
    run_bayesopt = args.run_bayesopt
    days = args.days
    
    if adverse_event == 'DEATH':
        if output_path is None: output_path = f'{root_path}/{death_folder}/models'
        initialize_folders(output_path)
        prep_data = lambda: PrepDataEDHD(adverse_event='death', target_keyword='Mortality')
        prep = prep_data()
        df = prep.get_data(missing_thresh=80, treatment_intents=['P'], verbose=False)
    else:
        raise NotImplementedError(f'Sorry, {adverse_event} is not supported yet')
        
    X, Y, tag = prep.split_and_transform_data(
        df, 
        one_hot_encode=False, 
        clip=False, 
        normalize=False, 
        remove_immediate_events=True, 
        split_date=split_date, 
        verbose=True
    )
    
    if train_ml:
        mlcv = TrainMLCV(X, Y, tag, prep_data, output_path)
        mlcv.tune_and_train(run_bayesopt=run_bayesopt, run_training=True, save_preds=True)
    
    if train_rnn:
        rnncv = TrainRNNCV(X, Y, tag, prep_data, output_path)
        rnncv.tune_and_train(run_bayesopt=run_bayesopt, run_training=True, run_calibration=True, save_preds=True)

    preds = load_pickle(f'{output_path}/preds', 'ML_preds')
    preds_rnn = load_pickle(f'{output_path}/preds', 'RNN_preds')
    for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
    train_ens = TrainENS(X, Y, tag, output_path, preds)
    train_ens.tune_and_train(run_bayesopt=train_ensemble, run_calibration=True, calibrate_pred=True)
    
    if evaluate:
        preds, labels = train_ens.preds.copy(), train_ens.labels.copy()
        eval_models = EvaluateClf(output_path, preds, labels)
        eval_models.get_evaluation_scores(display_ci=True, load_ci=False, save_ci=True)

if __name__ == '__main__':
    main()
