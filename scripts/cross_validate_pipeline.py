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
Script to run an alternate training pipeline 
- 5-fold cross validation is incorporated in the bayes search
"""
import argparse
import os
import sys
sys.path.append(os.getcwd())

from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pandas as pd

from src import logger
from src.config import root_path, death_folder, split_date
from src.conf_int import ScoreConfidenceInterval
from src.evaluate import CLF_SCORE_FUNCS, EvaluateClf
from src.model import IsotonicCalibrator
from src.prep_data import PrepDataEDHD
from src.train import Trainer, Ensembler
from src.utility import initialize_folders, load_pickle, save_pickle, twolevel

class TrainerCV(Trainer):
    def __init__(self, X, Y, tag, prep_data, output_path, task_type='C'):
        super().__init__(X, Y, tag, output_path, task_type=task_type)
        self.orig_datasets = self.datasets.copy()
        self.orig_labels = self.labels.copy()
        self.orig_datasets['Development'] = X[tag['cohort'] == 'Development']
        self.orig_labels['Development'] = Y[tag['cohort'] == 'Development']
        
        self.datasets = {}
        self.labels = {}
        
        self.prep_data = prep_data
        self.prep_kwargs = dict(verbose=False, ohe_kwargs={'verbose': False})
        
        self.kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        
    def run(self, bayesopt=True, evaluate=True, algs=None, bayes_kwargs=None, **kwargs):
        if algs is None: algs = self.models
        
        # run 5-fold cross validation bayesopt
        if bayesopt: 
            if bayes_kwargs is None: bayes_kwargs = {}
            for alg in algs: 
                best_param = self.bayesopt(alg, **bayes_kwargs)
                
        # get 5-fold cross validation scores
        if evaluate:
            for alg in algs:
                best_param = load_pickle(
                    f'{self.output_path}/best_params', f'{alg}_params',
                    err_msg=(f'Please tune hyperparameters for {alg}')
                )
                models, preds = self.cross_validate(alg, **best_param)
                save_pickle(preds, f'{self.output_path}/preds/cv', f'{alg}_cv_preds')
                # preds = load_pickle(f'{self.output_path}/preds/cv', f'{alg}_cv_preds')
                labels = [self.orig_labels['Development'].loc[pred.index] for pred in preds]
                cv_scores = get_cv_scores(preds, labels, self.output_path)
                cv_scores.to_csv(f'{self.output_path}/tables/cv/{alg}_cv_scores.csv')
                        
        # return back to original for final model training
        self.datasets = self.orig_datasets
        self.labels = self.orig_labels
        prep = self.prep_data()
        self.datasets['Train'] = prep.transform_data(self.datasets['Train'].copy(), **self.prep_kwargs)
        cols = self.datasets['Train'].columns
        self.datasets['Valid'] = prep.transform_data(self.datasets['Valid'].copy(), **self.prep_kwargs)[cols]
        self.datasets['Test'] = prep.transform_data(self.datasets['Test'].copy(), **self.prep_kwargs)[cols]
        self.n_features, self.n_targets = self.datasets['Train'].shape[1], self.labels['Train'].shape[1]
        super().run(bayesopt=False, algs=algs, **kwargs)
        
    def cross_validate(self, alg, **kwargs):
        X, Y = self.orig_datasets['Development'], self.orig_labels['Development']
        ikns = self.ikns[X.index]
        
        models, preds = [], []
        for fold, (train_idxs, valid_idxs) in enumerate(self.kf.split(X, Y.any(axis=1), ikns)):
            X_train, X_valid = X.iloc[train_idxs], X.iloc[valid_idxs]
            Y_train, Y_valid = Y.iloc[train_idxs], Y.iloc[valid_idxs]
            
            prep = self.prep_data()
            X_train = prep.transform_data(X_train.copy(), **self.prep_kwargs)
            X_valid = prep.transform_data(X_valid.copy(), **self.prep_kwargs)
            X_valid = X_valid[X_train.columns] # make sure columns match
            
            self.datasets['Train'], self.datasets['Valid'] = X_train, X_valid
            self.labels['Train'], self.labels['Valid'] = Y_train, Y_valid
            self.n_features, self.n_targets = X_train.shape[1], Y_train.shape[1]
            
            model = self.train_model(alg, save=False, **kwargs)
            pred = self.predict(model, 'Valid', alg)
            
            models.append(model)
            preds.append(pred)
        return models, preds
        
    def _eval_func(self, alg, **kwargs):
        kwargs = self.convert_hyperparams(kwargs)
        models, preds = self.cross_validate(alg, **kwargs)
        Y = self.orig_labels['Development']
        scores = [self.score_func(Y.loc[pred.index], pred) for pred in preds]
        return np.mean(scores)
    
class EnsemblerCV(Ensembler):
    def __init__(self, X, Y, tag, output_path, cv_preds, final_preds, task_type='C'):
        super().__init__(X, Y, tag, output_path, preds=final_preds, task_type=task_type)
        self.labels['Development'] = Y[tag['cohort'] == 'Development']
        # convert the format
        self.cv_algs = list(cv_preds.keys())
        self.nfolds = len(cv_preds[self.cv_algs[0]])
        self.cv_preds = {f'Fold{i}': [cv_preds[alg][i] for alg in self.cv_algs]
                         for i in range(self.nfolds)}
        
    def run(self, bayesopt=True, calibrate=True, evaluate=True, random_state=42):
        # run 5-fold cross validation bayesopt
        if bayesopt: 
            best_param = self.bayesopt('ENS', random_state=random_state, filename='ENS_cv_params')
                
        # get 5-fold cross validation scores
        if evaluate:
            best_param = load_pickle(
                f'{self.output_path}/best_params', 'ENS_cv_params',
                err_msg='Please tune hyperparameters for the ensemble model'
            )
            ensemble_weights = [best_param[alg] for alg in self.cv_algs]
            preds = self.cross_validate(ensemble_weights)
            if self.task_type == 'C' and calibrate:
                for i, pred in enumerate(preds):
                    calibrator = IsotonicCalibrator(self.target_events)
                    calibrator.calibrate(pred, self.labels['Development'].loc[pred.index])
                    preds[i] = calibrator.predict(pred)
            save_pickle(preds, f'{self.output_path}/preds/cv', 'ENS_cv_preds')
            # preds = load_pickle(f'{self.output_path}/preds/cv', 'ENS_cv_preds')
            labels = [self.labels['Development'].loc[pred.index] for pred in preds]
            cv_scores = get_cv_scores(preds, labels, self.output_path)
            cv_scores.to_csv(f'{self.output_path}/tables/cv/ENS_cv_scores.csv')
            
        # get final scores for all splits
        # super().run(bayesopt=True, calibrate=True, random_state=random_state)
            
    def cross_validate(self, ensemble_weights):
        preds = []
        for fold, pred in self.cv_preds.items():
            avg_pred = np.average(pred, axis=0, weights=ensemble_weights)
            avg_pred = pd.DataFrame(avg_pred, index=pred[0].index, columns=pred[0].columns)
            preds.append(avg_pred)
        return preds
    
    def _eval_func(self, **kwargs):
        ensemble_weights = [kwargs[alg] for alg in self.cv_algs]
        if not np.any(ensemble_weights): return -1e10 # weights are all zeros
        preds = self.cross_validate(ensemble_weights)
        Y = self.labels['Development']
        scores = [self.score_func(Y.loc[pred.index], pred) for pred in preds]
        return np.mean(scores)
    
def get_cv_scores(preds, labels, output_path):
    score_df = pd.DataFrame(columns=twolevel)
    SCI = ScoreConfidenceInterval(output_path, CLF_SCORE_FUNCS)
    for fold, (pred, label) in enumerate(zip(preds, labels)):
        assert all(pred.columns == label.columns)
        for target_event in label.columns:
            Y_true, Y_pred = label[target_event], pred[target_event]
            metrics = {name: func(Y_true, Y_pred) for name, func in CLF_SCORE_FUNCS.items()}
            ci = SCI.get_score_confidence_interval(Y_true, Y_pred, name=f'{target_event}_cv_fold{fold}')
            for name, (lower, upper) in ci.items():
                score_df.loc[f'Fold {fold+1}', (target_event, name)] = metrics[name]
                score_df.loc[f'Fold {fold+1}', (target_event, f'{name}_lower')] = lower
                score_df.loc[f'Fold {fold+1}', (target_event, f'{name}_upper')] = upper
    score_df.loc['Average'] = score_df.mean()
    return score_df
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adverse-event', type=str, required=True, choices=['ACU', 'CAN', 'CYTO', 'DEATH'])
    parser.add_argument('--algorithm', type=str, required=True, choices=['LR', 'RF', 'XGB', 'NN', 'RNN', 'ENS'])
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--bayesopt', action='store_true')
    msg = 'within number of days an event to occur after a treatment session to be a target. Only for ACU.'
    parser.add_argument('--days', type=int, required=False, default=30, help=msg)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    adverse_event = args.adverse_event
    alg = args.algorithm
    output_path = args.output_path
    bayesopt = args.bayesopt
    evaluate = args.evaluate
    days = args.days
    
    if adverse_event == 'DEATH':
        if output_path is None:
            output_path = f'{root_path}/{death_folder}/models'
        initialize_folders(output_path, extra_folders=['preds/cv', 'tables/cv'])
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
    
    if alg == 'ENS':
        cv_preds = {} # cross validation predictions
        for filename in os.listdir(f'{output_path}/preds/cv'):
            if not os.path.isfile(f'{output_path}/preds/cv/{filename}'): continue
            alg = filename.split('_')[0]
            cv_preds[alg] = load_pickle(f'{output_path}/preds/cv', filename.replace('.pkl', ''))
            
        preds = {} # final model predictions for all splits
        for filename in os.listdir(f'{output_path}/preds'):
            if not os.path.isfile(f'{output_path}/preds/{filename}'): continue
            preds.update(load_pickle(f'{output_path}/preds', filename.replace('.pkl', '')))
            
        trainer = EnsemblerCV(X, Y, tag, output_path, cv_preds, preds)
        trainer.run(bayesopt=True)
    else:
        trainer= TrainerCV(X, Y, tag, prep_data, output_path)
        trainer.run(bayesopt=bayesopt, algs=[alg], train=True, save_preds=True, pred_filename=f'{alg}_preds')

    if evaluate:
        preds, labels = trainer.preds.copy(), trainer.labels.copy()
        evaluator = EvaluateClf(output_path, preds, labels)
        evaluator.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=True)

if __name__ == '__main__':
    main()
