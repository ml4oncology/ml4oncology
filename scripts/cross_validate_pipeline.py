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

from src import logger
from src.config import root_path, death_folder, split_date
from src.evaluate import EvaluateClf
from src.prep_data import PrepDataEDHD
from src.train import Trainer, Ensembler
from src.utility import initialize_folders, load_pickle, save_pickle

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
        
    def run(self, bayesopt=True, algs=None, bayes_kwargs=None, **kwargs):
        # run 5-fold cross validation bayesopt
        if bayesopt: 
            if bayes_kwargs is None: bayes_kwargs = {}
            if algs is None: algs = self.models
            for alg in algs: 
                best_param = self.bayesopt(alg, **bayes_kwargs)
        
        # return back to original 
        self.datasets = self.orig_datasets
        self.labels = self.orig_labels
        prep = self.prep_data()
        self.datasets['Train'] = prep.transform_data(self.datasets['Train'].copy(), **self.prep_kwargs)
        cols = self.datasets['Train'].columns
        self.datasets['Valid'] = prep.transform_data(self.datasets['Valid'].copy(), **self.prep_kwargs)[cols]
        self.datasets['Test'] = prep.transform_data(self.datasets['Test'].copy(), **self.prep_kwargs)[cols]
        self.n_features, self.n_targets = self.datasets['Train'].shape[1], self.labels['Train'].shape[1]
        super().run(bayesopt=False, algs=algs, **kwargs)
        
    def _eval_func(self, alg, **kwargs):
        kwargs = self.convert_hyperparams(kwargs)
        X, Y = self.orig_datasets['Development'], self.orig_labels['Development']
        ikns = self.ikns[X.index]
        
        scores = []
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idxs, valid_idxs) in enumerate(kf.split(X, Y.any(axis=1), ikns)):
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
            scores.append(self.score_func(self.labels['Valid'], pred))
            
        return np.mean(scores)
    
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
    
    if alg == 'ENS':
        preds = {}
        for filename in os.listdir(f'{output_path}/preds'):
            if not os.path.isfile(f'{output_path}/preds/{filename}'): continue
            preds.update(load_pickle(f'{output_path}/preds', filename))
        trainer = Ensembler(X, Y, tag, output_path, preds)
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
