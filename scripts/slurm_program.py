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
import os
import pickle
import shutil
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from src.config import (
    root_path, cyto_folder, acu_folder, can_folder, death_folder, reco_folder, 
    split_date
)
from src.evaluate import (Evaluate)
from src.prep_data import (PrepDataEDHD, PrepDataCYTO)
from src.surv.survival import TrainSurv
from src.train import (TrainML, TrainRNN, TrainENS)
from src.utility import (initialize_folders, load_predictions, save_predictions)

class End2EndPipeline():
    def __init__(self):
        self.output_path = self.get_output_path()
        self.dataset = self.get_data_splits()
        
    def run(self, run_bayesopt=False, run_training=True):
        if run_bayesopt or run_training:
            self.tune_and_train(
                rnn=False, run_bayesopt=run_bayesopt, run_training=run_training
            )
            self.tune_and_train(
                rnn=True, run_bayesopt=run_bayesopt, run_training=run_training
            )
        self.compute_ensemble(run_bayesopt=True, run_calibration=True)
        score_df = self.evaluate()
        return score_df
    
    def tune_and_train(self, rnn=False, run_bayesopt=True, run_training=True):
        train = self.get_train(rnn=rnn)
        train.tune_and_train(
            run_bayesopt=run_bayesopt, run_training=run_training, 
            save_preds=True
        )
    
    def evaluate(self):
        ens = self.compute_ensemble(run_bayesopt=False, run_calibration=False)
        
        eval_models = Evaluate(
            output_path=self.output_path, preds=ens.preds, labels=ens.labels, 
            orig_data=None
        )
        
        score_df = eval_models.get_evaluation_scores(
            display_ci=True, save_ci=True, save_score=True
        )
        return score_df
    
    def compute_ensemble(self, run_bayesopt=True, run_calibration=True):
        preds = load_predictions(save_dir=f'{self.output_path}/predictions')
        preds_rnn = load_predictions(
            save_dir=f'{self.output_path}/predictions', 
            filename='rnn_predictions'
        )
        for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
            
        train_ens = TrainENS(self.dataset, self.output_path, preds)
        train_ens.tune_and_train(
            run_bayesopt=run_bayesopt, run_calibration=run_calibration
        )
        return train_ens
    
    def get_train(self, rnn=False, processes=64):
        if rnn:
            return TrainRNN(self.dataset, self.output_path)
        else:
            X_train, X_valid, X_test, Y_train, Y_valid, Y_test = self.dataset
            drop_ikn = lambda df: df.drop(columns=['ikn'])
            dataset = (
                drop_ikn(X_train), drop_ikn(X_valid), drop_ikn(X_test), 
                Y_train, Y_valid, Y_test
            )
            return TrainML(dataset, self.output_path, n_jobs=processes)
        
    def get_output_path(self):
        raise NotImplementedError
        
    def get_data_splits(self):
        raise NotImplementedError
        
    def add_ikn(self, train, valid, test, data):
        train['ikn'] = data['ikn']
        valid['ikn'] = data['ikn'] 
        test['ikn'] = data['ikn']
    
class End2EndPipelinePROACCT(End2EndPipeline):
    def __init__(self, days=30):
        self.days = days
        super().__init__()
        
    def get_output_path(self):
        return f'{root_path}/{acu_folder}/models/within_{self.days}_days'
        
    def get_data_splits(self):
        target_keyword = f'_within_{self.days}days'
        prep = PrepDataEDHD(adverse_event='acu')
        
        model_data = prep.get_data(target_keyword, missing_thresh=80)
        
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(
            prep.dummify_data(model_data.copy()), target_keyword=target_keyword
        )
        
        print(prep.get_label_distribution(Y_train, Y_valid, Y_test))
        
        Y_train.columns = Y_train.columns.str.replace(target_keyword, '')
        Y_valid.columns = Y_valid.columns.str.replace(target_keyword, '')
        Y_test.columns = Y_test.columns.str.replace(target_keyword, '')
        
        self.add_ikn(X_train, X_valid, X_test, model_data)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    
    def run(self, run_bayesopt=False, run_training=True):
        initialize_folders(
            self.output_path, 
            extra_folders=[
                'figures/important_groups', 'figures/rnn_train_performance'
            ]
        )
        
        if not run_bayesopt and self.days != 30:
            from_path = self.output_path.replace(str(self.days), "30")
            to_path = self.output_path
            
            # copy best parameters from 'within 30 days' models
            for alg in ['LR', 'NN', 'RF', 'XGB', 'RNN']:
                shutil.copyfile(
                    f'{from_path}/best_params/{alg}_best_param.pkl',
                    f'{to_path}/best_params/{alg}_best_param.pkl'
                )
                
        score_df = super().run(
            run_bayesopt=run_bayesopt, run_training=run_training
        )
        
        return score_df
    
class End2EndPipelineDEATH(End2EndPipeline):
    def get_output_path(self):
        return f'{root_path}/{death_folder}/models'
        
    def get_data_splits(self):
        target_keyword = 'Mortality'
        prep = PrepDataEDHD(adverse_event='death')
        
        model_data = prep.get_data(
            target_keyword, missing_thresh=75, treatment_intents=['P']
        )
        
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(
            prep.dummify_data(model_data.copy()), 
            target_keyword=target_keyword, 
            split_date=split_date
        )
        
        print(prep.get_label_distribution(Y_train, Y_valid, Y_test))
        
        self.add_ikn(X_train, X_valid, X_test, model_data)
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

class End2EndPipelineCYTO(End2EndPipeline):
    def get_output_path(self):
        return f'{root_path}/{cyto_folder}/models'
        
    def get_data_splits(self):
        prep = PrepDataCYTO()
        model_data = prep.get_data(missing_thresh=75)
        dataset = X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(
            prep.dummify_data(model_data.copy()), split_date=split_date
        )
        print(prep.get_label_distribution(Y_train, Y_valid, Y_test))
        self.add_ikn(X_train, X_valid, X_test, model_data)
        return dataset
    
class End2EndPipelineSurv:
    def __init__(self):
        self.main_dir = f'{root_path}/{reco_folder}'
        self.output_path = f'{self.main_dir}/models'
        
        filename = f'{self.main_dir}/data/final_dataset.pkl'
        with open(filename, 'rb') as file:
            dataset = pickle.load(file)
            
        filename = f'{self.main_dir}/data/propensity_score.pkl'
        with open(filename, 'rb') as file:
            propensity = pickle.load(file)
            
        matching_module = {name: df.columns.tolist() for name, df in propensity.items()}
        
        self.args = (dataset, self.output_path, matching_module)
        
    def run(self, run_bayesopt=True, run_training=True):
        train_surv = TrainSurv(*self.args)
        train_surv.tune_and_train(run_bayesopt=run_bayesopt, run_training=run_training)

if __name__ == '__main__':
    pipeline = End2EndPipelineDEATH()
    pipeline.run(run_bayesopt=False, run_training=True)
