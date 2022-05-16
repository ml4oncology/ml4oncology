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
import shutil
import sys
import os
sys.path.append(os.getcwd())
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

from scripts.utility import (initialize_folders, load_predictions, save_predictions)
from scripts.config import (root_path, cyto_folder, acu_folder, can_folder, death_folder)
from scripts.prep_data import (PrepDataEDHD, PrepDataCYTO)
from scripts.train import (TrainML, TrainRNN, TrainENS)
from scripts.evaluate import (Evaluate)

class End2EndPipeline():
    def __init__(self):
        self.output_path = self.get_output_path()
        self.dataset = self.get_data_splits()
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = self.dataset
        self.labels = {'Train': Y_train, 'Valid': Y_valid, 'Test': Y_test}
        
    def get_output_path(self):
        raise NotImplementedError
        
    def get_data_splits(self):
        raise NotImplementedError

    def get_train(self, rnn=False, processes=64):
        if rnn:
            return TrainRNN(self.dataset, self.output_path)
        else:
            X_train, Y_train, X_valid, Y_valid, X_test, Y_test = self.dataset
            dataset = (X_train.drop(columns=['ikn']), Y_train, 
                       X_valid.drop(columns=['ikn']), Y_valid, 
                       X_test.drop(columns=['ikn']), Y_test)
            return TrainML(dataset, self.output_path, n_jobs=processes)
        
    def tune_and_train(self, rnn=False, run_bayesopt=True, run_training=True):
        train = self.get_train(rnn=rnn)
        train.tune_and_train(run_bayesopt=run_bayesopt, run_training=run_training, save_preds=True)
    
    def compute_ensemble(self, run_bayesopt=True, run_calibration=True):
        preds = load_predictions(save_dir=f'{self.output_path}/predictions')
        preds_rnn = load_predictions(save_dir=f'{self.output_path}/predictions', filename='rnn_predictions')
        for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
        train_ens = TrainENS(self.output_path, preds, self.labels)
        train_ens.tune_and_train(run_bayesopt=run_bayesopt, run_calibration=run_calibration)
        return train_ens

    def evaluate(self):
        ens = self.compute_ensemble(run_bayesopt=False, run_calibration=False)
        eval_models = Evaluate(output_path=self.output_path, preds=ens.preds, labels=self.labels, orig_data=None)
        score_df = eval_models.get_evaluation_scores(splits=['Test'], display_ci=True, save_ci=True, save_score=True)
        return score_df
    
    def run(self, run_bayesopt=False, run_training=True):
        self.tune_and_train(rnn=False, run_bayesopt=run_bayesopt, run_training=run_training)
        self.tune_and_train(rnn=True, run_bayesopt=run_bayesopt, run_training=run_training)
        self.compute_ensemble(run_bayesopt=True, run_calibration=True)
        score_df = self.evaluate()
        return score_df

class End2EndPipelineEDHD(End2EndPipeline):
    def __init__(self, adverse_event='acu', days=30):
        self.adverse_event = adverse_event
        self.days = days
        super().__init__()
        
    def get_output_path(self):
        if self.adverse_event == 'acu':
            output_path = f'{root_path}/{acu_folder}/models/within_{self.days}_days'
        elif self.adverse_event == 'death':
            output_path = f'{root_path}/{death_folder}/models'
        else:
            raise ValueError('adverse event must be either acu or death')
        return output_path
        
    def get_data_splits(self):
        target_keywords = {'acu': f'_within_{self.days}days', 'death': 'Mortality'}
        target_keyword = target_keywords[self.adverse_event]
        prep = PrepDataEDHD(adverse_event=self.adverse_event)
        model_data = prep.get_data(target_keyword, missing_thresh=80)
        model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
        train, valid, test = prep.split_data(prep.dummify_data(model_data), target_keyword=target_keyword, convert_to_float=False)
        (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train, valid, test
        print(prep.get_label_distribution(Y_train, Y_valid, Y_test))
        Y_train.columns = Y_train.columns.str.replace(target_keyword, '')
        Y_valid.columns = Y_valid.columns.str.replace(target_keyword, '')
        Y_test.columns = Y_test.columns.str.replace(target_keyword, '')
        X_train['ikn'], X_valid['ikn'], X_test['ikn'] = model_data['ikn'], model_data['ikn'], model_data['ikn']
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
    
    def run(self, run_bayesopt=False, run_training=True):
        initialize_folders(self.output_path, extra_folders=['figures/important_groups', 'figures/rnn_train_performance'])
        if not run_bayesopt and self.adverse_event == 'acu':
            # copy best parameters from 'within 30 days' models
            for alg in ['LR', 'NN', 'RF', 'XGB', 'RNN']:
                shutil.copyfile(f'{self.output_path.replace(str(self.days), "30")}/best_params/{alg}_classifier_best_param.pkl', 
                                f'{self.output_path}/best_params/{alg}_classifier_best_param.pkl')
        score_df = super().run(run_bayesopt=run_bayesopt, run_training=run_training)
        return score_df

class End2EndPipelineCYTO(End2EndPipeline):
    def __init__(self):
        self.split_date = '2017-06-30'
        super().__init__()
        
    def get_output_path(self):
        return f'{root_path}/{cyto_folder}/models'
        
    def get_data_splits(self):
        prep = PrepDataCYTO()
        model_data = prep.get_data(include_first_date=True, missing_thresh=75)
        model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
        train, valid, test = prep.split_data(prep.dummify_data(model_data), split_date=self.split_date, convert_to_float=False)
        (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train, valid, test
        Y_train = prep.regression_to_classification(Y_train)
        Y_valid = prep.regression_to_classification(Y_valid)
        Y_test = prep.regression_to_classification(Y_test)
        print(prep.get_label_distribution(Y_train, Y_valid, Y_test))
        X_train['ikn'], X_valid['ikn'], X_test['ikn'] = model_data['ikn'], model_data['ikn'], model_data['ikn']
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

if __name__ == '__main__':
    # pipeline = End2EndPipelineEDHD(adverse_event='acu', days=180)
    pipeline = End2EndPipelineCYTO()
    pipeline.run(run_bayesopt=True, run_training=True)
