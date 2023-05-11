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
import argparse
import pickle
import os
import sys
sys.path.append(os.getcwd())

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from src.utility import load_ml_model, load_ensemble_weights
from src.config import split_date, variable_groupings_by_keyword
from src.prep_data import PrepDataCYTO, PrepDataEDHD, PrepDataCAN
from src.train import TrainML, TrainRNN

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')

class FeatImportance:    
    def __init__(self, output_path):
        self.output_path = output_path
        
        X, Y, tag, data = self.get_data()
        mask = tag['split'] == 'Test'
        self.Y_test = Y[mask]
        self.X_test = X[mask]
        self.enc_columns = self.X_test.columns # one-hot encoded columns
        self.orig_columns = data.columns.drop('ikn') # original columns
        # do not include missingness variables
        self.orig_columns = self.orig_columns[~self.orig_columns.str.endswith('_is_missing')]
        
        self.train_rnn = TrainRNN(X, Y, tag, output_path)
        self.train_ml = TrainML(X, Y, tag, output_path)
        
        self.params =  {'n_repeats': 5, 'random_state': 42}
        
        # load pretrained models 
        self.models = {}
        self.ensemble_weights = load_ensemble_weights(save_dir=f'{self.output_path}/best_params')
        for alg in self.ensemble_weights:
            if alg == 'RNN':
                filename = f'{self.output_path}/best_params/RNN_best_param.pkl'
                with open(filename, 'rb') as file: rnn_model_param = pickle.load(file)
                del rnn_model_param['learning_rate']
                self.models[alg] = self.train_rnn.get_model(load_saved_weights=True, **rnn_model_param)
            else:
                self.models[alg] = load_ml_model(self.output_path, alg)
    
    def get_feature_importance(self, alg):
        """Run permutation feature importance across each column/feature"""
        model = None if alg == 'ENS' else self.models[alg]
        
        # get original score
        pred_prob = self.predict(alg, model, self.X_test)
        orig_scores = roc_auc_score(self.Y_test, pred_prob, average=None)
        
        result = pd.DataFrame()
        for col in self.orig_columns:
            permute_cols = self.enc_columns[self.enc_columns.str.startswith(col)]
            importances = []
            for i in range(self.params['n_repeats']):
                new_scores = self._get_new_scores(permute_cols, alg, model, self.params['random_state'] + i)
                importances.append(orig_scores - new_scores)
                
            result.loc[col, self.train_ml.target_events] = np.mean(importances, axis=0)
            logging.info(f'Successfully computed permutation feature importance scores for {col} (size of '
                         f'permute_cols: {len(permute_cols)})')
        
        # save results
        filepath = f'{self.output_path}/perm_importance/{alg}_feature_importance.csv'
        result.to_csv(filepath, index_label='index')
        
    def get_group_importance(self, alg):
        """Run permutation importance across each group of columns/features"""
        model = None if alg == 'ENS' else self.models[alg]
        
        # get original score
        pred_prob = self.predict(alg, model, self.X_test)
        orig_scores = roc_auc_score(self.Y_test, pred_prob, average=None)
        
        result = pd.DataFrame()
        for group, keyword in variable_groupings_by_keyword.items():
            permute_cols = self.enc_columns[self.enc_columns.str.contains(keyword)]
            if len(permute_cols) == 0: continue
            
            importances = []
            for i in range(self.params['n_repeats']):
                new_scores = self._get_new_scores(permute_cols, alg, model, self.params['random_state'] + i)
                importances.append(orig_scores - new_scores)
                
            result.loc[group, self.train_ml.target_events] = np.mean(importances, axis=0)
            logging.info(f'Successfully computed permutation group importance scores for {group} (size of '
                         f'permute_cols: {len(permute_cols)})')
            
        # save results
        filepath = f'{self.output_path}/perm_importance/{alg}_group_importance.csv'
        result.to_csv(filepath, index_label='index')
                         
    def _get_new_scores(self, permute_cols, alg, model, random_state):
        X = self.X_test.copy()
        X[permute_cols] = X[permute_cols].sample(frac=1, random_state=random_state).set_axis(X.index)
        pred_prob = self.predict(alg, model, X)
        new_scores = roc_auc_score(self.Y_test, pred_prob, average=None)
        return new_scores
                         
    def predict(self, alg, model, X):
        X = X.astype(float)
        if alg == 'ENS':
            pred_prob, ensemble_weights = [], []
            for alg, weight in self.ensemble_weights.items():
                pred_prob.append(self.predict(alg, self.models[alg], X))
                ensemble_weights.append(weight)
            pred_prob = np.average(pred_prob, axis=0, weights=ensemble_weights)
            
        elif alg == 'RNN':
            self.train_rnn.tensor_dataset['Test'] = self.train_rnn.transform_to_tensor_dataset(X, self.Y_test)
            pred_prob, index_arr = self.train_rnn._get_model_predictions(self.models[alg], 'Test')
            assert all(self.Y_test.index == index_arr) # double check
            
        else:
            self.train_ml.preds['Test'][alg] = None # claer cache
            self.train_ml.datasets['Test'] = X # set the new input
            pred_prob = self.train_ml.predict(model, 'Test', alg)
            pred_prob = pred_prob.to_numpy()
        return pred_prob
    
    def get_data(self):
        self.prep.event_dates = pd.DataFrame() # reset event dates
        data = self.prep.get_data(**self.get_data_kwargs)
        X, Y, tag = self.prep.split_and_transform_data(data, verbose=False, **self.split_data_kwargs)
        Y = self.clean_y(Y)
        
        # only keep feature columns in data (remove columns used for labelling)
        cols = data.columns
        feat_cols = cols[~cols.str.contains(self.prep.target_keyword)]
        data = data[feat_cols]
        
        return X, Y, tag, data
                         
    def clean_y(self, Y):
        """You can overwrite this to do custom operations"""
        return Y
            
class CYTOFeatImportance(FeatImportance):
    """Permutation feature importance for cytopenia"""
    def __init__(self, output_path):
        self.prep = PrepDataCYTO()
        self.get_data_kwargs = {'missing_thresh': 80}
        self.split_data_kwargs = {'split_date': split_date}
        super().__init__(output_path)
    
class PROACCTFeatImportance(FeatImportance):
    """Permutation feature importance for acute care use (ED/H)
    ED - Emergency Department visits
    H - Hospitalizations
    """
    def __init__(self, output_path, days=30):
        self.prep = PrepDataEDHD(adverse_event='acu', target_keyword=f'within_{days}_days')
        self.get_data_kwargs = {'missing_thresh': 80}
        self.split_data_kwargs = {'remove_immediate_events': True}
        super().__init__(output_path)
    
    def clean_y(self, Y):
        Y.columns = Y.columns.str.replace(f' {self.prep.target_keyword}', '')
        return Y
    
class DEATHFeatImportance(FeatImportance):
    """Permutation feature importance for death"""
    def __init__(self, output_path):  
        self.prep = PrepDataEDHD(adverse_event='death', target_keyword='Mortality')
        self.get_data_kwargs = {'missing_thresh': 80, 'treatment_intents': ['P']}
        self.split_data_kwargs = {'split_date': split_date, 'remove_immediate_events': True}
        super().__init__(output_path)
    
class CANFeatImportance(FeatImportance):
    """Permutation feature importance for cisplatin-associated nephrotoxicity"""
    def __init__(self, output_path, adverse_event):
        self.prep = PrepDataCAN(adverse_event=adverse_event, target_keyword='SCr|dialysis|next')
        self.get_data_kwargs = {'missing_thresh': 80, 'include_comorbidity': True}
        self.split_data_kwargs = {'split_date': split_date}
        super().__init__(output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adverse-event', type=str, required=True, choices=['ACU', 'AKI', 'CKD', 'CYTO', 'DEATH'])
    parser.add_argument('--output-path', type=str, default='./')
    parser.add_argument('--algorithm', type=str, default='ENS', choices=['ENS', 'LR', 'XGB', 'RF', 'NN', 'RNN'])
    msg = ("Run permutation group importance (permute subset of alike features) instead of permutation feature "
           "importance (permute each feature)")
    parser.add_argument('--permute-group', action='store_true', help=msg)
    msg = 'within number of days an event to occur after a treatment session to be a target. Only for ACU.'
    parser.add_argument('--days', type=int, required=False, default=30, help=msg)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    adverse_event = args.adverse_event
    output_path = args.output_path
    algorithm = args.algorithm
    permute_group = args.permute_group
    days = args.days
    
    if adverse_event == 'ACU':
        fi = PROACCTFeatImportance(output_path, days=days)
    elif adverse_event == 'DEATH':
        fi = DEATHFeatImportance(output_path)
    elif adverse_event == 'CYTO':
        fi = CYTOFeatImportance(output_path)
    elif adverse_event == 'AKI':
        fi = CANFeatImportance(output_path, adverse_event='aki')
    elif adverse_event == 'CKD':
        fi = CANFeatImportance(output_path, adverse_event='ckd')
        
    if permute_group:
        fi.get_group_importance(alg=algorithm)
    else:
        fi.get_feature_importance(alg=algorithm)

if __name__ == '__main__':
    main()
