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
import sys
sys.path.append(os.getcwd())
import argparse
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

from scripts.utility import (load_ml_model, load_ensemble_weights)
from scripts.config import (root_path, cyto_folder, acu_folder, can_folder, death_folder,
                            blood_types, variable_groupings_by_keyword)
from scripts.prep_data import (PrepDataCYTO, PrepDataEDHD, PrepDataCAN)
from scripts.train import (TrainML, TrainRNN)

from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier as placeholderClassifier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')

class PermImportance:    
    def __init__(self, output_path, preload=True):
        self.output_path = output_path
        
        # get test data with 
        #   1. original features 
        #   2. one-hot encoded features
        orig_data_splits, data_splits = self.get_data()
        _, _, _, _, self.X_test, self.Y_test, = data_splits
        _, _, _, _, self.orig_X_test, _ = orig_data_splits
        
        # initialize RNN and ML Training class - you need some of their member functions
        self.train_rnn = TrainRNN(data_splits, output_path)
        self.train_ml = TrainML(data_splits, output_path)
        
        self.ikns = self.X_test.pop('ikn') # patient id column
        self.dummy_cols = self.X_test.columns
        self.target_types = self.Y_test.columns.tolist()
        self.params =  {'n_repeats': 5, 'random_state': 42}
        
        if preload:
            # load pretrained models 
            self.ensemble_weights = load_ensemble_weights(save_dir=f'{self.output_path}/best_params')
            self.models = {}
            for algorithm in self.ensemble_weights:
                if algorithm == 'RNN':
                    filename = f'{self.output_path}/best_params/RNN_classifier_best_param.pkl'
                    with open(filename, 'rb') as file: 
                        rnn_model_param = pickle.load(file)
                    del rnn_model_param['learning_rate']
                    self.models[algorithm] = self.train_rnn.get_model(load_saved_weights=True, **rnn_model_param)
                else:
                    self.models[algorithm] = load_ml_model(self.output_path, algorithm)
        
    def load_data(self):
        raise NotImplementedError
        
    def clean_y(self):
        raise NotImplementedError
        
    def get_groupings(self):
        raise NotImplementedError
        
    def get_data_splits(self, data, include_ikn=False):
        train, valid, test = self.prep.split_data(data, target_keyword=self.target_keyword, convert_to_float=False)
        (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train, valid, test
        if include_ikn:
            # the ikn corresponding to each data split are assigned by matching the indices
            X_train['ikn'], X_valid['ikn'], X_test['ikn'] = data['ikn'], data['ikn'], data['ikn']
        data_splits = (X_train, self.clean_y(Y_train), X_valid, self.clean_y(Y_valid), X_test, self.clean_y(Y_test)) 
        return data_splits
        
    def get_data(self):
        data = self.load_data()
        data, clip_thresholds = self.prep.clip_outliers(data)
        
        # original feature data splits
        orig_data_splits = self.get_data_splits(data)
        
        # one-hot encoded feature data splits
        data_splits = self.get_data_splits(self.prep.dummify_data(data), include_ikn=True)

        return orig_data_splits, data_splits
    
    def predict(self, algorithm, model, X, split='Test'):
        X = X.astype(float)
        
        if algorithm == 'ENS':
            pred_prob, ensemble_weights = [], []
            for alg, weight in self.ensemble_weights.items():
                pred_prob.append(self.predict(alg, self.models[alg], X))
                ensemble_weights.append(weight)
            pred_prob = np.average(pred_prob, axis=0, weights=ensemble_weights)
            
        elif algorithm == 'RNN':
            X['ikn'] = self.ikns
            self.train_rnn.dataset_splits[split] = self.train_rnn.transform_to_tensor_dataset(X, self.Y_test)
            pred_prob, index_arr = self.train_rnn._get_model_predictions(self.models[algorithm], split)
            # double check and clean up
            assert all(self.Y_test.index == index_arr)
            del X['ikn']
            
        else:
            self.train_ml.preds[split][algorithm] = None # claer cache
            self.train_ml.data_splits[split] = (X, self.Y_test) # set the new input
            pred_prob, _ = self.train_ml.predict(model, split, algorithm)
            pred_prob = pred_prob.values
        return pred_prob
    
    def get_feature_importance(self, algorithm, columns=None):
        """Run permutation importance across the original, NOT one-hot-encoded columns/features
        """
        if columns is None: columns = self.orig_X_test.columns
        model = None if algorithm == 'ENS' else self.models[algorithm]
        result = pd.DataFrame()
        
        # get original score
        pred_prob = self.predict(algorithm, model, self.X_test)
        orig_scores = roc_auc_score(self.Y_test, pred_prob, average=None)
            
        for col in columns:
            importances = []
            for i in range(self.params['n_repeats']):
                X = self.orig_X_test.copy()
                X[col] = X[col].sample(frac=1, random_state=self.params['random_state']+i).set_axis(X.index)
                X = self.prep.dummify_data(X)
                X[self.dummy_cols.difference(X.columns)] = 0 # add the missing columns
                X = X[self.dummy_cols]
                pred_prob = self.predict(algorithm, model, X)
                importances.append(orig_scores - roc_auc_score(self.Y_test, pred_prob, average=None))
            result.loc[col, self.target_types] = np.mean(importances, axis=0)
            logging.info(f'Successfully computed perm feature importance scores for {col}')
        result.to_csv(f'{self.output_path}/perm_importance/{algorithm}_feature_importance.csv', index_label='index')
        
    def get_group_importance(self, algorithm):
        """Run permutation importance across subgroups of columns/features
        """
        model = None if algorithm == 'ENS' else self.models[algorithm]
        result = pd.DataFrame()
        
        # get original score
        pred_prob = self.predict(algorithm, model, self.X_test)
        orig_scores = roc_auc_score(self.Y_test, pred_prob, average=None)
            
        for group, keyword in self.get_groupings().items():
            permute_cols = self.dummy_cols[self.dummy_cols.str.contains(keyword)]
            importances = []
            for i in range(self.params['n_repeats']):
                X = self.X_test.copy()
                X[permute_cols] = X[permute_cols].sample(frac=1, random_state=self.params['random_state']+i).set_index(X.index)
                pred_prob = self.predict(algorithm, model, X)
                importances.append(orig_scores - roc_auc_score(self.Y_test, pred_prob, average=None))
            result.loc[group, self.target_types] = np.mean(importances, axis=0)
            logging.info(f'Successfully computed perm group importance scores for {group}')
        result.to_csv(f'{self.output_path}/perm_importance/{algorithm}_group_importance.csv', index_label='index')
            
class PermImportanceCYTO(PermImportance):
    """Permutation importance for cytopenia
    """
    def __init__(self, output_path):
        self.prep = PrepDataCYTO()
        self.target_keyword = 'target'
        super().__init__(output_path)
        
    def load_data(self):
        return self.prep.get_data(missing_thresh=75)
    
    def clean_y(self, Y):
        return self.prep.regression_to_classification(Y)
    
    def get_groupings(self):
        return variable_groupings_by_keyword
    
class PermImportanceEDHD(PermImportance):
    """
    ED - Emergency Department visits
    H - Hospitalizations
    D - Death
    
    Permutation importance for acute care use (ED/H) or death (D)
    """
    def __init__(self, output_path, adverse_event='acu', days=None):
        self.adverse_event = adverse_event
        if self.adverse_event not in {'acu', 'death'}: 
            raise ValueError('advese_event must be either acu (acute case use) or death')
        
        if self.adverse_event == 'acu':
            if days is None: raise ValueError('days can not be None for target acu')
            self.target_keyword = f'_within_{days}days'
        elif self.adverse_event == 'death':
            self.target_keyword = 'Mortality'
            
        self.prep = PrepDataEDHD(self.adverse_event)
        super().__init__(output_path)
    
    def load_data(self):
        return self.prep.get_data(self.target_keyword, missing_thresh=80)
    
    def clean_y(self, Y):
        Y.columns = Y.columns.str.replace(self.target_keyword, '')
        return Y
    
    def get_groupings(self):
        return variable_groupings_by_keyword
    
class PermImportanceCAN(PermImportance):
    """Permutation importance for cisplatin-induced nephrotoxicity
    """
    def __init__(self, output_path, adverse_event):
        self.prep = PrepDataCAN(adverse_event=adverse_event)
        self.target_keyword = 'SCr'
        if adverse_event == 'ckd':
            self.target_keyword += '|eGFR'
        super().__init__(output_path)
        
    def load_data(self):
        return self.prep.get_data(missing_thresh=80)
    
    def clean_y(self, Y):
        return self.prep.regression_to_classification(Y)
    
    def get_groupings(self):
        return variable_groupings_by_keyword

def main(pm, algorithm='ENS', permute_group=False):
    if permute_group:
        pm.get_group_importance(algorithm=algorithm)
    else:
        pm.get_feature_importance(algorithm=algorithm)
        
def main_cyto(permute_group=False):
    output_path = f'{root_path}/{cyto_folder}/models'
    pm = PermImportanceCYTO(output_path)
    main(pm, algorithm='XGB', permute_group=permute_group)

def main_acu(days=30, permute_group=False):
    output_path = f'{root_path}/{acu_folder}/models/within_{days}_days'
    pm = PermImportanceEDHD(output_path, adverse_event='acu', days=days)
    main(pm, algorithm='ENS', permute_group=permute_group)
    
def main_death(permute_group=False):
    output_path = f'{root_path}/{death_folder}/models'
    pm = PermImportanceEDHD(output_path, adverse_event='death')
    main(pm, algorithm='ENS', permute_group=permute_group)
        
def main_caaki(permute_group=False):
    adverse_event = 'aki'
    output_path = f'{root_path}/{can_folder}/models/{adverse_event.upper()}'
    pm = PermImportanceCAN(output_path, adverse_event)
    main(pm, algorithm='ENS', permute_group=permute_groups)
    
def main_cackd(permute_group=False):
    adverse_event = 'ckd'
    output_path = f'{root_path}/{can_folder}/models/{adverse_event.upper()}'
    pm = PermImportanceCAN(output_path, adverse_event)
    main(pm, algorithm='ENS', permute_group=permute_group)
    
if __name__ == '__main__':
    main_events = {'CYTO': main_cyto, # Cytopenia
                   'ACU': main_acu,  # Acute Care Use
                   'CAAKI': main_caaki, # Cisplatin-Associated Acute Kidney Injury
                   'CACKD': main_cackd, # Cisplatin-Associated Chronic Kidney Disease
                   'DEATH': main_death}
    parser = argparse.ArgumentParser()
    parser.add_argument('--adverse-event', default='CYTO', choices=main_events.keys())
    parser.add_argument('--permute-group', action='store_true', default=False, 
                        help="Run permutation group importance (permute subset of alike features) " +\
                        "instead of permutation feature importance (permute each feature)")
    args = vars(parser.parse_args())
    main_event = main_events[args['adverse_event']]
    main_event(permute_group=args['permute_group'])
