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
Script to compute shap values
"""
import argparse
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
np.random.seed(42)
import shap

from src import logger
from src.config import min_chemo_date, split_date
from src.prep_data import PrepDataEDHD
from src.train import Trainer
from src.utility import load_pickle, save_pickle

class SHAPModel:
    def __init__(self, data, trainer, target_event='365d Mortality', drop_cols=None):
        self.data = data
        self.trainer = trainer
        self.target_idx = trainer.target_events.index(target_event)
        if drop_cols is None: drop_cols = ['ikn', 'visit_date']
        self.drop_cols = drop_cols
        
        # load models
        output_path = trainer.output_path
        self.lr_model = load_pickle(output_path, 'LR')
        self.rf_model = load_pickle(output_path, 'RF')
        self.xgb_model = load_pickle(output_path, 'XGB')
        self.nn_model = load_pickle(output_path, 'NN')
        self.rnn_model = load_pickle(output_path, 'RNN')
        self.rnn_model.model.rnn.flatten_parameters()
        self.ensemble_weights = load_pickle(f'{output_path}/best_params', 'ENS_params')
        self.ensemble_weights = {alg: w for alg, w in self.ensemble_weights.items() if w > 0}
        
    def predict(self, X):
        weights, preds = [], []
        for alg, weight in self.ensemble_weights.items():
            weights.append(weight)
            preds.append(self._predict(alg, X))
        pred = np.average(preds, axis=0, weights=weights)
        return pred
    
    def _predict(self, alg, X):
        if alg == 'RNN':
            return self._rnn_predict(X)

        X = X.drop(columns=self.drop_cols)
        if alg == 'LR':
            return self.lr_model.model.estimators_[self.target_idx].predict_proba(X)[:, 1]
        elif alg == 'XGB':
            return self.xgb_model.model.estimators_[self.target_idx].predict_proba(X)[:, 1]
        elif alg == 'RF':
            return self.rf_model.model.estimators_[self.target_idx].predict_proba(X)[:, 1]
        elif alg == 'NN':
            return self.nn_model.predict(X)[:, self.target_idx].cpu().detach().numpy()
        else:
            raise ValueError(f'{alg} not supported')
        
    def _rnn_predict(self, X):
        # Reformat to sequential data
        X = X.copy()
        N = len(X)
        # get the ikn and visit date for this sample row
        res = {}
        for col in ['ikn', 'visit_date']:
            mask = X[col] != -1
            val = X.loc[mask, col].unique()
            assert len(val) == 1
            res[col] = val[0]
        # get patient's historical data
        # NOTE: historical data is NOT permuted
        hist = self.data.query(f'ikn == {res["ikn"]} & visit_date < {res["visit_date"]}').copy()
        n = len(hist)
        hist = pd.concat([hist] * N) # repeat patient historical data for each sample row
        # set up new ikn for each sample row
        ikns = np.arange(0, N, 1) + res['ikn']
        hist['ikn'] = np.repeat(ikns, n)
        X['ikn'] = ikns
        # combine historical data and sample rows togethers
        X['visit_date'] = res['visit_date']
        X = pd.concat([X, hist]).sort_values(by=['ikn', 'visit_date'], ignore_index=True)

        # Get the RNN predictions
        self.trainer.ikns = X['ikn']
        self.trainer.labels['Test'] = pd.DataFrame(True, columns=self.trainer.target_events, index=X.index) # dummy variable
        X = X.drop(columns=self.drop_cols)
        self.trainer.datasets['Test'] = X # set the new input
        pred = self.trainer.predict(self.rnn_model, 'Test', 'RNN', calibrated=True)
        pred = pred.to_numpy()[n::n+1, self.target_idx] # only take the predictions for sample rows
        return pred

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adverse-event', type=str, required=True, choices=['ACU', 'AKI', 'CKD', 'CYTO', 'DEATH'])
    parser.add_argument('--output-path', type=str, default='./')
    parser.add_argument('--algorithm', type=str, default='ENS', choices=['ENS', 'LR', 'XGB', 'RF', 'NN', 'RNN'])
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    adverse_event = args.adverse_event
    output_path = args.output_path
    algorithm = args.algorithm
    
    if adverse_event != 'DEATH':
        raise NotImplementedError('Sorry, only adverse_event DEATH is currently supported')
    if algorithm != 'ENS':
        raise NotImplementedError('Sorry, only algorithm ENS is currently supported')
    
    # set up the data
    prep = PrepDataEDHD(adverse_event='death', target_keyword='Mortality')
    model_data = prep.get_data(missing_thresh=80, treatment_intents=['P'], verbose=False)
    X, Y, tag = prep.split_and_transform_data(
        model_data, split_date=split_date, remove_immediate_events=True, ohe_kwargs={'verbose': False}
    )
    model_data = model_data.loc[tag.index] # remove sessions in model_data that were excluded during split_and_transform
    trainer = Trainer(X, Y, tag, output_path)
    
    visit_date = prep.event_dates['visit_date']
    test_mask = tag['split'] == 'Test'    
    data = pd.concat([
        X[test_mask].astype(float), 
        (visit_date[test_mask] - pd.Timestamp(min_chemo_date)).dt.days.astype(int), 
        tag.loc[test_mask, 'ikn']
    ], axis=1)
    
    # use a 1000 random patient sample subset of the test data
    sampled_ikns = np.random.choice(data['ikn'].unique(), size=1000)
    mask = data['ikn'].isin(sampled_ikns)
    data, bg_dist = data[mask], data[~mask]
    drop_cols = ['ikn', 'visit_date']
    bg_dist[drop_cols] = -1

    # compute shap values for the ENS model
    # NOTE: the explainer will loop through each sample row, and create multiple versions of the sample row
    # with different feature permutations, where the values are replaced with the background distribution values
    shap_model = SHAPModel(data, trainer, target_event='365d Mortality', drop_cols=drop_cols)
    explainer = shap.Explainer(shap_model.predict, bg_dist, seed=42)
    shap_values = explainer(data, max_evals=800)
    save_pickle(shap_values, f'{output_path}/feat_importance', 'shap_values_1000_patient')
    
if __name__ == '__main__':
    main()
