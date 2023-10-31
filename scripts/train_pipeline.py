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
import argparse
import os
import pickle
import shutil
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from src.config import split_date
from src.evaluate import EvaluateClf
from src.prep_data import PrepDataEDHD, PrepDataCYTO
from src.train import Ensembler, Trainer
from src.utility import initialize_folders, load_pickle, save_pickle

class Pipeline():
    def __init__(self, output_path):
        self.output_path = output_path
        
    def run(self, algorithm, evaluate=True, bayesopt=False):
        X, Y, tag = self.get_data()
        trainer = self.train(X, Y, tag, algorithm, bayesopt=bayesopt)
        if evaluate:
            score_df = self.evaluate(trainer.preds, trainer.labels)
            print(score_df)
        
    def train(self, X, Y, tag, alg, bayesopt=False):
        args = (X, Y, tag, self.output_path)
        if alg == 'ENS':
            preds = self.load_predictions()
            ensembler = Ensembler(*args, preds)
            ensembler.run(bayesopt=True)
            return ensembler
        
        trainer = Trainer(*args)
        trainer.run(bayesopt=bayesopt, train=True, save_preds=True, pred_filename=f'{alg}_preds', algs=[alg])
        return trainer
        
    def evaluate(self, preds, labels):
        evaluator = EvaluateClf(self.output_path, preds, labels)
        score_df = evaluator.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=True, save_score=True)
        return score_df
        
    def load_predictions(self):
        preds = {}
        pred_dir = f'{self.output_path}/preds'
        for filename in os.listdir(pred_dir):
            if not os.path.isfile(f'{pred_dir}/{filename}'): continue
            preds.update(load_pickle(pred_dir, filename))
        return preds
    
    def get_data(self):
        raise NotImplementedError
    
class PROACCTPipeline(Pipeline):
    def __init__(self, *args, days=30, **kwargs):
        self.days = days
        self.target_keyword = f'within_{self.days}_days'
        super().__init__(*args, **kwargs)
        
    def get_data(self):
        prep = PrepDataEDHD(adverse_event='acu', target_keyword=self.target_keyword)
        model_data = prep.get_data(missing_thresh=80)
        X, Y, tag = prep.split_and_transform_data(model_data, remove_immediate_events=True)
        model_data = model_data.loc[tag.index]
        print(prep.get_label_distribution(Y, tag, with_respect_to='sessions'))
        Y.columns = Y.columns.str.replace(f' {self.target_keyword}', '')
        return X, Y, tag
        
class DEATHPipeline(Pipeline):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.target_keyword = 'Mortality'
        
    def get_data(self, evaluate=True, **kwargs):
        prep = PrepDataEDHD(adverse_event='death', target_keyword=self.target_keyword)
        model_data = prep.get_data(missing_thresh=80, treatment_intents=['P'])
        X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date, remove_immediate_events=True)
        model_data = model_data.loc[tag.index]
        print(prep.get_label_distribution(Y, tag, with_respect_to='sessions'))
        return X, Y, tag

class CYTOPipeline(Pipeline):  
    def get_data(self):
        prep = PrepDataCYTO()
        model_data = prep.get_data(missing_thresh=80)
        X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date)
        model_data = model_data.loc[tag.index]
        print(prep.get_label_distribution(Y, tag, with_respect_to='sessions'))
        return X, Y, tag
    
class CANPipeline(Pipeline):  
    def get_data(self):
        raise NotImplementedError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adverse-event', type=str, required=True, choices=['ACU', 'CAN', 'CYTO', 'DEATH'])
    parser.add_argument('--algorithm', type=str, required=True, choices=['LR', 'RF', 'XGB', 'NN', 'RNN', 'ENS'])
    parser.add_argument('--output-path', type=str, default='./')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--bayesopt', action='store_true')
    msg = 'within number of days an event to occur after a treatment session to be a target. Only for ACU.'
    parser.add_argument('--days', type=int, required=False, default=30, help=msg)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    adverse_event = args.adverse_event
    algorithm = args.algorithm
    output_path = args.output_path
    bayesopt = args.bayesopt
    evaluate = args.evaluate
    days = args.days
    
    if adverse_event == 'ACU':
        pipeline = PROACCTPipeline(output_path, days=days)
    elif adverse_event == 'DEATH':
        pipeline = DEATHPipeline(output_path)
    elif adverse_event == 'CYTO':
        pipeline = CYTOPipeline(output_path)
    elif adverse_event == 'CAN':
        pipeline = CANPipeline(output_path)
        
    pipeline.run(algorithm=algorithm, evaluate=evaluate, bayesopt=bayesopt)

if __name__ == '__main__':
    main()
