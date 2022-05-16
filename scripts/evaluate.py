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
import tqdm
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, 
                             average_precision_score, precision_recall_curve,
                             confusion_matrix)
from scripts.preprocess import (split_and_parallelize)
from scripts.utility import (twolevel, compute_bootstrap_scores, group_pred_by_outcome)
from scripts.config import (blood_types, clean_variable_mapping)
from scripts.visualize import (get_bbox)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')

class Evaluate:
    """
    Evaluate any/all of ML (Machine Learning), RNN (Recurrent Neural Network), and ENS (Ensemble) models
    """
    def __init__(self, output_path, preds, labels, orig_data):
        """
        Args:
            orig_data (pd.DataFrame): the original dataset before one-hot encoding and splitting
        """
        self.output_path = output_path
        self.preds = preds
        self.labels = labels
        self.splits = ['Valid', 'Test']
        self.models = list(preds['Test'].keys())
        self.target_types = list(labels['Test'].columns)
        self.orig_data = orig_data
        
    def get_bootstrapped_scores(self, Y_true, Y_pred_prob, algorithm, col, ci_df, n_bootstraps=10000):
        split, target_type = col
        ci_index = f'{algorithm}_{split}_{target_type}'
        if ci_index not in ci_df.index:
            auc_scores = compute_bootstrap_scores(Y_true, Y_pred_prob, n_bootstraps=n_bootstraps)
            logging.info(f'Completed bootstrap computations for {ci_index}')
            auroc_scores, auprc_scores = np.array(auc_scores).T
            ci_df.loc[(ci_index, 'AUROC'), range(n_bootstraps)] = auroc_scores
            ci_df.loc[(ci_index, 'AUPRC'), range(n_bootstraps)] = auprc_scores
        return ci_df.loc[ci_index].values
        
    def get_confidence_interval(self, Y_true, Y_pred_prob, algorithm, col, score_df, ci_df):
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(Y_true, Y_pred_prob, algorithm, col, ci_df)
        for name, scores in [('AUROC Score', auroc_scores), ('AUPRC Score', auprc_scores)]:
            lower, upper = np.percentile(scores, [2.5, 97.5]).round(3)
            score_df.loc[(algorithm, name), col] = f'{score_df.loc[(algorithm, name), col]} ({lower}-{upper})'
        return score_df
    
    def cyto_baseline_score(self, score_df, splits=None, target_types=None, digit_round=3):
        """This baseline model predicts the target mean for each previously measured blood count bin
        """
        if target_types is None: target_types = self.target_types
        if splits is None: splits = self.splits
        
        for target_type in target_types:
            # compute mean of target for each previous blood count bin
            blood_type = [bt for bt, info in blood_types.items() if info['cytopenia_name'] == target_type][0]
            base_col = f'baseline_{blood_type}_count'
            mean_targets = {}
            Y = self.labels['Train']
            X = self.orig_data.loc[Y.index]
            blood_count_bins = X[base_col].round(1)
            for blood_count, group in blood_count_bins.groupby(blood_count_bins):
                means = Y.loc[group.index].mean()
                mean_targets[blood_count] = means.loc[target_type]
            
            # compute baseline score
            algorithm = f'Baseline - Event Rate Per Blood Count Bin'.title()
            for split in splits:
                Y = self.labels[split]
                X = self.orig_data.loc[Y.index]
                blood_count_bins = X[base_col].round(1)
                Y_true = Y[target_type]
                Y_pred_prob = blood_count_bins.map(mean_targets).fillna(0)
                col = (split, target_type)
                score_df.loc[(algorithm, 'AUROC Score'), col] = np.round(roc_auc_score(Y_true, Y_pred_prob), digit_round)
                score_df.loc[(algorithm, 'AUPRC Score'), col] = np.round(average_precision_score(Y_true, Y_pred_prob), digit_round)

        return score_df
    
    def baseline_score(self, score_df, splits=None, target_types=None, base_cols=None, digit_round=3):
        """This baseline model predicts the target mean for each entry in a specified column of the training set
        Column must only have one unique entry per row.
        """
        if target_types is None: target_types = self.target_types
        if splits is None: splits = self.splits
        if base_cols is None: base_cols = ['regimen']
        
        for base_col in base_cols:
            # compute mean of target for each categorical entry of a column in training set
            mean_targets = {target_type: {} for target_type in target_types}
            Y = self.labels['Train']
            X = self.orig_data.loc[Y.index]
            for entry, group in X.groupby(base_col):
                means = Y.loc[group.index].mean()
                for target_type, mean in means.items():
                    mean_targets[target_type][entry] = mean
        
            # compute baseline score
            col_name = clean_variable_mapping.get(base_col, base_col).replace('_', ' ')
            algorithm = f'Baseline - Event Rate Per {col_name}'.title()
            for split in splits:
                Y = self.labels['Train']
                X = self.orig_data.loc[Y.index]
                for target_type in target_types:
                    Y_true = Y[target_type]
                    Y_pred_prob = X[base_col].map(mean_targets[target_type]).fillna(0)
                    col = (split, target_type)
                    score_df.loc[(algorithm, 'AUROC Score'), col] = np.round(roc_auc_score(Y_true, Y_pred_prob), digit_round)
                    score_df.loc[(algorithm, 'AUPRC Score'), col] = np.round(average_precision_score(Y_true, Y_pred_prob), digit_round)
        return score_df

    def get_evaluation_scores(self, algorithms=None, target_types=None, splits=None, 
                              get_baseline=False, baseline_cols=None, get_cyto_baseline=False, 
                              display_ci=False, load_ci=False, save_ci=False,
                              save_score=True, pred_thresh=None, digit_round=3, verbose=True):
        """Evaluate the best models, compute the AUROC, AUPRC, F1, etc scores.
        
        Args:
            display_ci (bool): display confidence interval for AUROC and AUPRC
            load_ci (bool): load saved bootstrapped AUROC and AUPRC scores for computing confidence interval
            pred_thresh (float or None): value between [0,1]. If None, will not compute F1, accuracy, precision, recall score
            verbose (bool): print confusion matrix for the test split for each target
        """    
        if algorithms is None: algorithms = self.models
        if target_types is None: target_types = self.target_types
        if splits is None: splits = self.splits
            
        score_df = pd.DataFrame(index=twolevel, columns=twolevel)
        if get_baseline: 
            score_df = self.baseline_score(score_df, splits=splits, target_types=target_types, 
                                           base_cols=baseline_cols, digit_round=digit_round)
        if get_cyto_baseline:
            score_df = self.cyto_baseline_score(score_df, splits=splits, target_types=target_types, digit_round=digit_round)
            
        ci_df = pd.DataFrame(index=twolevel)
        if load_ci:
            ci_df = pd.read_csv(f'{self.output_path}/confidence_interval/bootstrapped_scores.csv', index_col=[0,1])
            ci_df.columns = ci_df.columns.astype(int)

        for algorithm in algorithms:
            for split in splits:
                pred_prob, Y = self.preds[split][algorithm], self.labels[split]
                for target_type in target_types:
                    Y_true = Y[target_type]
                    Y_pred_prob = pred_prob[target_type]
                    col = (split, target_type)
                    score_df.loc[(algorithm, 'AUROC Score'), col] = np.round(roc_auc_score(Y_true, Y_pred_prob), digit_round)
                    score_df.loc[(algorithm, 'AUPRC Score'), col] = np.round(average_precision_score(Y_true, Y_pred_prob), digit_round)
                    if display_ci: 
                        score_df = self.get_confidence_interval(Y_true, Y_pred_prob, algorithm, col, score_df, ci_df)
                    
                    if pred_thresh is not None:
                        Y_pred_bool = Y_pred_prob > pred_thresh
                        report = classification_report(Y_true, Y_pred_bool, output_dict=True, zero_division=1)
                        score_df.loc[(algorithm, 'Accuracy'), col] = np.round(report['accuracy'], digit_round)
                        # predicted true positive over all predicted positive
                        score_df.loc[(algorithm, 'Precision'), col] = np.round(report['True']['precision'], digit_round)
                        # predicted true positive over all true positive (aka senstivity)
                        score_df.loc[(algorithm, 'Recall'), col] = np.round(report['True']['recall'], digit_round)
                        # 2*precision*recall / (precision + recall)
                        score_df.loc[(algorithm, 'F1 Score'), col] = np.round(report['True']['f1-score'], digit_round)
                        
                        if verbose and split == 'Test':
                            cm = confusion_matrix(Y_true, Y_pred_bool)
                            cm = pd.DataFrame(cm, columns=['Predicted False', 'Predicted True'], index=['Actual False', 'Actual True'])
                            print(f"\n######## {algorithm} - {split} - {target_type} #########")
                            print(cm)
        
        if save_score: score_df.to_csv(f'{self.output_path}/tables/classification_results.csv')
        if save_ci: ci_df.to_csv(f'{self.output_path}/confidence_interval/bootstrapped_scores.csv')
            
        return score_df

    def plot_curve(self, algorithm, target_types, ax, curve_type='roc', legend_location='auto'):
        if curve_type == 'pr':
            curve_function, score_function = precision_recall_curve, average_precision_score
            label_name, xlabel, ylabel = 'AUPRC', 'Recall', 'Precision'
        elif curve_type == 'roc':
            curve_function, score_function = roc_curve, roc_auc_score
            label_name, xlabel, ylabel = 'AUROC', 'False Positive Rate', 'True Positive Rate'
        
        pred, Y = self.preds['Test'][algorithm], self.labels['Test']
        for target_type in target_types:
            y_true = Y[target_type]
            y_scores = pred[target_type]
            x, y, thresholds = curve_function(y_true, y_scores)
            score = np.round(score_function(y_true, y_scores), 2)
            label = target_type + f' ({label_name}={score})'
            if curve_type == 'pr': x, y = y, x
            ax.plot(x, y, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(algorithm)
        ax.legend(loc=legend_location)

    def plot_curves(self, curve_type='roc', target_types=None, figsize=(12,9), legend_location='lower left'):
        if curve_type not in {'roc', 'pr'}: 
            raise ValueError("curve_type must be set to 'roc' or 'pr'")
        if target_types is None: target_types = self.target_types
            
        nrows, ncols = int(len(self.models)/2), 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.3)
        
        for idx, algorithm in enumerate(self.models):
            self.plot_curve(algorithm, target_types, axes[idx], curve_type=curve_type, legend_location=legend_location)
        plt.savefig(f'{self.output_path}/figures/curves/{curve_type}.jpg', bbox_inches='tight', dpi=300)

    def plot_calibs(self, target_types=None, figsize=(12,9), savefig=True, filename='calib',
                    include_pred_hist=False, n_bins=20, calib_strategy='uniform'):
        if target_types is None: target_types = self.target_types
        
        nrows = int(len(self.models)/2)
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0.3)
        gs = GridSpec(3*nrows, 2)

        for idx, algorithm in enumerate(self.models):
            if include_pred_hist:
                row = int(idx / 2) * 3
                col = idx % 2
                ax = fig.add_subplot(gs[row:row+2, col])
            else:
                ax = fig.add_subplot(nrows, 2, idx+1)
                
            axis_max_limit = 0
            pred_prob, Y = self.preds['Valid'][algorithm], self.labels['Valid']
            for target_type in target_types:
                y_true = Y[target_type]
                y_pred = pred_prob[target_type]
                prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=calib_strategy)
                axis_max_limit = np.max([axis_max_limit, prob_true.max(), prob_pred.max()])
                ax.plot(prob_pred, prob_true, label=target_type)
            ax.plot([0,axis_max_limit],[0,axis_max_limit],'k:', label='perfect calibration')
            ax.set_xlabel('Predicted Probability') # binned predicted probability (e.g. 0.0-0.1, 0.1-0.2, etc) - mean of each bin
            ax.set_ylabel('Empirical Probability') # fraction of positive class in each bin
            ax.set_title(algorithm)  
            ax.legend()
            
            if include_pred_hist:
                ax = fig.add_subplot(gs[row+2, col], sharex=ax)
                for target_type in target_types:
                    y_pred = pred_prob[target_type]
                    ax.hist(y_pred, range=(0,axis_max_limit), bins=n_bins, label=target_type, histtype='step')
                ax.set_xlabel('Predicted Probability')
                ax.set_ylabel('Count')
                
        if include_pred_hist:
            plt.tight_layout()
            
        if savefig:
            plt.savefig(f'{self.output_path}/figures/curves/{filename}.jpg', bbox_inches='tight', dpi=300)
        
    def plot_cdf_pred(self, target_types=None, figsize=(12,9)): 
        if target_types is None: target_types = self.target_types
            
        nrows = int(len(self.models)/2)
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0.3)
            
        for idx, algorithm in enumerate(self.models):
            pred_prob, Y = self.preds['Test'][algorithm], self.labels['Test']
            N = len(Y)
            ax = fig.add_subplot(nrows, 2, idx+1)
            for target_type in target_types:
                x = pred_prob[target_type]
                x = np.sort(x)
                y = np.arange(N) / float(N)
                plt.plot(x, y, label=target_type)
            plt.title(f'CDF of {algorithm} Predictions')
            plt.xlabel('Predictions')
            plt.ylabel('Cumulative Probability')
            plt.legend()
        plt.savefig(f'{self.output_path}/figures/curves/cdf_pred.jpg', bbox_inches='tight', dpi=300)
    
    def all_plots_for_single_target(self, algorithm='XGB', target_type='ACU', split='Test',
                                    n_bins=10, n_samples_for_ci=10, calib_strategy='quantile', figsize=(12,11)):
        # setup
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        
        pred_prob, Y = self.preds[split][algorithm], self.labels[split]
        y_true = Y[target_type]
        y_pred = pred_prob[target_type]
        
        # get 95% confidence interval
        col = (split, target_type)
        ci_df = pd.read_csv(f'{self.output_path}/confidence_interval/bootstrapped_scores.csv', index_col=[0,1])
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(y_true, y_pred, algorithm, col, ci_df)
        ap_lower, ap_upper = np.percentile(auprc_scores, [2.5, 97.5]).round(3)
        roc_lower, roc_upper = np.percentile(auroc_scores, [2.5, 97.5]).round(3)

        # get results
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        ap_score = np.round(average_precision_score(y_true, y_pred), 3)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_score = np.round(roc_auc_score(y_true, y_pred), 3)
        
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins*n_samples_for_ci, strategy=calib_strategy)
        prob_true = np.pad(prob_true, (0, n_bins*n_samples_for_ci - len(prob_true)), mode='edge')
        prob_pred = np.pad(prob_pred, (0, n_bins*n_samples_for_ci - len(prob_pred)), mode='edge')
        # save the calibration numbers
        np.save(f'{self.output_path}/figures/curves/calib_true_array.npy', prob_true)
        np.save(f'{self.output_path}/figures/curves/calib_pred_array.npy', prob_pred)
        prob_true, prob_pred = prob_true.reshape(n_bins, n_samples_for_ci), prob_pred.reshape(n_bins, n_samples_for_ci)
        mean_prob_true, mean_prob_pred = prob_true.mean(axis=1), prob_pred.mean(axis=1)
        lower_limit, upper_limit = prob_true[:, 0] - mean_prob_true, prob_true[:, -1] - mean_prob_true
        yerr = abs(np.array([lower_limit, upper_limit]))
        axis_max_limit = max(mean_prob_true.max(), mean_prob_pred.max())
        max_calib_error = np.max(abs(mean_prob_true - mean_prob_pred)).round(3)
        mean_calib_error = np.mean(abs(mean_prob_true - mean_prob_pred)).round(3)
        
        N = len(Y)
        x, y = np.sort(y_pred), np.arange(N) / float(N)

        # plot
        axes[0].plot(recall, precision, label=f'AUPRC={ap_score} (95% CI: {ap_lower}-{ap_upper})')
        axes[0].set_ylim(0, 1)
        axes[1].plot(fpr, tpr, label=f'AUROC={roc_score} (95% CI: {roc_lower}-{roc_upper})')
        axes[2].errorbar(mean_prob_pred, mean_prob_true, yerr=yerr, capsize=5.0, ecolor='firebrick')
        axes[2].text(axis_max_limit/2, 0.07, f'Mean Calibration Error {mean_calib_error}')
        axes[2].text(axis_max_limit/2, 0.1, f'Max Calibration Error {max_calib_error}')
        axes[2].plot([0,axis_max_limit],[0,axis_max_limit],'k:', label='Perfect Calibration')
        axes[2].set_ylim(-0.01, axis_max_limit+0.01)
        axes[2].set_ylim(-0.01, axis_max_limit+0.01)
        axes[3].plot(x, y)
        
        # label
        labels = [('Sensitivity', 'Positive Predictive Value', 'pr', True),
                  ('1 - Specificity', 'Sensitivity', 'roc', True),
                  ('Predicted Probability', 'Empirical Probability', 'calib', False),
                  (f'Predicted Probability of {target_type}', 'Cumulative Proportion of Predictions', 'cdf_pred', False)]
        for idx, (xlabel, ylabel, filename, remove_legend_line) in enumerate(labels):
            axes[idx].set_xlabel(xlabel)
            axes[idx].set_ylabel(ylabel)
            leg = axes[idx].legend(loc='lower right', frameon=False)
            if remove_legend_line: leg.legendHandles[0].set_linewidth(0)
            fig.savefig(f'{self.output_path}/figures/curves/{algorithm}_{target_type}_{filename}.jpg', 
                        bbox_inches=get_bbox(axes[idx], fig), dpi=300) 
        plt.savefig(f'{self.output_path}/figures/curves/{algorithm}_{target_type}.jpg', bbox_inches='tight', dpi=300)
        plt.show()
        
    def threshold_op_points(self, pred_thresholds, algorithm, target_types=None, 
                            split='Test', include_outcome_recall=False, event_dates=None, digit_round=3):
        if target_types is None: target_types = self.target_types
        df = pd.DataFrame(columns=twolevel)
        df.index.name = 'Prediction Threshold'
        
        pred_prob, Y = self.preds[split][algorithm], self.labels[split]
        
        if include_outcome_recall:
            if event_dates is None: 
                raise ValueError('Please provide the event dates. See PrepDataEDHD class. Note only ED/H events are supported')
            event_dates = event_dates.loc[Y.index]
            event_dates['ikn'] = self.orig_data.loc[Y.index, 'ikn']
        
        for target_type in tqdm.tqdm(target_types):
            Y_true = Y[target_type]
            Y_pred_prob = pred_prob[target_type]
            
            if include_outcome_recall:
                event = target_type.split('_')[-1] # events will be either ACU, ED, or H
                if event == 'ACU':
                    mask = event_dates['next_H_date'].notnull() | event_dates['next_ED_date'].notnull()
                else:
                    mask = event_dates[f'next_{event}_date'].notnull()
                event_dates['true'] = Y_true
                worker = partial(group_pred_by_outcome, event=event)
                
            for threshold in pred_thresholds:
                threshold = np.round(threshold, 2)
                Y_pred_bool = Y_pred_prob > threshold
                df.loc[threshold, (target_type, 'Warning Rate')] = Y_pred_bool.mean()
                df.loc[threshold, (target_type, 'PPV')] = precision_score(Y_true, Y_pred_bool, zero_division=1)
                name = 'Trigger-Level Recall' if include_outcome_recall else 'Sensitivity'
                df.loc[threshold, (target_type, name)] = recall_score(Y_true, Y_pred_bool, zero_division=1)
                if include_outcome_recall:
                    event_dates['pred'] = Y_pred_bool
                    grouped_preds = split_and_parallelize(event_dates[mask], worker, processes=8)
                    grouped_preds = pd.DataFrame(grouped_preds, columns=['chemo_idx', 'pred']).set_index('chemo_idx')
                    result = pd.concat([event_dates.loc[grouped_preds.index], event_dates[~mask]]) # select the rows of interest
                    result.loc[grouped_preds.index, 'pred'] = grouped_preds['pred'] # update the predictions
                    df.loc[threshold, (target_type, 'Outcome-Level Recall')] = recall_score(result['true'], result['pred'], zero_division=1)
                df.loc[threshold, (target_type, 'NPV')] = precision_score(~Y_true, ~Y_pred_bool, zero_division=1)
                df.loc[threshold, (target_type, 'Specificity')] = recall_score(~Y_true, ~Y_pred_bool, zero_division=1)
        df = df.round(digit_round)
        df.to_csv(f'{self.output_path}/tables/threshold_performance.csv')
        return df
    
    def plot_rnn_training_curve(self, train_losses, valid_losses, train_scores, valid_scores, 
                                figsize=(12,9), save=False, save_path=None):
        fig = plt.figure(figsize=figsize)
        data = {'Training Loss': train_losses, 'Validation Loss': valid_losses, 
                'Training Accuracy': train_scores, 'Validation Accuracy': valid_scores}
        plt.subplots_adjust(hspace=0.3)
        for i, (title, value) in enumerate(data.items()):
            ax = fig.add_subplot(2, 2, i+1)
            plt.plot(range(len(value)), value)
            plt.title(title)
            plt.legend(self.target_types)
        if save: plt.savefig(save_path)
        plt.show()
