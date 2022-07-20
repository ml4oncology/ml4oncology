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
import tqdm
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.proportion import proportion_confint
from sklearn.calibration import calibration_curve
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, 
                             average_precision_score, precision_recall_curve,
                             confusion_matrix)
from scripts.preprocess import (split_and_parallelize)
from scripts.utility import (twolevel, compute_bootstrap_scores, group_pred_by_outcome, pred_thresh_binary_search)
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
        self.ci_df = pd.DataFrame(index=twolevel) # bootstrapped scores for confidence interval
        
    def load_bootstrapped_scores(self):
        self.ci_df = pd.read_csv(f'{self.output_path}/confidence_interval/bootstrapped_scores.csv', index_col=[0,1])
        self.ci_df.columns = self.ci_df.columns.astype(int)
        
    def get_bootstrapped_scores(self, Y_true, Y_pred_prob, algorithm, split, target_type, n_bootstraps=10000):
        ci_index = f'{algorithm}_{split}_{target_type}'
        if ci_index not in self.ci_df.index:
            auc_scores = compute_bootstrap_scores(Y_true, Y_pred_prob, n_bootstraps=n_bootstraps)
            logging.info(f'Completed bootstrap computations for {ci_index}')
            auroc_scores, auprc_scores = np.array(auc_scores).T
            self.ci_df.loc[(ci_index, 'AUROC'), range(n_bootstraps)] = auroc_scores
            self.ci_df.loc[(ci_index, 'AUPRC'), range(n_bootstraps)] = auprc_scores
        return self.ci_df.loc[ci_index].values
        
    def get_confidence_interval(self, Y_true, Y_pred_prob, algorithm, col, score_df):
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(Y_true, Y_pred_prob, algorithm, *col)
        for name, scores in [('AUROC Score', auroc_scores), ('AUPRC Score', auprc_scores)]:
            lower, upper = np.percentile(scores, [2.5, 97.5]).round(3)
            score_df.loc[(algorithm, name), col] = f'{score_df.loc[(algorithm, name), col]} ({lower}-{upper})'
        return score_df
    
    def get_calibration_confidence_interval(self, y_true, y_pred, n_bins, calib_strategy):
        """
        Get confidence interval for the true probabilities (proportion of positive labels in each 
        predicted probability bin) of the calibration curve

        Reference: github.com/scikit-learn/scikit-learn/blob/main/sklearn/clibration.py#L895
        """
        if calib_strategy == 'quantile':
            quantiles = np.linspace(0, 1, n_bins+1)
            bins = np.percentile(y_pred, quantiles*100)
        elif calib_strategy == 'uniform':
            bins = np.linspace(0, 1.0, n_bins+1)
        else:
            raise ValueError('calib_strategy must be either quantile or uniform')

        # compute which bin each label belongs to
        bin_ids = np.digitize(y_pred, bins[1:-1]) - 1
        # WARNING: Newer sklearn version uses the line below
        #          Ensure you use the correct line according to your sklearn version
        # bin_ids = np.searchsorted(bins[1:-1], y_pred)

        y_true = pd.Series(np.array(y_true), index=bin_ids)
        lower_limit, upper_limit = [], []
        for idx, group in y_true.groupby(y_true.index): # iterate through each bin
            # compute 95% CI for that bin using binormial confidence interval (because of binormial distribution (i.e. True, False))
            lower, upper = proportion_confint(count=group.sum(), nobs=len(group), method='wilson', alpha=(1-0.95))
            lower_limit.append(lower)
            upper_limit.append(upper)
        return np.array(lower_limit), np.array(upper_limit)
    
    def baseline_score(self, score_df, splits=None, target_types=None, base_cols=None, digit_round=3):
        """This baseline model outputs the corresponding target rate of
        1. each entry in a categorical column
        2. each bin in a numerical column
        from the training set
        
        E.g. if patient is taking regimen X, baseline model outputs target rate of regimen X in the training set
        E.g. if patient's blood count measurement is X, baseline model outputs target rate of blood count bin in which X belongs to
        
        Why not predict previously measured blood count directly? Because we need prediction probability to calculate AUROC.
        Predicting if previous blood count is less than x threshold will output a 0 or 1, resulting in a single point on the ROC curve.
        """
        if target_types is None: target_types = self.target_types
        if splits is None: splits = self.splits
        if base_cols is None: base_cols = ['regimen']
        override_variable_mapping = {f'baseline_{bt}_count': 'Blood Count' for bt in blood_types}
        override_variable_mapping.update({'baseline_eGFR': 'Gloemrular Filteration Rate'})
        
        for base_col in base_cols:
            mean_targets = {target_type: {} for target_type in target_types}
            Y = self.labels['Train']
            X = self.orig_data.loc[Y.index, base_col]
            numerical_col = X.dtype == float
            if numerical_col: X, bins = pd.cut(X, bins=100, retbins=True)
            # compute target rate of each categorical entry or numerical bin of the column
            for group_name, group in X.groupby(X):
                means = Y.loc[group.index].mean()
                for target_type, mean in means.items():
                    mean_targets[target_type][group_name] = mean
            
            # get baseline algorithm name
            name = clean_variable_mapping.get(base_col, base_col)
            name = override_variable_mapping.get(base_col, name)
            if numerical_col: name += ' Bin'
            algorithm = f'Baseline - Event Rate Per {name}'.replace('_', ' ').title()
            
            # special case for blood count measurements (don't get baseline score for a different blood count target)
            bt = base_col.replace('baseline_', '').replace('_count', '') # e.g. baseline_neutrophil_count -> neutrophil
            target_names = [blood_types[bt]['cytopenia_name']] if bt in blood_types else target_types
            
            # compute baseline score
            for split in splits:
                Y = self.labels[split]
                X = self.orig_data.loc[Y.index, base_col]
                if numerical_col: X = pd.cut(X, bins=bins).astype(object)
                for target_type in target_names:
                    Y_true = Y[target_type]
                    Y_pred_prob = X.map(mean_targets[target_type]).fillna(0)
                    col = (split, target_type)
                    score_df.loc[(algorithm, 'AUROC Score'), col] = np.round(roc_auc_score(Y_true, Y_pred_prob), digit_round)
                    score_df.loc[(algorithm, 'AUPRC Score'), col] = np.round(average_precision_score(Y_true, Y_pred_prob), digit_round)
        return score_df

    def get_evaluation_scores(self, algorithms=None, target_types=None, splits=None, 
                              display_ci=False, load_ci=False, save_ci=False,
                              get_baseline=False, baseline_cols=None, 
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
        if load_ci: self.load_bootstrapped_scores()
            
        score_df = pd.DataFrame(index=twolevel, columns=twolevel)
        if get_baseline: 
            score_df = self.baseline_score(score_df, splits=splits, target_types=target_types, 
                                           base_cols=baseline_cols, digit_round=digit_round)
            
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
                        score_df = self.get_confidence_interval(Y_true, Y_pred_prob, algorithm, col, score_df)
                    
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
        if save_ci: self.ci_df.to_csv(f'{self.output_path}/confidence_interval/bootstrapped_scores.csv')
            
        return score_df

    def plot_auc_curve(self, ax, algorithm, target_types, split='Test', curve_type='roc', legend_location='best', title=''):
        # setup
        if curve_type == 'pr':
            curve_function, score_function = precision_recall_curve, average_precision_score
            label_name, xlabel, ylabel = 'AUPRC', 'Sensitivity', 'Positive Predictive Value' # aka Recall, Precision
        elif curve_type == 'roc':
            curve_function, score_function = roc_curve, roc_auc_score
            label_name, xlabel, ylabel = 'AUROC', '1 - Specificity', 'Sensitivity', # aka False Positive Rate, True Positive Rate'
        curve_idx = int(curve_type == 'pr')
        one_target = len(target_types) == 1
        
        pred, Y = self.preds[split][algorithm], self.labels[split]
        for target_type in target_types: 
            # get the curve numbers
            y_true = Y[target_type]
            y_scores = pred[target_type]
            x, y, thresholds = curve_function(y_true, y_scores)
            if curve_type == 'pr': x, y = y, x
            # get the score and 95% CI
            score = np.round(score_function(y_true, y_scores), 3)
            ci_scores = self.get_bootstrapped_scores(y_true, y_scores, algorithm, split, target_type)[curve_idx]
            lower, upper = np.percentile(ci_scores, [2.5, 97.5]).round(3)
            label = f'{label_name}={score} (95% CI: {lower}-{upper})'
            if not one_target: label = f'{target_type}\n{label}'
            # plot it
            ax.plot(x, y, label=label)
            
        if title: ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if curve_type == 'pr': ax.set_ylim(-0.05, 1.05)
        leg = ax.legend(loc=legend_location, frameon=False)
        if one_target: leg.legendHandles[0].set_linewidth(0) # remove the legend line
            
    def plot_pred_cdf(self, ax, algorithm, target_types, split='Test', legend_location='best', title=''): 
        one_target = len(target_types) == 1
        pred, Y = self.preds[split][algorithm], self.labels[split]
        N = len(Y)
        y = np.arange(N) / float(N)
        for target_type in target_types: 
            x = np.sort(pred[target_type])
            label = None if one_target else target_type
            ax.plot(x, y, label=label)
        if title: ax.set_title(title)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Cumulative Proportion of Predictions')
        if not one_target: ax.legend(loc=legend_location, frameon=False)

    def plot_curves(self, curve_type='roc', target_types=None, legend_location='best', 
                    figsize=(12,18), padding=None, save=True):
        if curve_type not in {'roc', 'pr', 'pred_cdf'}: 
            raise ValueError("curve_type must be set to 'roc', 'pr', or 'pred_cdf'")
        if target_types is None: target_types = self.target_types
        if padding is None: padding = {'pad_y1': 0.3}
        if self.ci_df.empty: self.load_bootstrapped_scores()
            
        nrows, ncols = len(self.models)//2, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, algorithm in enumerate(self.models):
            if curve_type in {'roc', 'pr'}:
                self.plot_auc_curve(axes[idx], algorithm, target_types, curve_type=curve_type, 
                                    legend_location=legend_location, title=algorithm)
            elif curve_type == 'pred_cdf':
                self.plot_pred_cdf(axes[idx], algorithm, target_types, 
                                   legend_location=legend_location, title=algorithm)
            if save: 
                filename = f'{self.output_path}/figures/curves/{algorithm}_{curve_type}.jpg'
                fig.savefig(filename, bbox_inches=get_bbox(axes[idx], fig, **padding), dpi=300) 
                
        if save: 
            filename = f'{self.output_path}/figures/curves/{curve_type}.jpg'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            
        plt.show()
        
    def plot_calib(self, ax, algorithm, target_types, split='Test', legend_location='best', title='', 
                   n_bins=10, calib_strategy='quantile', calib_ci=False, save=False):
        if calib_ci and len(target_types) > 1:
            raise ValueError('Displaying calibration confidence interval with more than one target will \
                              make the plot messy. Do not set calib_ci to True when there are multiple targets')
        pred, Y = self.preds[split][algorithm], self.labels[split]
        for target_type in target_types:
            y_true = Y[target_type]
            y_pred = pred[target_type]
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=calib_strategy)
            axis_max_limit = max(prob_true.max(), prob_pred.max())
            max_calib_error = np.max(abs(prob_true - prob_pred)).round(3)
            mean_calib_error = np.mean(abs(prob_true - prob_pred)).round(3)
            if save:
                # save the calibration numbers
                np.save(f'{self.output_path}/figures/curves/{target_type}_calib_true_array.npy', prob_true)
                np.save(f'{self.output_path}/figures/curves/{target_type}_calib_pred_array.npy', prob_pred)
            if calib_ci:
                lower_limit, upper_limit = self.get_calibration_confidence_interval(y_true, y_pred, n_bins, calib_strategy)
                lower_limit, upper_limit = lower_limit - prob_true, upper_limit - prob_true
                yerr = abs(np.array([lower_limit, upper_limit]))
                ax.errorbar(prob_pred, prob_true, yerr=yerr, capsize=5.0, errorevery=n_bins//10, ecolor='firebrick')
                adjustment_factor = 1 if axis_max_limit > 0.25 else 3
                ax.text(axis_max_limit/2, 0.07/adjustment_factor, f'Mean Calibration Error {mean_calib_error}')
                ax.text(axis_max_limit/2, 0.1/adjustment_factor, f'Max Calibration Error {max_calib_error}')
                ax.set_ylim(-0.01, axis_max_limit+0.01)
            else:
                label = f'{target_type}\nMax Error={max_calib_error}, Mean Error={mean_calib_error}'
                ax.plot(prob_pred, prob_true, label=label)
        ax.plot([0,axis_max_limit],[0,axis_max_limit],'k:', label='Perfect Calibration')
        ax.set_xlabel('Predicted Probability') # binned predicted probability (e.g. 0.0-0.1, 0.1-0.2, etc) - mean of each bin
        ax.set_ylabel('Empirical Probability') # fraction of positive class in each bin
        if title: ax.set_title(title)
        ax.legend(loc=legend_location, frameon=False)
        
    def plot_calibs(self, target_types=None, split='Test', include_pred_hist=False, n_bins=10, 
                    calib_strategy='quantile', legend_location='best', figsize=(12,18), padding=None, save=True):
        if target_types is None: target_types = self.target_types
        if padding is None: padding = {'pad_y1': 0.3}
        
        nrows = len(self.models)//2
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3*nrows, 2, hspace=0.5)

        for idx, algorithm in enumerate(self.models):
            if include_pred_hist:
                row = int(idx / 2) * 3
                col = idx % 2
                ax = fig.add_subplot(gs[row:row+2, col])
            else:
                ax = fig.add_subplot(nrows, 2, idx+1)
                
            self.plot_calib(ax, algorithm, target_types, split=split, title=algorithm, 
                            n_bins=n_bins, calib_strategy=calib_strategy, legend_location=legend_location)
            
            if include_pred_hist:
                axis_max_limit = max(ax.axis())
                ax_hist = fig.add_subplot(gs[row+2, col], sharex=ax)
                for target_type in target_types:
                    y_pred = self.preds[split][algorithm][target_type]
                    ax_hist.hist(y_pred, range=(0,axis_max_limit), bins=n_bins, label=target_type, histtype='step')
                formatter = FuncFormatter(lambda y, pos: int(y*1e-3))
                ax_hist.yaxis.set_major_formatter(formatter)
                ax_hist.set_xlabel('Predicted Probability')
                ax_hist.set_ylabel('Count (In Thousands)')
            
            if save: 
                filename = f'{self.output_path}/figures/curves/{algorithm}_calib.jpg'
                fig.savefig(filename, bbox_inches=get_bbox(ax, fig, **padding), dpi=300) 
        if save:
            filename = f'{self.output_path}/figures/curves/calib.jpg'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            
        plt.show()
        
    def all_plots_for_single_target(self, algorithm='XGB', target_type='ACU', split='Test', calib_ci=True, 
                                    n_bins=10, calib_strategy='quantile', figsize=(12,11), save=True):
        # setup 
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        
        # plot
        self.plot_auc_curve(axes[0], algorithm, [target_type], split=split, curve_type='pr', legend_location='lower right')
        self.plot_auc_curve(axes[1], algorithm, [target_type], split=split, curve_type='roc', legend_location='lower right')
        self.plot_calib(axes[2], algorithm, [target_type], split=split, legend_location='lower right', 
                        n_bins=n_bins, calib_strategy=calib_strategy, calib_ci=calib_ci, save=save)
        self.plot_pred_cdf(axes[3], algorithm, [target_type], split=split)
        
        # save
        if save:
            for idx, filename in enumerate(['pr','roc','calib', 'pred_cdf']):
                filename = f'{self.output_path}/figures/curves/{algorithm}_{target_type}_{filename}.jpg'
                fig.savefig(filename, bbox_inches=get_bbox(axes[idx], fig), dpi=300) 
            filename = f'{self.output_path}/figures/curves/{algorithm}_{target_type}.jpg'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()
        
    def operating_points(self, algorithm, points, metric='threshold', target_types=None, split='Test', 
                         include_outcome_recall=False, event_dates=None, digit_round=3, save=True):
        """
        Evaluate how system performs at different operating points 
        (e.g. at different prediction thresholds, desired precision, desired sensitivity)
        """
        if metric not in {'threshold', 'precision', 'sensitivity'}: 
            raise ValueError('metric must be set to threshold, precision, or sensitivity')
        if target_types is None: target_types = self.target_types
        df = pd.DataFrame(columns=twolevel)
        df.index.name = metric.title()
        
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
                event = target_type.split('_')[-1] # event will be either ACU, ED, H, or '' for Death
                if event == 'ACU':
                    mask = event_dates['next_H_date'].notnull() | event_dates['next_ED_date'].notnull()
                elif event == 'ED' or event == 'H':
                    mask = event_dates[f'next_{event}_date'].notnull()
                else:
                    mask = event_dates['D_date'].notnull()
                event_dates['true'] = Y_true
                worker = partial(group_pred_by_outcome, event=event)
                
            for point in points:
                point = np.round(point, 2)
                
                if metric in {'precision', 'sensitivity'}:
                    # LOL binary search the threshold to get desired precision or sensitivity
                    threshold = pred_thresh_binary_search(Y_pred_prob, Y_true, desired_target=point, metric=metric)
                else:
                    threshold = point
                Y_pred_bool = Y_pred_prob > threshold
                
                df.loc[point, (target_type, 'Warning Rate')] = Y_pred_bool.mean()
                
                if metric != 'threshold':
                    df.loc[point, (target_type, 'Prediction Threshold')] = threshold
                
                if metric != 'precision':
                    df.loc[point, (target_type, 'PPV')] = precision_score(Y_true, Y_pred_bool, zero_division=1)
                    
                if metric != 'sensitivity':
                    name = 'Trigger-Level Recall' if include_outcome_recall else 'Recall'
                    df.loc[point, (target_type, name)] = recall_score(Y_true, Y_pred_bool, zero_division=1)
                    
                if include_outcome_recall:
                    event_dates['pred'] = Y_pred_bool
                    grouped_preds = split_and_parallelize(event_dates[mask], worker, processes=8)
                    grouped_preds = pd.DataFrame(grouped_preds, columns=['chemo_idx', 'pred']).set_index('chemo_idx')
                    result = pd.concat([event_dates.loc[grouped_preds.index], event_dates[~mask]]) # select the rows of interest
                    result.loc[grouped_preds.index, 'pred'] = grouped_preds['pred'] # update the predictions
                    df.loc[point, (target_type, 'Outcome-Level Recall')] = recall_score(result['true'], result['pred'], zero_division=1)
                    
                df.loc[point, (target_type, 'NPV')] = precision_score(~Y_true, ~Y_pred_bool, zero_division=1)
                df.loc[point, (target_type, 'Specificity')] = recall_score(~Y_true, ~Y_pred_bool, zero_division=1)
        df = df.round(digit_round)
        if save: df.to_csv(f'{self.output_path}/tables/{metric}_performance.csv')
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
        
    def plot_decision_curve_analysis(self, algorithm, target_types=None, split='Test', 
                                     xlim=None, figsize=(), padding=None):
        # setup 
        if target_types is None: target_types = self.target_types
        N = len(target_types)
        if not figsize: figsize = (6, 5*N)
        if padding is None: padding = {}
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0.3)
        
        pred, Y = self.preds[split][algorithm], self.labels[split]
        for idx, target_type in enumerate(target_types):
            targets, predictions = Y[target_type], pred[target_type]
            # compute net benefit for model, treat all, treat none
            fpr, tpr, threshold = roc_curve(targets, predictions)
            sensitivity, specificity, prevalence = tpr, 1 - fpr, targets.mean()
            net_benefit = sensitivity*prevalence - (1 - specificity)*(1 - prevalence)*threshold
            treat_all = prevalence - ((1 - prevalence)*threshold) / (1 - threshold)
            threshold, net_benefit, treat_all = threshold[1:], net_benefit[1:], treat_all[1:]
            treat_none = np.zeros(threshold.shape)
            # plot decision curve analysis
            ax = fig.add_subplot(N,1,idx+1)
            y_max = 0
            for label, y in [('System', net_benefit), ('All', treat_all), ('None', treat_none)]:
                y_max = max(y_max, y.max())
                ax.plot(threshold, y, label=label)
            ax.set_xlabel('Threshold Probability')
            ax.set_ylabel('Net Benefit')
            if xlim is not None: ax.set_xlim(*xlim)
            ax.set_ylim((y_max/-4, y_max+y_max*0.1)) # bound/scale the y axis to make plot look nicer
            ax.legend(frameon=False)
            fig.savefig(f'{self.output_path}/figures/decision_curve/{algorithm}_{target_type}.jpg', 
                        bbox_inches=get_bbox(ax, fig, **padding), dpi=300) 
            ax.set_title(target_type) # set title AFTER saving individual figures
        plt.savefig(f'{self.output_path}/figures/decision_curve/{algorithm}.jpg', bbox_inches='tight', dpi=300)
        plt.show()
        
class EvaluateBaselineModel(Evaluate):
    def __init__(self, base_col, preds_min, preds_max, preds, labels, orig_data, output_path):
        super().__init__(output_path, preds, labels, orig_data)
        self.col = base_col
        self.preds_min = preds_min
        self.preds_max = preds_max
        
        save_path = f'{output_path}/figures/baseline'
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        self.name_mapping = {'baseline_eGFR': 'Pre-Treatment eGFR', 'next_eGFR': '9-month Post-Treatment eGFR'}
        self.name_mapping.update({f'baseline_{bt}_count': f'Pre-Treatment {bt.title()} Count' for bt in blood_types})
        self.name_mapping.update({f'target_{bt}_count': f'Before Next Treatment {bt.title()} Count' for bt in blood_types})
        
    def plot_loess(self, ax, algorithm, target_type, split):            
        x = self.orig_data[self.col].sort_values()
        pred = self.preds[split][algorithm].loc[x.index, target_type]
        pred_min = self.preds_min.loc[x.index, target_type]
        pred_max = self.preds_max.loc[x.index, target_type]
        ax.plot(x, pred)
        ax.fill_between(x, pred_min, pred_max, alpha=0.25)
        xlabel = self.name_mapping.get(self.col, self.col)
        ylabel = f'Prediction for {self.name_mapping.get(target_type, target_type)}'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
            
    def all_plots_for_single_target(self, algorithm='LOESS', target_type='AKI', split='Test',
                                    n_bins=10, calib_strategy='quantile', figsize=(12,12), 
                                    save=True, filename=''):
        # setup 
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        # plot
        self.plot_loess(axes[0], algorithm, target_type, split=split)
        self.plot_auc_curve(axes[1], algorithm, [target_type], split=split, curve_type='pr', legend_location='lower right')
        self.plot_auc_curve(axes[2], algorithm, [target_type], split=split, curve_type='roc', legend_location='lower right')
        # hotfix - bound the predictions to (0, 1) for calibration
        tmp = self.preds[split][algorithm][target_type].copy() 
        self.preds[split][algorithm][target_type] = self.preds[split][algorithm][target_type].clip(lower=0, upper=1) 
        self.plot_calib(axes[3], algorithm, [target_type], split=split, legend_location='lower right', 
                        n_bins=n_bins, calib_strategy=calib_strategy)
        self.preds[split][algorithm][target_type] = tmp
        
        # save
        if save:
            if not filename: filename = algorithm
            savepath = f'{self.output_path}/figures/baseline/{target_type}_{filename}.jpg'
            plt.savefig(savepath, bbox_inches='tight', dpi=300)
        plt.show()
        
    def all_plots(self, split='Test', **kwargs):
        Y = self.labels[split]
        for target_type, Y_true in Y.iteritems():
            if Y_true.nunique() < 2: continue # no pos examples, no point in plotting/evaluating
            self.all_plots_for_single_target(target_type=target_type, **kwargs)
