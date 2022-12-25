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
from functools import partial
import os

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, 
    average_precision_score,
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    roc_curve, 
)
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (blood_types, clean_variable_mapping)
from src.utility import twolevel
from src.utility import (
    compute_bootstrap_scores, 
    outcome_recall_score, 
    pred_thresh_binary_search,
    split_and_parallelize, 
)
from src.visualize import get_bbox

import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%I:%M:%S'
)

class Evaluate:
    """Evaluate any/all of ML (Machine Learning), RNN (Recurrent Neural Network),
    and ENS (Ensemble) models
    
    Attributes:
        ci_df (pd.DataFrame): table of bootstrapped AUROC and AUPRC scores
            for computing confidence interval
        splits (list): a sequence of data splits (str) to evaluate our models on
    """
    def __init__(self, output_path, preds, labels, orig_data, digit_round=3):
        """
        Args:
            orig_data (pd.DataFrame): the original dataset before one-hot 
                encoding and splitting
            preds (dict): mapping of data splits (str) and their associated 
                predictions by each algorithm (dict of str: pd.DataFrame)
            labels (dict): mapping of data splits (str) and their associated 
                labels (pd.Series)
        """
        self.output_path = output_path
        self.preds = preds
        self.labels = labels
        self.splits = ['Valid', 'Test']
        self.models = list(preds['Test'].keys())
        self.target_events = list(labels['Test'].columns)
        self.orig_data = orig_data
        self.round = lambda x: np.round(x, digit_round)
        # bootstrapped scores for confidence interval
        self.ci_df = pd.DataFrame(index=twolevel)
        
    def load_bootstrapped_scores(self, filename='bootstrapped_scores'):
        filepath = f'{self.output_path}/confidence_interval/{filename}.csv'
        self.ci_df = pd.read_csv(filepath, index_col=[0,1])
        self.ci_df.columns = self.ci_df.columns.astype(int)
        
    def save_bootstrapped_scores(self, filename='bootstrapped_scores'):
        filepath = f'{self.output_path}/confidence_interval/{filename}.csv'
        self.ci_df.to_csv(filepath)
        
    def get_bootstrapped_scores(
        self, 
        Y_true, 
        Y_pred_prob, 
        algorithm, 
        split, 
        target_event, 
        subgroup_name=None,
        n_bootstraps=10000
    ):
        ci_index = f'{algorithm}_{split}_{target_event}'
        if subgroup_name is not None:
            ci_index = f'{ci_index}_{subgroup_name}'
        if ci_index not in self.ci_df.index:
            auc_scores = compute_bootstrap_scores(
                Y_true, Y_pred_prob, n_bootstraps=n_bootstraps
            )
            logging.info(f'Completed bootstrap computations for {ci_index}')
            auroc_scores, auprc_scores = np.array(auc_scores).T
            self.ci_df.loc[(ci_index, 'AUROC'), range(n_bootstraps)] = auroc_scores
            self.ci_df.loc[(ci_index, 'AUPRC'), range(n_bootstraps)] = auprc_scores
        return self.ci_df.loc[ci_index].values
        
    def get_auc_confidence_interval(
        self, 
        Y_true, 
        Y_pred_prob, 
        algorithm, 
        col, 
        score_df
    ):
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(
            Y_true, Y_pred_prob, algorithm, *col
        )
        for name, scores in [('AUROC Score', auroc_scores), 
                             ('AUPRC Score', auprc_scores)]:
            lower, upper = np.percentile(scores, [2.5, 97.5]).round(3)
            score = score_df.loc[(algorithm, name), col]
            if not (lower <= score <= upper): 
                logging.warning(f'{algorithm} {name} {score} does not lie '
                                f'between {lower} and {upper}')
            score_df.loc[(algorithm, name), col] = f'{score} ({lower}-{upper})'
        return score_df
    
    def get_calibration_confidence_interval(
        self, 
        y_true, 
        y_pred, 
        n_bins, 
        calib_strategy
    ):
        """Get confidence interval for the true probabilities (proportion of
        positive labels in each predicted probability bin) of the calibration 
        curve

        Reference: 
            [1] github.com/scikit-learn/scikit-learn/blob/main/sklearn/clibration.py#L895
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
        # Ensure you use the correct line according to your sklearn version
        # bin_ids = np.searchsorted(bins[1:-1], y_pred)

        y_true = pd.Series(np.array(y_true), index=bin_ids)
        lower_limit, upper_limit = [], []
        for idx, group in y_true.groupby(y_true.index): # loop through each bin
            # compute 95% CI for that bin using binormial confidence interval 
            # (because of binormial distribution (i.e. True, False))
            lower, upper = proportion_confint(
                count=group.sum(), nobs=len(group), method='wilson', 
                alpha=(1-0.95)
            )
            lower_limit.append(lower)
            upper_limit.append(upper)
        return np.array(lower_limit), np.array(upper_limit)
    
    def baseline_score(
        self, 
        score_df, 
        base_cols, 
        splits=None, 
        target_events=None, 
        display_ci=True
    ):
        """This baseline model outputs the corresponding target rate of
        1. each category of a categorical column
        2. each bin of a numerical column
        from the training set. 
        
        E.g. if patient is taking regimen X, baseline model outputs target rate 
             of regimen X in the training set
        E.g. if patient's blood count measurement is X, baseline model outputs 
             target rate of blood count bin in which X belongs to
             
        Each column in base_cols acts as a baseline model.
        
        Why not predict previously measured blood count directly? 
        Because we need prediction probability to calculate AUROC.
        Predicting if previous blood count is less than x threshold will output 
        a 0 or 1, resulting in a single point on the ROC curve.
        """
        if target_events is None: target_events = self.target_events
        if splits is None: splits = self.splits
        override_variable_mapping = {f'baseline_{bt}_count': 'Blood Count' 
                                     for bt in blood_types}
        override_variable_mapping.update(
            {'baseline_eGFR': 'Gloemrular Filteration Rate'}
        )
        
        for base_col in base_cols:
            mean_targets = {target_event: {} for target_event in target_events}
            Y = self.labels['Train']
            X = self.orig_data.loc[Y.index, base_col]
            numerical_col = X.dtype == float
            if numerical_col: X, bins = pd.cut(X, bins=100, retbins=True)
            # compute target rate of each category or numerical bin of the 
            # column
            for group_name, group in X.groupby(X):
                means = Y.loc[group.index].mean()
                for target_event, mean in means.items():
                    mean_targets[target_event][group_name] = mean
            
            # get baseline algorithm name
            name = clean_variable_mapping.get(base_col, base_col)
            name = override_variable_mapping.get(base_col, name)
            if numerical_col: name += ' Bin'
            algorithm = f'Baseline - Event Rate Per {name}'
            algorithm = algorithm.replace('_', ' ').title()
            
            # special case for blood count measurements 
            # don't get baseline score for a different blood count target
            bt = base_col.replace('baseline_', '').replace('_count', '')
            if bt in blood_types:
                target_names = [blood_types[bt]['cytopenia_name']]
            else:
                target_names = target_events
            
            # compute baseline score
            for split in splits:
                Y = self.labels[split]
                X = self.orig_data.loc[Y.index, base_col]
                if numerical_col: X = pd.cut(X, bins=bins).astype(object)
                for target_event in target_names:
                    Y_true = Y[target_event]
                    Y_pred_prob = X.map(mean_targets[target_event]).fillna(0)
                    col = (split, target_event)
                    auroc = self.round(roc_auc_score(Y_true, Y_pred_prob))
                    auprc = self.round(average_precision_score(Y_true, Y_pred_prob))
                    score_df.loc[(algorithm, 'AUROC Score'), col] = auroc
                    score_df.loc[(algorithm, 'AUPRC Score'), col] = auprc
                    
                    if display_ci: 
                        score_df = self.get_auc_confidence_interval(
                            Y_true, Y_pred_prob, algorithm, col, score_df
                        )
        return score_df

    def get_evaluation_scores(
        self, 
        algorithms=None, 
        target_events=None, 
        splits=None, 
        pred_thresh=None, 
        baseline_cols=None, 
        display_ci=False, 
        load_ci=False, 
        save_ci=False, 
        save_score=True, 
        verbose=True
    ):
        """Compute the AUROC, AUPRC, accuracy, precision, recall, F1 scores
        for each given algorithm and data split
        
        Args:
            pred_thresh (float): value between [0,1]. If None, will not compute 
                F1, accuracy, precision, recall score
            baseline_cols (list): a sequence of variable names (str) to compute 
                baseline scores (each variable is used as a single baseline 
                model). If None, no baseline model scores will be measured
            display_ci (bool): display confidence interval for AUROC and AUPRC
            load_ci (bool): If True load saved bootstrapped AUROC and AUPRC 
                scores for computing confidence interval
            verbose (bool): print confusion matrix for the test split for each 
                target event
        """    
        if algorithms is None: algorithms = self.models
        if target_events is None: target_events = self.target_events
        if splits is None: splits = self.splits
        if load_ci: self.load_bootstrapped_scores()
            
        score_df = pd.DataFrame(index=twolevel, columns=twolevel)
        if baseline_cols is not None: 
            score_df = self.baseline_score(
                score_df, baseline_cols, splits=splits, 
                target_events=target_events, display_ci=display_ci
            )
            
        for alg in algorithms:
            for split in splits:
                pred_prob, Y = self.preds[split][alg], self.labels[split]
                for target_event in target_events:
                    Y_true = Y[target_event]
                    Y_pred_prob = pred_prob[target_event]
                    col = (split, target_event)
                    auroc = self.round(roc_auc_score(Y_true, Y_pred_prob))
                    auprc = self.round(average_precision_score(Y_true, Y_pred_prob))
                    score_df.loc[(alg, 'AUROC Score'), col] = auroc
                    score_df.loc[(alg, 'AUPRC Score'), col] = auprc
                    if display_ci: 
                        score_df = self.get_auc_confidence_interval(
                            Y_true, Y_pred_prob, alg, col, score_df
                        )
                    
                    if pred_thresh is not None:
                        Y_pred_bool = Y_pred_prob > pred_thresh
                        report = classification_report(
                            Y_true, Y_pred_bool, output_dict=True, zero_division=1
                        )
                        accuracy = self.round(report['accuracy'])
                        score_df.loc[(alg, 'Accuracy'), col] = accuracy
                        # predicted true positive over all predicted positive
                        precision = self.round(report['True']['precision'])
                        score_df.loc[(alg, 'Precision'), col] = precision
                        # predicted true positive over all true positive
                        recall = self.round(report['True']['recall'])
                        score_df.loc[(alg, 'Recall'), col] = recall
                        # 2*precision*recall / (precision + recall)
                        f1_score = self.round(report['True']['f1-score'])
                        score_df.loc[(alg, 'F1 Score'), col] = f1_score
                        
                        if verbose and split == 'Test':
                            cm = confusion_matrix(Y_true, Y_pred_bool)
                            cm = pd.DataFrame(
                                cm, 
                                columns=['Predicted False', 'Predicted True'], 
                                index=['Actual False', 'Actual True']
                            )
                            print(f"\n#######################################")
                            print(f"# {alg} - {split} - {target_event}")
                            print(f"\n#######################################")
                            print(cm)
        
        if save_score: 
            filepath = f'{self.output_path}/tables/classification_results.csv'
            score_df.to_csv(filepath)
        if save_ci: self.save_bootstrapped_scores()
            
        return score_df

    def plot_auc_curve(
        self, 
        ax, 
        algorithm, 
        target_events,
        split='Test', 
        curve_type='roc', 
        legend_location='best', 
        remove_legend_line=False,
        title=None, 
        ylim=(-0.05, 1.05),
        mask=None,
        mask_name=None,
    ):
        """
        Args:
            mask (pd.Series): An alignable boolean series to filter samples in 
                the corresponding data split. If None, no filtering is done.
            mask_name (str): The name associated with the masking (e.g. Female
                Only). Default is None. If mask is provided, mask_name must 
                also be provided.
        """
        # setup
        if curve_type == 'pr':
            curve_function = precision_recall_curve
            score_function = average_precision_score
            label_name = 'AUPRC'
            xlabel, ylabel = 'Sensitivity', 'Positive Predictive Value'
        elif curve_type == 'roc':
            curve_function = roc_curve
            score_function = roc_auc_score
            label_name = 'AUROC'
            xlabel, ylabel = '1 - Specificity', 'Sensitivity'
        curve_idx = int(curve_type == 'pr')
        one_target = len(target_events) == 1
        
        pred, Y = self.preds[split][algorithm], self.labels[split]
        if mask is not None:
            if mask_name is None:
                raise ValueError('please provide a mask_name')
            pred, Y = pred[mask], Y[mask]
            
        for target_event in target_events: 
            # get the curve numbers
            y_true = Y[target_event]
            y_scores = pred[target_event]
            x, y, thresholds = curve_function(y_true, y_scores)
            if curve_type == 'pr': x, y = y, x
            # get the score and 95% CI
            score = self.round(score_function(y_true, y_scores))
            ci_scores = self.get_bootstrapped_scores(
                y_true, y_scores, algorithm, split, target_event, 
                subgroup_name=mask_name
            )[curve_idx]
            lower, upper = np.percentile(ci_scores, [2.5, 97.5]).round(3)
            label = f'{label_name}={score} (95% CI: {lower}-{upper})'
            if not one_target: label = f'{target_event}\n{label}'
            if mask_name is not None: label = f'{mask_name} {label}'
            # plot it
            ax.plot(x, y, label=label)
        
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, ylim=ylim)
        leg = ax.legend(loc=legend_location, frameon=False)
        if remove_legend_line: 
            leg.legendHandles[0].set_linewidth(0)
            
    def plot_pred_cdf(
        self, 
        ax, 
        algorithm, 
        target_events, 
        split='Test', 
        legend_location='best', 
        title=None
    ): 
        one_target = len(target_events) == 1
        pred, Y = self.preds[split][algorithm], self.labels[split]
        N = len(Y)
        y = np.arange(N) / float(N)
        for target_event in target_events: 
            x = np.sort(pred[target_event])
            label = None if one_target else target_event
            ax.plot(x, y, label=label)
        ax.set(
            title=title, xlabel='Predicted Probability', 
            ylabel='Cumulative Proportion of Predictions'
        )
        if not one_target: ax.legend(loc=legend_location, frameon=False)

    def plot_curves(
        self, 
        curve_type='roc', 
        target_events=None, 
        legend_location='best', 
        figsize=(12,18), 
        padding=None, 
        save=True
    ):
        if curve_type not in {'roc', 'pr', 'pred_cdf'}: 
            raise ValueError("curve_type must be set to roc, pr, or pred_cdf")
        if target_events is None: target_events = self.target_events
        if padding is None: padding = {'pad_y1': 0.3}
        if self.ci_df.empty: self.load_bootstrapped_scores()
            
        nrows, ncols = len(self.models)//2, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, alg in enumerate(self.models):
            if curve_type in {'roc', 'pr'}:
                self.plot_auc_curve(
                    axes[idx], alg, target_events, curve_type=curve_type, 
                    legend_location=legend_location, title=alg
                )
            elif curve_type == 'pred_cdf':
                self.plot_pred_cdf(
                    axes[idx], alg, target_events, 
                    legend_location=legend_location, title=alg
                )
            if save: 
                fig.savefig(
                    f'{self.output_path}/figures/curves/{alg}_{curve_type}.jpg',
                    bbox_inches=get_bbox(axes[idx], fig, **padding), dpi=300
                ) 
                
        if save: 
            filepath = f'{self.output_path}/figures/curves/{curve_type}.jpg'
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            
        plt.show()
        
    def plot_calib(
        self, 
        ax, 
        algorithm, 
        target_events, 
        split='Test', 
        legend_location='best', 
        title=None, 
        n_bins=10, 
        calib_strategy='quantile', 
        calib_ci=False, 
        show_perf_calib=True,
        mask=None,
        mask_name=None,
        save=False
    ):
        """
        Args:
            mask (pd.Series): An alignable boolean series to filter samples in 
                the corresponding data split. If None, no filtering is done.
            mask_name (str): The name associated with the masking (e.g. Female
                Only). Default is None. If mask is provided, mask_name must 
                also be provided.
        """
        if calib_ci and len(target_events) > 1:
            raise ValueError('Displaying calibration confidence interval with '
                             'more than one target will make the plot messy. Do '
                             'not set calib_ci to True when there are multiple '
                             'targets')
        one_target = len(target_events) == 1
            
        pred, Y = self.preds[split][algorithm], self.labels[split]
        if mask is not None:
            if mask_name is None:
                raise ValueError('please provide a mask_name')
            pred, Y = pred[mask], Y[mask]
            
        for target_event in target_events:
            y_true = Y[target_event]
            y_pred = pred[target_event]
            # bin predicted probability (e.g. 0.0-0.1, 0.1-0.2, etc) 
            # prob_true: fraction of positive class in each bin
            # prob_pred: mean of each bin
            prob_true, prob_pred = calibration_curve(
                y_true, y_pred, n_bins=n_bins, strategy=calib_strategy
            )
            axis_max_limit = max(prob_true.max(), prob_pred.max())
            max_calib_error = np.max(abs(prob_true - prob_pred)).round(3)
            mean_calib_error = np.mean(abs(prob_true - prob_pred)).round(3)
            if save:
                # save the calibration numbers
                np.save(f'{self.output_path}/figures/curves/'
                        f'{target_event}_calib_true_array.npy', prob_true)
                np.save(f'{self.output_path}/figures/curves/'
                        f'{target_event}_calib_pred_array.npy', prob_pred)
            if calib_ci:
                lower_limit, upper_limit = self.get_calibration_confidence_interval(
                    y_true, y_pred, n_bins, calib_strategy
                )
                lower_limit -= prob_true
                upper_limit -= prob_true
                yerr = abs(np.array([lower_limit, upper_limit]))
                ax.errorbar(
                    prob_pred, prob_true, yerr=yerr, capsize=5.0, 
                    errorevery=n_bins//10, ecolor='firebrick'
                )
                adjustment_factor = 1 if axis_max_limit > 0.25 else 3
                ax.text(
                    axis_max_limit/2, 0.07/adjustment_factor, 
                    f'Mean Calibration Error {mean_calib_error}'
                )
                ax.text(
                    axis_max_limit/2, 0.1/adjustment_factor, 
                    f'Max Calibration Error {max_calib_error}'
                )
                ax.set_ylim(-0.01, axis_max_limit+0.01)
            else:
                label = (f'Max Error={max_calib_error} '
                         f'Mean Error={mean_calib_error}')
                if not one_target: label = f'{target_event}\n{label}'
                if mask_name is not None: label = f'{mask_name} {label}'
                ax.plot(prob_pred, prob_true, label=label)
        
        if show_perf_calib:
            ax.plot(
                [0,axis_max_limit], [0,axis_max_limit], 'k:', 
                label='Perfect Calibration'
            )
            
        ax.set(
            title=title, xlabel='Predicted Probability', 
            ylabel='Empirical Probability'
        )
        ax.legend(loc=legend_location, frameon=False)
        
    def plot_calibs(
        self, 
        target_events=None, 
        split='Test', 
        include_pred_hist=False, 
        n_bins=10,        
        calib_strategy='quantile', 
        legend_location='best', 
        figsize=(12,18), 
        padding=None, 
        save=True
    ):
        if target_events is None: target_events = self.target_events
        if padding is None: padding = {'pad_y1': 0.3}
        
        nrows = len(self.models)//2
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3*nrows, 2, hspace=0.5)

        for idx, alg in enumerate(self.models):
            if include_pred_hist:
                row = int(idx / 2) * 3
                col = idx % 2
                ax = fig.add_subplot(gs[row:row+2, col])
            else:
                ax = fig.add_subplot(nrows, 2, idx+1)
                
            self.plot_calib(
                ax, alg, target_events, split=split, title=alg, n_bins=n_bins, 
                calib_strategy=calib_strategy, legend_location=legend_location
            )
            
            if include_pred_hist:
                axis_max_limit = max(ax.axis())
                ax_hist = fig.add_subplot(gs[row+2, col], sharex=ax)
                for target_event in target_events:
                    y_pred = self.preds[split][alg][target_event]
                    ax_hist.hist(
                        y_pred, range=(0,axis_max_limit), bins=n_bins,
                        label=target_event, histtype='step'
                    )
                formatter = FuncFormatter(lambda y, pos: int(y*1e-3))
                ax_hist.yaxis.set_major_formatter(formatter)
                ax_hist.set_xlabel('Predicted Probability')
                ax_hist.set_ylabel('Count (In Thousands)')
            
            if save: 
                fig.savefig(
                    f'{self.output_path}/figures/curves/{alg}_calib.jpg', 
                    bbox_inches=get_bbox(ax, fig, **padding), dpi=300
                ) 
        if save:
            filepath = f'{self.output_path}/figures/curves/calib.jpg'
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            
        plt.show()
        
    def all_plots_for_single_target(
        self, 
        alg='XGB', 
        target_event='ACU', 
        split='Test', 
        calib_ci=True, 
        n_bins=10, 
        calib_strategy='quantile', 
        figsize=(12,12), 
        save=True
    ):
        # setup 
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        
        # plot
        self.plot_auc_curve(
            axes[0], alg, [target_event], split=split, curve_type='pr', 
            legend_location='lower right', remove_legend_line=True, 
        )
        self.plot_auc_curve(
            axes[1], alg, [target_event], split=split, curve_type='roc', 
            legend_location='lower right', remove_legend_line=True, 
        )
        self.plot_calib(
            axes[2], alg, [target_event], split=split, n_bins=n_bins, 
            legend_location='lower right', calib_strategy=calib_strategy, 
            calib_ci=calib_ci, save=save
        )
        self.plot_decision_curve(axes[3], alg, target_event, split)
        # self.plot_pred_cdf(axes[3], alg, [target_event], split=split)
        
        # save
        if save:
            for idx, filename in enumerate(['pr', 'roc', 'calib', 'dc']):
                filename = f'{alg}_{target_event}_{filename}'
                fig.savefig(
                    f'{self.output_path}/figures/curves/{filename}.jpg', 
                    bbox_inches=get_bbox(axes[idx], fig), dpi=300
                ) 
            plt.savefig(
                f'{self.output_path}/figures/curves/{alg}_{target_event}.jpg',
                bbox_inches='tight', dpi=300
            )
        plt.show()
        
    def operating_points(
        self, 
        algorithm, 
        points, 
        metric='threshold', 
        target_events=None, 
        split='Test',        
        include_outcome_recall=False, 
        mask=None,
        save=True, 
        **kwargs
    ):
        """Evaluate how system performs at different operating points 
        (e.g. prediction thresholds, desired precision, desired sensitivity)
        
        Args:
            include_outcome_recall (bool): If True, include outcome-level 
                recall scores. Only ED/H/D events are supported.
            mask (pd.Series): An alignable boolean series to filter samples in 
                the corresponding data split. If None, no filtering is done.
            **kwargs: keyword arguments fed into outcome_recall_score (if we 
                are including outcome-level recall)
        """
        if metric not in {'threshold', 'precision', 'sensitivity', 'warning_rate'}:
            raise ValueError('metric must be set to threshold, precision, '
                             'sensitivity, or warning_rate')
            
        if target_events is None: target_events = self.target_events
        df = pd.DataFrame(columns=twolevel)
        df.index.name = metric.title()
        
        pred_prob, Y = self.preds[split][algorithm], self.labels[split]
        if mask is not None: pred_prob, Y = pred_prob[mask], Y[mask]
        
        for target_event in tqdm(target_events):
            Y_true = Y[target_event]
            Y_pred_prob = pred_prob[target_event]
                
            for point in points:
                point = np.round(point, 2)
                
                if metric in {'precision', 'sensitivity', 'warning_rate'}:
                    # LOL binary search the threshold to get desired precision,
                    # sensitivity, or warning_rate
                    threshold = pred_thresh_binary_search(
                        Y_pred_prob, Y_true, desired_target=point, metric=metric,
                        zero_division=1
                    )
                else:
                    threshold = point
                    
                Y_pred_bool = Y_pred_prob > threshold
                
                if metric != 'warning_rate':
                    col = (target_event, 'Warning Rate')
                    df.loc[point, col] = Y_pred_bool.mean()
                
                if metric != 'threshold':
                    col = (target_event, 'Prediction Threshold')
                    df.loc[point, col] = threshold
                
                if metric != 'precision':
                    col = (target_event, 'PPV')
                    df.loc[point, col] = precision_score(
                        Y_true, Y_pred_bool, zero_division=1
                    )
                    
                if metric != 'sensitivity':
                    name = 'Recall' 
                    if include_outcome_recall: 
                        name = f'Trigger-Level {name}'
                        
                    col = (target_event, name)
                    df.loc[point, col] = recall_score(
                        Y_true, Y_pred_bool, zero_division=1
                    )
                    
                if include_outcome_recall:
                    col = (target_event, 'Outcome-Level Recall')
                    df.loc[point, col] = outcome_recall_score(
                        Y_true, Y_pred_bool, target_event, **kwargs
                    )
                
                col = (target_event, 'NPV')
                df.loc[point, col] = precision_score(
                    ~Y_true, ~Y_pred_bool, zero_division=1
                )
                
                col = (target_event, 'Specificity')
                df.loc[point, col] = recall_score(
                    ~Y_true, ~Y_pred_bool, zero_division=1
                )
                
        df = self.round(df)
        if save: 
            df.to_csv(f'{self.output_path}/tables/{metric}_performance.csv')
            
        return df
    
    def plot_rnn_training_curve(
        self, 
        train_losses, 
        valid_losses, 
        train_scores, 
        valid_scores, 
        figsize=(12,9), 
        save=False, 
        save_path=None
    ):
        fig = plt.figure(figsize=figsize)
        data = {
            'Training Loss': train_losses, 
            'Validation Loss': valid_losses, 
            'Training Accuracy': train_scores, 
            'Validation Accuracy': valid_scores
        }
        plt.subplots_adjust(hspace=0.3)
        for i, (title, value) in enumerate(data.items()):
            ax = fig.add_subplot(2, 2, i+1)
            plt.plot(range(len(value)), value)
            plt.title(title)
            plt.legend(self.target_events)
        if save: plt.savefig(save_path)
        plt.show()
        
    def plot_decision_curve(
        self, 
        ax, 
        algorithm, 
        target_event, 
        split, 
        xlim=None, 
        colors=None
    ):
        if colors is None: 
            colors = {'System': '#1f77b4', 'All': '#bcbd22', 'None': '#2ca02c'}
        targets = self.labels[split][target_event]
        predictions = self.preds[split][algorithm][target_event]
            
        # compute net benefit for model and treat all
        fpr, tpr, thresh = roc_curve(targets, predictions)
        sensitivity, specificity, prevalence = tpr, 1 - fpr, targets.mean()
        net_benefit = sensitivity*prevalence - (1 - specificity)*(1 - prevalence)*thresh
        treat_all = prevalence - ((1 - prevalence)*thresh) / (1 - thresh)
        thresh, net_benefit, treat_all = thresh[1:], net_benefit[1:], treat_all[1:]
        df = pd.DataFrame(
            data=np.array([thresh, net_benefit, treat_all]).T, 
            columns=['Threshold', 'System', 'All']
        )
        
        # plot decision curve analysis
        y_max = 0
        for label, y in df[['System', 'All']].iteritems():
            y_max = max(y_max, y.max())
            ax.plot(thresh, y, label=label, color=colors[label])
        ax.plot(
            thresh, np.zeros(thresh.shape), label='None', color=colors['None'],
            linestyle='--'
        )
        ax.set(
            xlabel='Threshold Probability', ylabel='Net Benefit', xlim=xlim, 
            # bound/scale the y axis to make plot look nicer
            ylim=(y_max/-4, y_max*1.1) 
        )
        ax.legend(frameon=False)
        
        return df
        
    def plot_decision_curves(
        self, 
        algorithm, 
        target_events=None, 
        split='Test', 
        xlim=None, 
        save=True, 
        figsize=(), 
        padding=None
    ):
        """Plots net benefit vs threshold probability, with "intervention for 
        all" and "intervention for none" lines. Threshold probability can 
        indicate the preference of doctors (missing disease vs unnecessary 
        intervention).
        
        E.g. Threshold probability = 10%
             Missing a disease is 9 times worse than doing an unnecessary 
             intervention. Worth to do 10 interventions to find 1 disease
        
        Net benefit can indicate the benefit (true positives) minus the cost 
        (false positives times the number of false positives that are worth 
        one true positive)
        
        The absolute number of net benfit is true positives.
        
        E.g. Net Benefit = 0.07
             Identifying correctly 7 patients with disease out of every 100 patients
        
        Reference:
            [1] https://diagnprognres.biomedcentral.com/articles/10.1186/s41512-019-0064-7
        """
        # setup 
        if target_events is None: target_events = self.target_events
        N = len(target_events)
        if not figsize: figsize = (6, 6*N)
        if padding is None: padding = {}
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        result = {}
        
        for idx, target_event in enumerate(target_events):
            ax = fig.add_subplot(N,1,idx+1)
            result[target_event] = self.plot_decision_curve(
                ax, algorithm, target_event, split, xlim=xlim
            )
            if save:
                filename = f'{algorithm}_{target_event}'
                fig.savefig(
                    f'{self.output_path}/figures/decision_curve/{filename}.jpg',
                    bbox_inches=get_bbox(ax, fig, **padding), dpi=300
                ) 
            ax.set_title(target_event) # set title AFTER saving individual figures
            
        if save:
            plt.savefig(
                f'{self.output_path}/figures/decision_curve/{algorithm}.jpg', 
                bbox_inches='tight', dpi=300
            )
        plt.show()
        
        return result
        
class EvaluateBaselineModel(Evaluate):
    def __init__(
        self, 
        base_col, 
        preds_min, 
        preds_max, 
        preds, 
        labels, 
        orig_data, 
        output_path
    ):
        super().__init__(output_path, preds, labels, orig_data)
        self.col = base_col
        self.preds_min = preds_min
        self.preds_max = preds_max
        
        save_path = f'{output_path}/figures/baseline'
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        self.name_mapping = {
            'baseline_eGFR': 'Pre-Treatment eGFR', 
            'next_eGFR': '9-month Post-Treatment eGFR'
        }
        self.name_mapping.update(
            {f'baseline_{bt}_count': f'Pre-Treatment {bt.title()} Count' 
             for bt in blood_types}
        )
        self.name_mapping.update(
            {f'target_{bt}_count': f'Before Next Treatment {bt.title()} Count'
             for bt in blood_types}
        )
        
    def plot_loess(self, ax, alg, target_event, split):
        x = self.orig_data[self.col].sort_values()
        pred = self.preds[split][alg].loc[x.index, target_event]
        pred_min = self.preds_min.loc[x.index, target_event]
        pred_max = self.preds_max.loc[x.index, target_event]
        ax.plot(x, pred)
        ax.fill_between(x, pred_min, pred_max, alpha=0.25)
        xlabel = self.name_mapping.get(self.col, self.col)
        ylabel = f'Prediction for {self.name_mapping.get(target_event, target_event)}'
        ax.set(xlabel=xlabel, ylabel=ylabel)
            
    def all_plots_for_single_target(
        self, 
        alg='LOESS', 
        target_event='AKI', 
        split='Test',
        n_bins=10, 
        calib_strategy='quantile', 
        figsize=(12,12), 
        save=True, 
        filename=''
    ):
        # setup 
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        # plot
        self.plot_loess(axes[0], alg, target_event, split=split)
        self.plot_auc_curve(
            axes[1], alg, [target_event], split=split, curve_type='pr', 
            legend_location='lower right'
        )
        self.plot_auc_curve(
            axes[2], alg, [target_event], split=split, curve_type='roc', 
            legend_location='lower right'
        )
        # hotfix - bound the predictions to (0, 1) for calibration
        tmp = self.preds[split][alg][target_event].copy() 
        self.preds[split][alg][target_event].clip(lower=0, upper=1, inplace=True)
        self.plot_calib(
            axes[3], alg, [target_event], split=split, n_bins=n_bins, 
            legend_location='lower right', calib_strategy=calib_strategy
        )
        self.preds[split][alg][target_event] = tmp
        
        # save
        if save:
            if not filename: filename = alg
            filename = f'{target_event}_{filename}'
            plt.savefig(
                f'{self.output_path}/figures/baseline/{filename}.jpg', 
                bbox_inches='tight', dpi=300
            )
        plt.show()
        
    def all_plots(self, split='Test', **kwargs):
        Y = self.labels[split]
        for target_event, Y_true in Y.iteritems():
            if Y_true.nunique() < 2: 
                # no pos examples, no point in plotting/evaluating
                continue
            self.all_plots_for_single_target(target_event=target_event, **kwargs)
