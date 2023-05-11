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
from collections import defaultdict
from functools import partial
import itertools
import math
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import logging
from src.conf_int import (
    AUCConfidenceInterval,
    get_confidence_interval,
    get_calibration_confidence_interval,
)
from src.config import (
    blood_types, 
    cancer_code_mapping,
    clean_variable_mapping
)
from src.summarize import SubgroupSummary
from src.utility import (
    split_and_parallelize,
    get_clean_variable_name,
    get_clean_variable_names,
    group_pred_by_outcome,
    pred_thresh_binary_search,
    twolevel
)
from src.visualize import get_bbox

def get_perf_at_operating_point(
    point, 
    Y_true,
    Y_pred_prob, 
    op_metric='threshold', 
    perf_metrics=None, 
    include_ci=False,
    event_df=None,
    **outcome_recall_kwargs
):
    """Evaluate how system performs at an operating point
    (e.g. prediction threshold, desired precision, desired recall)

    Args:
        perf_metrics (list): a sequence of performance metrics (str) to 
            calculate. If None, will calculate all available performance
            metrics
        event_df (pd.DataFrame): table of relevant event dates 
            (e.g. D_date, next_ED_date, visit_date, etc) associated with 
            each session. Also includes ikn (patient id). If None, certain 
            performance metrics can't be calculated
        outcome_recall_kwargs (dict): a mapping of keyword arguments fed into
            outcome_recall_score if it is calculated. Note only ED/H/D events 
            are supported for outcome_recall_score
    """
    all_kwargs = {'zero_division': 1}

    metrics = ['threshold', 'warning_rate', 'precision', 'recall']
    if op_metric not in metrics:
        raise ValueError(f'op_metric must be set to one of {metrics}')

    if perf_metrics is None:
        perf_metrics = [
            'event_rate', 'NPV', 'specificity', 'outcome_level_recall',
            'first_warning_rate', 'first_warning_precision',
        ] + metrics

    if op_metric != 'threshold':
        # binary search the threshold to get desired warning rate, 
        # precision, or sensitivity,
        thresh = pred_thresh_binary_search(
            Y_true, Y_pred_prob, desired_target=point, metric=op_metric,
            **all_kwargs
        )
    else:
        thresh = point
    Y_pred_bool = Y_pred_prob > thresh
    
    def get_ci(mask):
        lower, upper = get_confidence_interval(mask, method='binomial')
        return f'({lower:.3f}-{upper:.3f})'

    result = {}
    if 'count' in perf_metrics:
        result['N'] = len(Y_true)
    if 'threshold' in perf_metrics:
        result['Prediction Threshold'] = thresh
    if 'event_rate' in perf_metrics:
        score = Y_true.mean()
        if include_ci: score = f'{score:.3f} {get_ci(Y_true)}'
        result['Event Rate'] = score
    if 'warning_rate' in perf_metrics:
        # Proportion of treatments
        # where a warning was issued
        result['Warning Rate'] = Y_pred_bool.mean()
    if 'precision' in perf_metrics:
        # Proportion of treatments in which a warning was issued
        # where target event occurs
        score = precision_score(Y_true, Y_pred_bool, **all_kwargs)
        if include_ci: score = f'{score:.3f} {get_ci(Y_true[Y_pred_bool])}'
        result['PPV'] = score
    if 'recall' in perf_metrics:  
        # Proportion of treatments in which target event occurs
        # where a warning was issued
        score = recall_score(Y_true, Y_pred_bool, **all_kwargs)
        if include_ci: score = f'{score:.3f} {get_ci(Y_pred_bool[Y_true])}'
        result['Recall'] = score
    if 'NPV' in perf_metrics:
        # Proportion of treatments in which target event does not occur
        # where a warning was not issued
        result['NPV'] = precision_score(~Y_true, ~Y_pred_bool, **all_kwargs)
    if 'specificity' in perf_metrics:
        # Proportion of treatments in which a warning was not issued
        # where target event does not occurs
        result['Specificity'] = recall_score(~Y_true, ~Y_pred_bool, **all_kwargs)
    if 'outcome_level_recall' in perf_metrics:
        # Proportion of target events 
        # where at least one warning was issued in the lookback window
        score = outcome_recall_score(
            Y_true, Y_pred_bool, event_df=event_df, include_ci=include_ci,
            **outcome_recall_kwargs
        )
        if include_ci:
            score, lower, upper = score
            score = f'{score:.3f} ({lower:.3f}-{upper:.3f})'
        result['Outcome-Level Recall'] = score

    if event_df is None:
        return result

    # filter out treatments after the first warning
    tmp = event_df.loc[Y_pred_bool.index].copy()
    tmp['pred'] = Y_pred_bool
    cumsum = tmp.groupby('ikn')['pred'].cumsum()
    # keep treatments where first warning occured
    mask = (cumsum == 1) & (cumsum != cumsum.shift())
    # keep treatments prior to first warning
    mask |= cumsum < 1

    if 'first_warning_rate' in perf_metrics:
        # Proportion of treatments
        # where a first warning was issued
        # (we ignore all treatments after first warning)
        result['First Warning Rate'] = Y_pred_bool[mask].mean()

    if 'first_warning_precision' in perf_metrics:
        # Proportion of treatments in which a first warning was issued
        # where target event occurs
        result['First Warning PPV'] = precision_score(
            Y_true[mask], Y_pred_bool[mask], **all_kwargs
        )

    return result

def outcome_recall_score(
    Y_true, 
    Y_pred_bool, 
    target_event='ACU',
    event_df=None, 
    lookback_window=30,
    include_ci=False,
):
    """Compute the outcome-level recall/sensitivity. The proportion of events 
    where at least one warning was issued prior to the event 
    
    Args:
        event_df (pd.DataFrame): table of relevant event dates (e.g. D_date,
            next_ED_date, visit_date, etc) associated with each session. Also 
            includes ikn (patient id)
        lookback_window (int): number of days prior to the event in which a 
            warning is valid
    """
    if event_df is None: 
        raise ValueError('Please provide the event_df')
    
    if target_event.endswith('Mortality'):
        # e.g. target_event = '14d Mortality'
        lookback_window = int(target_event.split('d ')[0])
        event = 'D'
    else:
        # event will be either ACU, ED, or H
        # e.g. target_event = 'INFX_H'
        event = target_event.split('_')[-1]
    
    event_col_map = {
        'ACU': ['next_H_date', 'next_ED_date'], 
        'H': ['next_H_date'], 
        'ED': ['next_ED_date'], 
        'D': ['death_date']
    }
    if event not in event_col_map:
        raise NotImplementedError(f'Does not support {target_event}')
    event_cols = event_col_map[event]
    
    event_df = event_df.loc[Y_true.index].copy()
    event_df['true'] = Y_true
    event_df['pred'] = Y_pred_bool
    event_df['event_date'] = event_df[event_cols].min(axis=1)
    
    # exclude sessions without outcome dates
    mask = False
    for col in event_cols: mask |= event_df[col].notnull()
    
    # group predictions by outcome
    worker = partial(group_pred_by_outcome, lookback_window=lookback_window)
    grouped_preds = split_and_parallelize(event_df[mask], worker, processes=8)
    idxs, pred = zip(*grouped_preds)
    result = event_df.loc[list(idxs)].copy()
    result['pred'] = pred
    
    score = recall_score(result['true'], result['pred'], zero_division=1)
    if include_ci:
        mask = result['pred'][result['true']]
        lower, upper = get_confidence_interval(mask, method='binomial')
        return score, lower, upper
    else:
        return score

###############################################################################
# Main Evaluation
###############################################################################
class Evaluate:
    """Evaluate any/all of ML (Machine Learning), RNN (Recurrent Neural Network),
    and ENS (Ensemble) models
    
    Attributes:
        splits (list): a sequence of data splits (str) to evaluate our models on
    """
    def __init__(self, output_path, preds, labels, orig_data):
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
        self.ci = AUCConfidenceInterval(output_path)

    def get_evaluation_scores(
        self, 
        algs=None, 
        target_events=None, 
        splits=None, 
        baseline_cols=None, 
        display_ci=False, 
        load_ci=False, 
        save_ci=False, 
        save_score=True
    ):
        """Compute the AUROC and AUPRC for each given algorithm and data split
        
        Args:
            baseline_cols (list): a sequence of variable names (str) to compute 
                baseline scores (each variable is used as a single baseline 
                model). If None, no baseline model scores will be measured
            display_ci (bool): display confidence interval for AUROC and AUPRC
            load_ci (bool): If True load saved bootstrapped AUROC and AUPRC 
                scores for computing confidence interval
        """    
        if algs is None: algs = self.models
        if target_events is None: target_events = self.target_events
        if splits is None: splits = self.splits
        if load_ci: self.ci.load_bootstrapped_scores()
            
        score_df = pd.DataFrame(index=twolevel, columns=twolevel)
        if baseline_cols is not None: 
            score_df = self.get_baseline_scores(
                score_df, baseline_cols, splits=splits, 
                target_events=target_events, display_ci=display_ci
            )
        
        iterables = itertools.product(algs, splits, target_events)
        for alg, split, target_event in iterables:
            Y_true = self.labels[split][target_event]
            Y_pred_prob = self.preds[split][alg][target_event]
            auroc = roc_auc_score(Y_true, Y_pred_prob)
            auprc = average_precision_score(Y_true, Y_pred_prob)
            if display_ci: 
                ci = self.ci.get_auc_confidence_interval(
                    Y_true, Y_pred_prob, name=f'{alg}_{split}_{target_event}',
                    store=True, verbose=True
                )
                lower, upper = ci['AUROC']
                auroc = f'{auroc:.3f} ({lower:.3f}-{upper:.3f})'
                lower, upper = ci['AUPRC']
                auprc = f'{auprc:.3f} ({lower:.3f}-{upper:.3f})'
                
            score_df.loc[(alg, 'AUROC Score'), (split, target_event)] = auroc
            score_df.loc[(alg, 'AUPRC Score'), (split, target_event)] = auprc
            
        if save_score: 
            score_df.to_csv(f'{self.output_path}/tables/evaluation_scores.csv')
        if save_ci: self.ci.save_bootstrapped_scores()
            
        return score_df
    
    def get_baseline_scores(
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
        var_mapping = {
            'baseline_eGFR': 'Gloemrular Filteration Rate', 
            **{f'baseline_{bt}_count': 'Blood Count' for bt in blood_types},
            **clean_variable_mapping
        }
        
        for base_col in base_cols:
            mean_targets = defaultdict(dict)
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
            name = var_mapping.get(base_col, base_col)
            if numerical_col: name += ' Bin'
            alg = f'Baseline - Event Rate Per {name}'.replace('_', ' ').title()
            
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
                    auroc = roc_auc_score(Y_true, Y_pred_prob)
                    auprc = average_precision_score(Y_true, Y_pred_prob)
                    if display_ci: 
                        ci = self.ci.get_auc_confidence_interval(
                            Y_true, Y_pred_prob, 
                            name=f'{alg}_{split}_{target_event}', store=True, 
                            verbose=True
                        )
                        lower, upper = ci['AUROC']
                        auroc = f'{auroc:.3f} ({lower:.3f}-{upper:.3f})'
                        lower, upper = ci['AUPRC']
                        auprc = f'{auprc:.3f} ({lower:.3f}-{upper:.3f})'
                    score_df.loc[(alg, 'AUROC Score'), col] = auroc
                    score_df.loc[(alg, 'AUPRC Score'), col] = auprc
        return score_df
    
    def get_perf_by_subgroup(
        self,
        subgroups=None,  
        pred_thresh=0.2, 
        alg='ENS',
        target_events=None,
        split='Test',
        save=True,
        **kwargs
    ):
        """
        Args:
            subgroups (list): a sequence of population subgroups (str) to 
                evaluate system performance on. If None, will evaluate on 
                select population subgroups
            pred_thresh (float or list): prediction threshold (float) for alarm
                trigger (when model outputs a risk probability greater than 
                prediction threshold, alarm is triggered) or sequence of 
                prediction thresholds corresponding to each target event
            **kwargs: keyword arguments fed into SubgroupPerformance
        """        
        if subgroups is None:
            subgroups = [
                'all', 'age', 'sex', 'immigrant', 'language', 'income', 
                'area_density', 'regimen', 'cancer_location', 'days_since_starting'
            ]
        if target_events is None: target_events = self.target_events
        
        pred_thresh_is_list = isinstance(pred_thresh, list)
        if pred_thresh_is_list: 
            assert(len(pred_thresh) == len(target_events))
        
        thresh = pred_thresh
        Y, pred_prob = self.labels[split], self.preds[split][alg]
        data = self.orig_data.loc[Y.index]
        
        sp = SubgroupPerformance(
            data, self.output_path, cohort_name=split, subgroups=subgroups, **kwargs
        )
        results = {}
        for idx, target_event in enumerate(tqdm(target_events)):
            Y_true = Y[target_event]
            Y_pred_prob = pred_prob[target_event]
            if pred_thresh_is_list: thresh = pred_thresh[idx]
                
            results[target_event] = sp.get_summary(
                Y_true, Y_pred_prob, thresh, target_event 
            )
        
        summary_df = pd.concat(results).T
        
        if save:
            filename = 'subgroup_performance'
            summary_df.to_csv(f'{self.output_path}/tables/{filename}.csv')
            
            if sp.display_ci: 
                filename = 'bootstrapped_subgroup_scores'
                sp.ci.save_bootstrapped_scores(filename=filename)

        return summary_df
    
    def operating_points(
        self, 
        points, 
        op_metric='threshold', 
        alg='ENS',
        target_events=None, 
        split='Test',
        mask=None,
        save=True,
        **kwargs
    ):
        """Evaluate how system performs at different operating points 
        (e.g. prediction thresholds, desired precisions, desired recalls)
        
        Args:
            mask (pd.Series): An alignable boolean series to filter samples in 
                the corresponding data split. If None, no filtering is done.
            **kwargs: keyword arguments fed into get_perf_at_operating_point
        """
        if target_events is None: target_events = self.target_events
            
        Y, pred_prob = self.labels[split], self.preds[split][alg]
        if mask is not None: Y, pred_prob = Y[mask], pred_prob[mask]
        
        result = {}
        for target_event in tqdm(target_events):
            Y_true = Y[target_event]
            Y_pred_prob = pred_prob[target_event]
                
            perf = {}
            for point in points:
                point = round(point, 2)
                perf[point] = get_perf_at_operating_point(
                    point, Y_true, Y_pred_prob, op_metric=op_metric, 
                    target_event=target_event, **kwargs
                )
                
            result[target_event] = pd.DataFrame(perf)
        
        df = pd.concat(result).T
        df.index.name = op_metric.title()
        df = df.round(3)
        if save: 
            df.to_csv(f'{self.output_path}/tables/{op_metric}_performance.csv')
            
        return df
    
    def all_plots_for_single_target(
        self, 
        alg='XGB', 
        target_event='ACU', 
        split='Test', 
        calib_ci=True, 
        n_bins=10, 
        calib_strategy='quantile', 
        figsize=(12,18), 
        save=True
    ):
        # setup 
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        Y_true = self.labels[split][target_event]
        Y_pred_prob = self.preds[split][alg][target_event]
        
        # plot
        self.plot_auc_curve(
            axes[0], Y_true, Y_pred_prob, curve_type='pr', legend_loc='lower right', 
            remove_legend_line=True, ci_name=f'{alg}_{split}_{target_event}',
        )
        self.plot_auc_curve(
            axes[1], Y_true, Y_pred_prob, curve_type='roc', legend_loc='lower right', 
            remove_legend_line=True, ci_name=f'{alg}_{split}_{target_event}',
        )
        self.plot_calib(
            axes[2], Y_true, Y_pred_prob, n_bins=n_bins, legend_loc='lower right',
            calib_strategy=calib_strategy, calib_ci=calib_ci,
        )
        self.plot_pred_cdf(axes[3], Y_pred_prob)
        self.plot_decision_curve(axes[4], Y_true, Y_pred_prob)
        
        # save
        if save:
            filenames = ['pr', 'roc', 'calib', 'cdf', 'dc']
            for idx, filename in enumerate(filenames):
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

    def plot_auc_curve(
        self, 
        ax, 
        Y_true,
        Y_pred_prob,
        curve_type='roc', 
        color=None,
        legend_loc='best', 
        remove_legend_line=False,
        title=None, 
        label_prefix=None,
        ylim=(-0.05, 1.05),
        ci_name=None,
    ):
        """
        Args:
            ci_name (str): confidence interval lookup name if stored. If 
                not found, will compute boostrapped AUC scores to compute 
                confidence interval
        """
        # setup
        if curve_type == 'pr':
            curve_func, score_func = precision_recall_curve, average_precision_score
            label, xlabel, ylabel = 'AUPRC', 'Sensitivity', 'Positive Predictive Value'
        elif curve_type == 'roc':
            curve_func, score_func = roc_curve, roc_auc_score
            label, xlabel, ylabel = 'AUROC', '1 - Specificity', 'Sensitivity'
            
        # get the curve numbers
        x, y, thresh = curve_func(Y_true, Y_pred_prob)
        if curve_type == 'pr': x, y = y, x
        
        # get the score and 95% CI
        ci = self.ci.get_auc_confidence_interval(
            Y_true, Y_pred_prob, name=ci_name
        )
        lower, upper = ci[label]
        score = score_func(Y_true, Y_pred_prob)
        
        # plot it
        label = f'{label}={score:.3f} (95% CI: {lower:.3f}-{upper:.3f})'
        if label_prefix is not None: label = f'{label_prefix}{label}'
        ax.plot(x, y, label=label, color=color)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, ylim=ylim)
        leg = ax.legend(loc=legend_loc, frameon=False)
        if remove_legend_line: leg.legendHandles[0].set_linewidth(0)
            
    def plot_pred_cdf(
        self, 
        ax, 
        Y_pred_prob,
        legend_loc='best', 
        label=None,
        title=None
    ): 
        N = len(Y_pred_prob)
        y = np.arange(N) / float(N)
        x = np.sort(Y_pred_prob)
        ax.plot(x, y, label=label)
        ax.set(
            title=title, xlabel='Predicted Probability', 
            ylabel='Cumulative Proportion of Predictions'
        )
        if label is not None: ax.legend(loc=legend_loc, frameon=False)
            
    def plot_calib(
        self, 
        ax, 
        Y_true,
        Y_pred_prob,
        color=None,
        legend_loc='best', 
        title=None, 
        label_prefix=None,
        n_bins=10, 
        calib_strategy='quantile', 
        calib_ci=False, 
        show_perf_calib=True
    ):
        if calib_ci:
            logging.warning('Displaying calibration confidence interval with '
                            'more than one target will make the plot messy. Do '
                            'not set calib_ci to True when there are multiple '
                            'targets')
            
        # bin predicted probability (e.g. 0.0-0.1, 0.1-0.2, etc) 
        # prob_true: fraction of positive class in each bin
        # prob_pred: mean of each bin
        prob_true, prob_pred = calibration_curve(
            Y_true, Y_pred_prob, n_bins=n_bins, strategy=calib_strategy
        )
        axis_max_limit = max(prob_true.max(), prob_pred.max())
        max_calib_error = np.max(abs(prob_true - prob_pred))
        mean_calib_error = np.mean(abs(prob_true - prob_pred))
    
        if calib_ci:
            lower, upper = get_calibration_confidence_interval(
                Y_true, Y_pred_prob, n_bins, calib_strategy
            )
            yerr = abs(np.array([lower - prob_true, upper - prob_true]))
            ax.errorbar(
                prob_pred, prob_true, yerr=yerr, capsize=5.0, 
                errorevery=n_bins//10, ecolor='firebrick'
            )
            adjustment_factor = 1 if axis_max_limit > 0.25 else 3
            ax.text(
                axis_max_limit/2, 0.07/adjustment_factor, 
                f'Mean Calibration Error {mean_calib_error:.3f}'
            )
            ax.text(
                axis_max_limit/2, 0.1/adjustment_factor, 
                f'Max Calibration Error {max_calib_error:.3f}'
            )
            ax.set_ylim(-0.01, axis_max_limit+0.01)
        else:
            label = (f'Max Error={max_calib_error:.3f} '
                     f'Mean Error={mean_calib_error:.3f}')
            if label_prefix is not None: label = f'{label_prefix}{label}'
            ax.plot(prob_pred, prob_true, label=label, color=color)
        
        if show_perf_calib:
            ax.plot(
                [0,axis_max_limit], [0,axis_max_limit], 'k:', 
                label='Perfect Calibration'
            )
            
        ax.set(
            title=title, xlabel='Predicted Probability', 
            ylabel='Empirical Probability'
        )
        ax.legend(loc=legend_loc, frameon=False)
        
        return prob_true, prob_pred
        
    def plot_decision_curve(
        self, 
        ax, 
        Y_true,
        Y_pred_prob,
        xlim=None, 
        colors=None
    ):
        if colors is None: 
            colors = {'System': '#1f77b4', 'All': '#bcbd22', 'None': '#2ca02c'}
            
        fpr, tpr, thresh = roc_curve(Y_true, Y_pred_prob)
        
        # the odds approaches infinity at these thresholds, let's remove them
        mask = thresh > 0.999
        fpr, tpr, thresh = fpr[~mask], tpr[~mask], thresh[~mask]
        
        # compute net benefit for model and treat all
        sensitivity, specificity, prevalence = tpr, 1 - fpr, Y_true.mean()
        odds = thresh / (1 - thresh)
        net_benefit = sensitivity*prevalence - (1 - specificity)*(1 - prevalence)*odds
        treat_all = prevalence - (1 - prevalence) * odds
        thresh, net_benefit, treat_all = thresh[1:], net_benefit[1:], treat_all[1:]
        
        df = pd.DataFrame(
            data=np.array([thresh, net_benefit, treat_all]).T, 
            columns=['Threshold', 'System', 'All']
        )
        
        # plot decision curve analysis
        y_max = 0
        for label, y in df[['System', 'All']].items():
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
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.2f}'))
        ax.legend(frameon=False)
        
        return df

    def plot_curves(
        self, 
        curve_type='roc', 
        algs=None,
        target_events=None, 
        split='Test',
        legend_loc='best', 
        figsize=(12,18), 
        padding=None, 
        save=True
    ):
        if curve_type not in ['roc', 'pr', 'pred_cdf']: 
            raise ValueError("curve_type must be set to roc, pr, or pred_cdf")
        if algs is None: algs = self.models
        if target_events is None: target_events = self.target_events
        if padding is None: padding = {'pad_y1': 0.3}
        if self.ci.bs_scores.empty: self.ci.load_bootstrapped_scores()
            
        nrows, ncols = math.ceil(len(algs)/2), 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()
        
        for i, alg in enumerate(algs):
            for target_event in target_events:
                Y_true = self.labels[split][target_event]
                Y_pred_prob = self.preds[split][alg][target_event]
                if curve_type in ['roc', 'pr']:
                    self.plot_auc_curve(
                        axes[i], Y_true, Y_pred_prob, curve_type=curve_type, 
                        legend_loc=legend_loc, title=alg, 
                        label_prefix=f'{target_event}\n',
                        ci_name=f'{alg}_{split}_{target_event}'
                    )
                elif curve_type == 'pred_cdf':
                    self.plot_pred_cdf(
                        axes[i], Y_pred_prob, legend_loc=legend_loc, 
                        label=target_event, title=alg
                    )
            if save: 
                fig.savefig(
                    f'{self.output_path}/figures/curves/{alg}_{curve_type}.jpg',
                    bbox_inches=get_bbox(axes[i], fig, **padding), dpi=300
                ) 
                
        if save: 
            filepath = f'{self.output_path}/figures/curves/{curve_type}.jpg'
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            
        plt.show()
        
    def plot_calibs(
        self, 
        algs=None,
        target_events=None, 
        split='Test', 
        include_pred_hist=False, 
        n_bins=10,        
        calib_strategy='quantile', 
        legend_loc='best', 
        figsize=(12,18), 
        padding=None, 
        save=True
    ):
        if algs is None: algs = self.models
        if target_events is None: target_events = self.target_events
        if padding is None: padding = {'pad_y1': 0.3}
        
        nrows, ncols = math.ceil(len(algs)/2), 2
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3*nrows, ncols, hspace=0.5)

        for idx, alg in enumerate(algs):
            if include_pred_hist:
                row = int(idx / 2) * 3
                col = idx % 2
                ax = fig.add_subplot(gs[row:row+2, col])
            else:
                ax = fig.add_subplot(nrows, ncols, idx+1)
                
            for i, target_event in enumerate(target_events):
                show_perf_calib = i == len(target_events) - 1 # at last one
                Y_true = self.labels[split][target_event]
                Y_pred_prob = self.preds[split][alg][target_event]
                prob_true, prob_pred = self.plot_calib(
                    ax, Y_true, Y_pred_prob, title=alg, n_bins=n_bins, 
                    calib_strategy=calib_strategy, legend_loc=legend_loc,
                    label_prefix=f'{target_event}\n', 
                    show_perf_calib=show_perf_calib,
                )
                
                if save:
                    # save the calibration numbers
                    np.save(f'{self.output_path}/figures/curves/'
                            f'{target_event}_calib_true_array.npy', prob_true)
                    np.save(f'{self.output_path}/figures/curves/'
                            f'{target_event}_calib_pred_array.npy', prob_pred)

            if include_pred_hist:
                axis_max_limit = max(ax.axis())
                ax_hist = fig.add_subplot(gs[row+2, col], sharex=ax)
                for target_event in target_events:
                    Y_pred_prob = self.preds[split][alg][target_event]
                    ax_hist.hist(
                        Y_pred_prob, range=(0,axis_max_limit), bins=n_bins,
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
        
    def plot_decision_curves(
        self, 
        alg, 
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
            Y_true = self.labels[split][target_event]
            Y_pred_prob = self.preds[split][alg][target_event]
            label = self.labels[split][target_event]
            ax = fig.add_subplot(N,1,idx+1)
            result[target_event] = self.plot_decision_curve(
                ax, Y_true, Y_pred_prob, xlim=xlim
            )
            if save:
                filename = f'{alg}_{target_event}'
                fig.savefig(
                    f'{self.output_path}/figures/decision_curve/{filename}.jpg',
                    bbox_inches=get_bbox(ax, fig, **padding), dpi=300
                ) 
            ax.set_title(target_event) # set title AFTER saving individual figures
            
        if save:
            plt.savefig(
                f'{self.output_path}/figures/decision_curve/{alg}.jpg', 
                bbox_inches='tight', dpi=300
            )
        plt.show()
        
        return result
    
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
    
    def get_confusion_matrix(self, Y_true, Y_pred_bool):
        cm = confusion_matrix(Y_true, Y_pred_bool)
        cm = pd.DataFrame(
            cm, 
            columns=['Predicted False', 'Predicted True'], 
            index=['Actual False', 'Actual True']
        )
        return cm

###############################################################################
# Baseline Evaluation
###############################################################################
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
        Y_true = self.labels[split][target_event]
        Y_pred_prob = self.preds[split][alg][target_event]

        # plot
        self.plot_loess(axes[0], alg, target_event, split=split)
        self.plot_auc_curve(
            axes[1], Y_true, Y_pred_prob, curve_type='pr', legend_loc='lower right',
            ci_name=f'{alg}_{split}_{target_event}'
        )
        self.plot_auc_curve(
            axes[2], Y_true, Y_pred_prob, curve_type='roc', legend_loc='lower right',
            ci_name=f'{alg}_{split}_{target_event}'
        )
        # hotfix - bound the predictions to (0, 1) for calibration
        self.plot_calib(
            axes[3], Y_true, Y_pred_prob.clip(lower=0, upper=1), n_bins=n_bins,
            legend_loc='lower right', calib_strategy=calib_strategy
        )
        
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
        for target_event, Y_true in Y.items():
            if Y_true.nunique() < 2: 
                # no pos examples, no point in plotting/evaluating
                continue
            self.all_plots_for_single_target(target_event=target_event, **kwargs)
            
###############################################################################
# Subgroup Evaluation
###############################################################################
class SubgroupPerformance(SubgroupSummary):
    def __init__(
        self, 
        data, 
        output_path,
        subgroups=None,
        display_ci=False, 
        load_ci=False,
        top=3,
        cohort_name='Test',
        perf_kwargs=None,
        **kwargs
    ):
        """
        Args:
            top (int): the number of most common categories. We only analyze 
                populations subgroups belonging to those top categories
            perf_kwargs: keyword arguments fed into get_perf_at_operating_point.
                If None, will use default params
            **kwargs: keyword arguments fed into SubgroupSummary
        """
        super().__init__(data, **kwargs)
        if subgroups is None:
            subgroups = [
                'all', 'age', 'sex', 'immigration', 'language', 'arrival', 
                'income', 'area_density', 'regimen', 'cancer_location', 
                'days_since_starting'
            ]
        self.subgroups = subgroups
        self.display_ci = display_ci
        if self.display_ci:
            self.ci = AUCConfidenceInterval(output_path)
            if load_ci:
                filename='bootstrapped_subgroup_scores'
                self.ci.load_bootstrapped_scores(filename=filename)
        self.top = top
        self.cohort_name = cohort_name
        if perf_kwargs is None:
            perf_kwargs = {'perf_metrics': ['precision', 'recall', 'event_rate']}
        self.perf_kwargs = perf_kwargs
        
    def get_summary(self, *args):
        summary = {}
        if 'all' in self.subgroups: 
            self.entire_cohort_summary(summary, *args)
        if 'age' in self.subgroups: 
            self.age_summary(summary, *args)
        if 'sex' in self.subgroups: 
            self.sex_summary(summary, *args)
        if 'immigrant' in self.subgroups: 
            self.immigration_summary(summary, *args)
        if 'language' in self.subgroups: 
            self.language_summary(summary, *args)
        if 'arrival' in self.subgroups: 
            self.arrival_summary(summary, *args)
        if 'world_region_of_birth' in self.subgroups: 
            self.world_region_of_birth_summary(summary, *args)
        if 'income' in self.subgroups: 
            self.income_summary(summary, *args)
        if 'area_density' in self.subgroups: 
            self.area_density_summary(summary, *args)
        if 'regimen' in self.subgroups: 
            self.most_common_category_summary(
                summary, *args, catcol='regimen', 
                transform=str.upper, top=self.top
            )
        if 'cancer_location' in self.subgroups: 
            self.most_common_category_summary(
                summary, *args, catcol='cancer_topog_cd', 
                mapping=cancer_code_mapping, top=self.top
            )
        if 'days_since_starting' in self.subgroups: 
            self.days_since_starting_regimen_summary(summary, *args)
        if 'cycle_length' in self.subgroups: 
            self.cycle_length_summary(summary, *args)
        if 'ckd' in self.subgroups: 
            self.ckd_summary(summary, *args)
            
        return pd.DataFrame(summary)

    def entire_cohort_summary(self, *args):
        mask = self.data['ikn'].notnull()
        self._summary(
            *args, mask=mask, subgroup=f'Entire {self.cohort_name} Cohort', 
            category=f'{self.N} patients'
        )

    def _summary(
        self, 
        result,
        Y_true, 
        Y_pred_prob, 
        pred_thresh,
        target_event,
        mask=None, 
        subgroup='', 
        category='',
        col=None,
    ):
        if mask is None: 
            raise ValueError('Please provide a mask')

        Y_true, Y_pred_prob = Y_true[mask], Y_pred_prob[mask]
        if len(set(Y_true)) < 2:
            logging.warning(f'No positive examples, skipping '
                            f'{target_event} {subgroup} {category}')
            return
        
        if col is None:
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            col = (subgroup, f'{category} ({num_patients/self.N*100:.1f}%)')
        
        # compute the scores
        perf = get_perf_at_operating_point(
            pred_thresh, Y_true, Y_pred_prob, target_event=target_event, 
            include_ci=self.display_ci, **self.perf_kwargs
        )
        auroc = np.round(roc_auc_score(Y_true, Y_pred_prob), 3)
        auprc = np.round(average_precision_score(Y_true, Y_pred_prob), 3)
        if self.display_ci:
            name = f'{subgroup}_{category}_{target_event}'
            ci = self.ci.get_auc_confidence_interval(
                Y_true, Y_pred_prob, name=name.replace(' ', '').lower()
            )
            lower, upper = ci['AUROC']
            auroc = f'{auroc} ({lower:.3f}-{upper:.3f})'
            lower, upper = ci['AUPRC']
            auprc = f'{auprc} ({lower:.3f}-{upper:.3f})'
        
        # store the scores
        perf = {k: round(v, 3) if isinstance(v, float) else v 
                for k, v in perf.items()}
        result[col] = {'AUROC': auroc, 'AUPRC': auprc, **perf}