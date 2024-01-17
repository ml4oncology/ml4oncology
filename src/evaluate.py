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
from functools import partial
import itertools
import math
import os

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix, 
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    roc_curve, 
    r2_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import logger
from src.conf_int import (
    ScoreConfidenceInterval,
    get_confidence_interval,
    get_calibration_confidence_interval,
)
from src.config import blood_types, cancer_code_mapping
from src.summarize import SubgroupSummary
from src.utility import (
    split_and_parallelize,
    group_pred_by_outcome,
    pred_thresh_binary_search,
    twolevel
)
from src.visualize import tile_plot, get_bbox

CLF_SCORE_FUNCS = {
    'AUROC': roc_auc_score,
    'AUPRC': average_precision_score
}
REG_SCORE_FUNCS = {
    'MAE': mean_absolute_error, 
    'MSE': mean_squared_error, 
    'RMSE': partial(mean_squared_error, squared=False), 
    'R2': r2_score
}
NAME_MAP = {
    'baseline_eGFR': 'Pre-Treatment eGFR',
    'next_eGFR': 'Post-Treatment eGFR',
    'eGFR_change': 'eGFR Change'
}
for bt in blood_types:
    NAME_MAP[f'baseline_{bt}_value'] = f'Pre-Treatment {bt.title()} Value'
    NAME_MAP[f'target_{bt}_value'] = f'Before Next Treatment {bt.title()} Value'


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
    def __init__(self, output_path, preds, labels, score_funcs, splits=None):
        """
        Args:
            preds (dict): mapping of algorithms (str) and their predictions for 
                each partition (dict of str: pd.DataFrame)
            labels (dict): mapping of data splits (str) and their associated 
                labels (pd.Series)
            score_funcs (dict): mapping of score name (str) and their associated
                score functions
        """
        self.output_path = output_path
        self.preds = preds
        self.labels = labels
        self.score_funcs = score_funcs
        if splits is None: splits = ['Valid', 'Test']
        self.splits = splits
        self.algs = list(preds.keys())
        self.target_events = list(labels['Test'].columns)
        self.ci = ScoreConfidenceInterval(output_path, score_funcs)

    def get_evaluation_scores(
        self, 
        algs=None, 
        target_events=None, 
        splits=None,  
        display_ci=False, 
        load_ci=False, 
        save_ci=False, 
        save_score=True
    ):
        """Compute the AUROC and AUPRC for each given algorithm and data split
        
        Args:
            display_ci (bool): display confidence interval for AUROC and AUPRC
            load_ci (bool): If True load saved bootstrapped AUROC and AUPRC 
                scores for computing confidence interval
        """    
        if algs is None: algs = self.algs
        if target_events is None: target_events = self.target_events
        if splits is None: splits = self.splits
        if load_ci: self.ci.load_bootstrapped_scores()
            
        score_df = pd.DataFrame(index=twolevel, columns=twolevel)        
        iterables = itertools.product(algs, splits, target_events)
        for alg, split, target_event in iterables:
            Y_true = self.labels[split][target_event]
            Y_pred = self.preds[alg][split][target_event]
            metrics = {name: func(Y_true, Y_pred) 
                       for name, func in self.score_funcs.items()}
            if display_ci: 
                ci = self.ci.get_score_confidence_interval(
                    Y_true, Y_pred, name=f'{alg}_{split}_{target_event}',
                    store=True, verbose=True
                )
                for name, (lower, upper) in ci.items():
                    score = f'{metrics[name]:.3f} ({lower:.3f}-{upper:.3f})'
                    metrics[name] = score
            
            for name, score in metrics.items():
                score_df.loc[(alg, name), (split, target_event)] = score
            
        if save_score: 
            score_df.to_csv(f'{self.output_path}/tables/evaluation_scores.csv')
        if save_ci: self.ci.save_bootstrapped_scores()
            
        return score_df

class EvaluateClf(Evaluate):
    """Evaluate classifiers"""
    def __init__(self, output_path, preds, labels, **kwargs):
        super().__init__(output_path, preds, labels, CLF_SCORE_FUNCS, **kwargs)
        
    def get_perf_by_subgroup(
        self,
        data,
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
            data (pd.DataFrame): the original dataset before one-hot encoding 
                and splitting
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
        Y, pred_prob = self.labels[split], self.preds[alg][split]
        data = data.loc[Y.index]
        
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
            
        Y, pred_prob = self.labels[split], self.preds[alg][split]
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
        save=True,
        img_format='svg'
    ):
        # setup 
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        Y_true = self.labels[split][target_event]
        Y_pred_prob = self.preds[alg][split][target_event]
        
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
                    f'{self.output_path}/figures/curves/{filename}.{img_format}', 
                    bbox_inches=get_bbox(axes[idx], fig), dpi=300, format=img_format
                ) 
            plt.savefig(
                f'{self.output_path}/figures/curves/{alg}_{target_event}.{img_format}',
                bbox_inches='tight', dpi=300, format=img_format
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
        ci = self.ci.get_score_confidence_interval(
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
            msg = ('Displaying calibration confidence interval with more than '
                   'one target will make the plot messy. Do not set calib_ci '
                   'to True when there are multiple targets')
            logger.warning(msg)
            
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
            self.plot_perf_calib(ax, axis_max_limit)
            
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
        figsize=None, 
        padding=None, 
        save=True
    ):
        if curve_type not in ['roc', 'pr', 'pred_cdf']: 
            raise ValueError("curve_type must be set to roc, pr, or pred_cdf")
        if algs is None: algs = self.algs
        if target_events is None: target_events = self.target_events
        if padding is None: padding = {'pad_y1': 0.3}
        if self.ci.bs_scores.empty: self.ci.load_bootstrapped_scores()
            
        nrows, ncols = math.ceil(len(algs)/2), 2
        if figsize is None: figsize = (ncols*6, nrows*6)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()
        
        for i, alg in enumerate(algs):
            for target_event in target_events:
                Y_true = self.labels[split][target_event]
                Y_pred_prob = self.preds[alg][split][target_event]
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
        figsize=None, 
        padding=None, 
        save=True
    ):
        if algs is None: algs = self.algs
        if target_events is None: target_events = self.target_events
        if padding is None: padding = {'pad_y1': 0.3}
        
        nrows, ncols = math.ceil(len(algs)/2), 2
        if figsize is None: figsize = (ncols*6, nrows*6)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3*nrows, ncols, hspace=0.5)

        for idx, alg in enumerate(algs):
            if include_pred_hist:
                row = int(idx / 2) * 3
                col = idx % 2
                ax = fig.add_subplot(gs[row:row+2, col])
            else:
                ax = fig.add_subplot(nrows, ncols, idx+1)
            
            axis_max_limit = 0
            for i, target_event in enumerate(target_events):
                Y_true = self.labels[split][target_event]
                Y_pred_prob = self.preds[alg][split][target_event]
                prob_true, prob_pred = self.plot_calib(
                    ax, Y_true, Y_pred_prob, title=alg, n_bins=n_bins, 
                    calib_strategy=calib_strategy, legend_loc=legend_loc,
                    label_prefix=f'{target_event}\n', show_perf_calib=False,
                )
                
                if save:
                    # save the calibration numbers
                    np.save(f'{self.output_path}/figures/curves/'
                            f'{target_event}_calib_true_array.npy', prob_true)
                    np.save(f'{self.output_path}/figures/curves/'
                            f'{target_event}_calib_pred_array.npy', prob_pred)

                cur_axis_max_limit = max(prob_true.max(), prob_pred.max())
                axis_max_limit = max(axis_max_limit, cur_axis_max_limit)
            self.plot_perf_calib(ax, axis_max_limit)
            ax.legend(loc=legend_loc, frameon=False)

            if include_pred_hist:
                axis_max_limit = max(ax.axis())
                ax_hist = fig.add_subplot(gs[row+2, col], sharex=ax)
                for target_event in target_events:
                    Y_pred_prob = self.preds[alg][split][target_event]
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
        figsize=None, 
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
        if figsize is None: figsize = (6, 6*N)
        if padding is None: padding = {}
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        result = {}
        
        for idx, target_event in enumerate(target_events):
            Y_true = self.labels[split][target_event]
            Y_pred_prob = self.preds[alg][split][target_event]
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
    
    def get_confusion_matrix(self, Y_true, Y_pred_bool):
        cm = confusion_matrix(Y_true, Y_pred_bool)
        cm = pd.DataFrame(
            cm, 
            columns=['Predicted False', 'Predicted True'], 
            index=['Actual False', 'Actual True']
        )
        return cm
    
    def plot_perf_calib(self, ax, axis_max_limit):
        ax.plot(
            [0, axis_max_limit], [0, axis_max_limit], 'k:',
            label='Perfect Calibration'
        )
    
class EvaluateReg(Evaluate):
    """Evaluate regressors"""
    def __init__(self, output_path, preds, labels, **kwargs):
        super().__init__(output_path, preds, labels, REG_SCORE_FUNCS, **kwargs)
        
    def plot_label_vs_pred(
        self,
        alg,
        target_event,
        split='Test',
        save=True,
        **kwargs 
    ):
        """
        Args:
            **kwargs: keyword arguments fed into tile_plot
        """
        Y_pred = self.preds[alg][split][target_event]
        Y_true = self.labels[split][target_event]
        name = NAME_MAP.get(target_event, target_event)
        tile_plot(
            Y_pred,
            Y_true,
            xlabel=f'Predicted {name}',
            ylabel=f'True {name}',
            **kwargs
        )
        if save:
            filename = f'{alg}_{target_event}_label_vs_pred'
            plt.savefig(
                f'{self.output_path}/figures/curves/{filename}.jpg',
                bbox_inches='tight', dpi=300
            )
        
    def plot_err_dist(self, alg, target_event, split='Test'):
        Y_pred = self.preds[alg][split][target_event]
        Y_true = self.labels[split][target_event]
        err = Y_true - Y_pred
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
        name = NAME_MAP.get(target_event, target_event)
        # histogram of error alues
        err.hist(bins=100, ax=axes[0])
        axes[0].set(
            xlabel=f'{name} Error Values', 
            ylabel='Number of Sessions'
        )
        
        # histogram of absolute error values
        err.abs().hist(bins=100, ax=axes[1])
        axes[1].set(
            xlabel=f'{name} Absolute Error Values', 
            ylabel='Number of Sessions'
        )
        
        # cumulative distribution function of absolute error values
        N = len(Y_true)
        y = np.arange(N) / float(N)
        x = err.abs().sort_values()
        axes[2].plot(x, y)
        axes[2].grid()
        axes[2].set_xticks(np.arange(0, int(x.max()+5), 5))
        axes[2].set(
            xlabel=f'{name} Absolute Error Values', 
            ylabel='Cumulative Proportion of Error Values'
        )
        plt.show()

###############################################################################
# Baseline Evaluation
###############################################################################
class EvaluateBaselineModel(Evaluate):
    def __init__(self, X, preds, labels, output_path, pred_ci=None):
        """
        Args:
            X (pd.Series): the original baseline predictor values
            pred_ci (dict): mapping of data splits (str) and their associated 
                upper and lower confidence interval predictions by each 
                algorithm (dict of str: tuple(pd.DataFrame, pd.DataFrame))
        """
        super().__init__(output_path, preds, labels, score_funcs=None)
        self.X = X
        self.base_col = X.name
        self.pred_ci = pred_ci

        save_path = f'{output_path}/figures/baseline'
        if not os.path.exists(save_path): os.makedirs(save_path)
        
    def plot_pred_vs_base(
        self, 
        ax, 
        alg, 
        target_event, 
        split, 
        show_diagonal=False, 
        clip_flat_edges=True,
    ):
        """
        Args:
            clip_flat_edges (bool): If True, remove any flat sections at 
                the beginning and end of the curve.
        """
        x = self.X.sort_values()
        pred = self.preds[alg][split].loc[x.index, target_event]
        if clip_flat_edges:
            initial_flat_mask = (pred == pred.iloc[0]).cumprod()
            end_flat_mask = (pred == pred.iloc[-1])[::-1].cumprod()[::-1]
            mask = (initial_flat_mask | end_flat_mask).astype(bool)
            x, pred = x[~mask], pred[~mask]
        ax.plot(x, pred)
        
        if self.pred_ci is not None:
            pred_min, pred_max = self.pred_ci[alg][split]
            pred_min = pred_min.loc[x.index, target_event]
            pred_max = pred_max.loc[x.index, target_event]
            ax.fill_between(x, pred_min, pred_max, alpha=0.25)
        
        if show_diagonal:
            axis_max = max(x.max(), pred.max())
            axis_min = min(x.min(), pred.min())
            ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k:')
        
        xlabel = NAME_MAP.get(self.base_col, self.base_col)
        ylabel = f'Prediction for {NAME_MAP.get(target_event, target_event)}'
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid()

    def plot_label_vs_base(
        self,
        target_event,
        split='Test',
        save=True,
        **kwargs
    ):
        """
        Args:
            **kwargs: keyword arguments fed into tile_plot
        """
        x = self.X.sort_values()
        y = self.labels[split].loc[x.index, target_event]
        tile_plot(
            x,
            y,
            xlabel=NAME_MAP.get(self.base_col, self.base_col),
            ylabel=f'True {NAME_MAP.get(target_event, target_event)}',
            **kwargs
        )
        if save:
            filename = f'{target_event}_label_vs_base'
            plt.savefig(
                f'{self.output_path}/figures/curves/{filename}.jpg',
                bbox_inches='tight', dpi=300
            )
        
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
            self.ci = ScoreConfidenceInterval(output_path, CLF_SCORE_FUNCS)
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
            msg = (f'No positive examples, skipping {target_event} '
                   f'{subgroup} {category}')
            logger.warning(msg)
            return
        
        if col is None:
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            col = (subgroup, f'{category} ({num_patients/self.N*100:.1f}%)')
        
        # compute the scores
        perf = get_perf_at_operating_point(
            pred_thresh, Y_true, Y_pred_prob, target_event=target_event, 
            include_ci=self.display_ci, **self.perf_kwargs
        )
        for name, func in CLF_SCORE_FUNCS.items():
            perf[name] = func(Y_true, Y_pred_prob)
        if self.display_ci:
            name = f'{subgroup}_{category}_{target_event}'
            ci = self.ci.get_score_confidence_interval(
                Y_true, Y_pred_prob, name=name.replace(' ', '').lower()
            )
            for name, (lower, upper) in ci.items():
                perf[name] = f'{perf[name]:.3f} ({lower:.3f}-{upper:.3f})'
          
        # store the scores
        result[col] = {k: round(v, 3) if isinstance(v, float) else v 
                       for k, v in perf.items()}
