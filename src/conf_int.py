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
"""
Module for computing confidence intervals
"""
from functools import partial

from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score
)
from statsmodels.stats.proportion import proportion_confint
import numpy as np
import pandas as pd

from src import logging
from src.utility import (
    twolevel,
    split_and_parallelize,
    get_cyto_rates,
)

def get_confidence_interval(data, method='basic', alpha=0.05):
    """Retrieve confidence interval (CI) of data
    
    Args:
        data (array-like): a sequence of values (boolean or float)
        method (str): the method to CI, either 'basic' or 'binomial'
        alpha (float): significance level of CI
    """
    if method == 'basic':
        lower_perc = 100 * alpha / 2
        upper_perc = 100 - lower_perc
        lower_ci, upper_ci = np.percentile(data, [lower_perc, upper_perc])
    elif method == 'binomial':
        lower_ci, upper_ci = proportion_confint(
            count=sum(data), nobs=len(data), method='wilson', alpha=alpha
        )
    return lower_ci, upper_ci

class AUCConfidenceInterval:
    """
    Attributes:
        bs_scores (pd.DataFrame): table of bootstrapped AUROC and AUPRC scores
            for computing AUC confidence intervals
    """
    def __init__(self, output_path):
        self.output_path = output_path
        # memory table / cache to store the bootstrapped scores
        self.bs_scores = pd.DataFrame(index=twolevel)
            
    def get_auc_confidence_interval(
        self, 
        Y_true, 
        Y_pred_prob, 
        **kwargs
    ):
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(
            Y_true, Y_pred_prob, **kwargs,
        )
        
        ci = {
            'AUROC': get_confidence_interval(auroc_scores, method='basic'),
            'AUPRC': get_confidence_interval(auprc_scores, method='basic')
        }
        return ci
    
    def get_bootstrapped_scores(
        self, 
        Y_true, 
        Y_pred_prob, 
        store=True,
        name='Bootstraps',
        n_bootstraps=10000,
        verbose=True
    ):
        # Check the cache
        if name not in self.bs_scores.index:
            auc_scores = compute_bootstrap_auc_scores(
                Y_true, Y_pred_prob, n_bootstraps=n_bootstraps
            )
            auroc_scores, auprc_scores = np.array(auc_scores).T
            if verbose: 
                logging.info(f'Completed bootstrap computations for {name}')
            if store:
                cols = range(n_bootstraps)
                self.bs_scores.loc[(name, 'AUROC'), cols] = auroc_scores
                self.bs_scores.loc[(name, 'AUPRC'), cols] = auprc_scores
        else:
            auroc_scores, auprc_scores = self.bs_scores.loc[name].to_numpy()
            
        return auroc_scores, auprc_scores
    
    def load_bootstrapped_scores(self, filename='bootstrapped_scores'):
        filepath = f'{self.output_path}/confidence_interval/{filename}.csv'
        self.bs_scores = pd.read_csv(filepath, index_col=[0,1])
        self.bs_scores.columns = self.bs_scores.columns.astype(int)
        
    def save_bootstrapped_scores(self, filename='bootstrapped_scores'):
        filepath = f'{self.output_path}/confidence_interval/{filename}.csv'
        self.bs_scores.to_csv(filepath)
        
###############################################################################
# Bootstrapping
###############################################################################
def compute_bootstrap_auc_scores(
    Y_true, 
    Y_pred_prob, 
    n_bootstraps=10000, 
    processes=32
):
    """Compute bootstrapped AUROC/AUPRC scores for computing their confidence 
    intervals
    """
    worker = partial(auc_bootstrap_worker, Y_true=Y_true, Y_pred_prob=Y_pred_prob)
    scores = split_and_parallelize(
        range(n_bootstraps), worker, split_by_ikns=False, processes=processes
    )
    
    n_skipped = n_bootstraps - len(scores)
    if n_skipped > 0: 
        logging.warning(f'{n_skipped} bootstraps with no pos examples were '
                        'skipped, skipped bootstraps will be replaced with '
                        'original score')
        # fill skipped boostraps with the original score
        orig_score = [(
            roc_auc_score(Y_true, Y_pred_prob), 
            average_precision_score(Y_true, Y_pred_prob),
        )]
        scores += orig_score * n_skipped
        
    return scores
    
def auc_bootstrap_worker(bootstrap_partition, Y_true, Y_pred_prob):
    """Resample the labels and predictions with replacement and recompute the 
    AUROC/AUPRC score
    """
    scores = []
    for random_seed in bootstrap_partition:
        y_true = bootstrap_sample(Y_true, random_seed)
        y_pred_prob = Y_pred_prob[y_true.index]
        if y_true.nunique() < 2:
            continue
        scores.append((
            roc_auc_score(y_true, y_pred_prob), 
            average_precision_score(y_true, y_pred_prob)
        ))
    return scores

def nadir_bootstrap_worker(bootstrap_partition, df, cycle_length, cyto_thresh):
    """Resample patients with replacement and recompute nadir day
    """
    cycle_days = range(cycle_length)
    nadir_days = []
    ikns = df['ikn'].unique()
    for i in bootstrap_partition:
        np.random.seed(i)
        sampled_ikns = np.random.choice(ikns, len(ikns), replace=True)
        mask = df['ikn'].isin(sampled_ikns)
        sampled_df = df.loc[mask, cycle_days]
        cytopenia_rates_per_day = get_cyto_rates(sampled_df, cyto_thresh)
        if all(cytopenia_rates_per_day == 0):
            # if no event, pick random day in the cycle
            nadir_day = np.random.choice(cycle_days)
        else:
            nadir_day = np.argmax(cytopenia_rates_per_day)
        nadir_days.append(nadir_day+1)
    return nadir_days

def compute_bootstrap_nadir_days(
    df, 
    cycle_length, 
    cyto_thresh, 
    n_bootstraps=1000, 
    processes=32
):
    """Compute bootstrapped nadir days for computing their confidence intervals
    """
    worker = partial(
        nadir_bootstrap_worker, 
        df=df, 
        cycle_length=cycle_length,
        cyto_thresh=cyto_thresh
    )
    nadir_days = split_and_parallelize(
        range(n_bootstraps), worker, split_by_ikns=False, processes=processes
    )
    return nadir_days

def bootstrap_sample(data, random_seed):
    np.random.seed(random_seed)
    N = len(data)
    weights = np.random.random(N) 
    return data.sample(
        n=N, replace=True, random_state=random_seed, weights=weights
    )

###############################################################################
# Binomial Confidence Interval
###############################################################################
def get_calibration_confidence_interval(
    Y_true, 
    Y_pred_prob, 
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
        bins = np.percentile(Y_pred_prob, quantiles*100)
    elif calib_strategy == 'uniform':
        bins = np.linspace(0, 1.0, n_bins+1)
    else:
        raise ValueError('calib_strategy must be either quantile or uniform')

    # compute which bin each label belongs to
    # bin_ids = np.digitize(Y_pred_prob, bins[1:-1]) - 1
    # WARNING: Newer sklearn version uses the line below
    # Ensure you use the correct line according to your sklearn version
    bin_ids = np.searchsorted(bins[1:-1], Y_pred_prob)

    y_true = pd.Series(np.array(Y_true), index=bin_ids)
    lower_limit, upper_limit = [], []
    for idx, group in y_true.groupby(y_true.index): # loop through each bin
        # compute 95% CI for that bin using binormial confidence interval 
        # (because of binormial distribution (i.e. True, False))
        lower_ci, upper_ci = get_confidence_interval(group, method='binomial')
        lower_limit.append(lower_ci)
        upper_limit.append(upper_ci)
    return np.array(lower_limit), np.array(upper_limit)
