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
import os

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from sklearn import tree
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
from xgboost import plot_tree
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import logger
from src.config import root_path, cyto_folder, cytopenia_grades, blood_types
from src.utility import (
    twolevel, 
    load_pickle, 
    get_clean_variable_names, 
    pred_thresh_binary_search,
    time_to_x_after_y,
)
###############################################################################
# Feature Importance
###############################################################################
def importance_plot(
    alg, 
    target_events, 
    save_dir, 
    figsize, 
    top=20,              
    importance_by='feature', 
    padding=None, 
    colors=None,
    smaller_is_better=False,
):
    """
    Args:
        smaller_is_better (bool): If True, smaller value means higher 
            importance
    """
    # NOTE: run `python scripts/feat_imp.py` in the command line before running
    # this function
    if importance_by not in ['feature', 'group']: 
        raise ValueError('importance_by must be either "feature" or "group"')
    if padding is None: padding = {}
        
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.3)
    N = len(target_events)
    if colors is None: colors = [None,]*N
    
    filename = f'{alg}_{importance_by}_importance'
    df = pd.read_csv(f'{save_dir}/feat_importance/{filename}.csv')
    df = df.set_index('index')
    df.index = get_clean_variable_names(df.index)
    if smaller_is_better: df = -1 * df
    
    summary = pd.DataFrame()
    for idx, target_event in tqdm(enumerate(target_events)):
        feat_importances = df[target_event]
        feat_importances = feat_importances.sort_values(ascending=False)
        feat_importances = feat_importances[0:top]
        feat_importances = feat_importances.round(4)
        summary[target_event] = [f'{feat} ({importance})' 
                                 for feat, importance in feat_importances.items()]
        ax = fig.add_subplot(N,1,idx+1)
        ax.barh(
            feat_importances.index, feat_importances.values, 
            color=colors[idx]
        )
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance Score')
        remove_top_right_axis(ax)
        
        # write the results
        filename = f'{alg}_{target_event}'
        filepath = f'{save_dir}/figures/important_{importance_by}s/{filename}.jpg'
        fig.savefig(filepath, bbox_inches=get_bbox(ax, fig, **padding), dpi=300)
        
        ax.set_title(target_event) # set title AFTER saving individual figures
        
    filepath = f'{save_dir}/figures/important_{importance_by}s/{alg}.jpg'
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.show()
    
    summary.index += 1
    return summary

###############################################################################
# Subgroup Performance
###############################################################################
def subgroup_performance_plot(
    df, 
    target_event='ACU', 
    subgroups=None, 
    padding=None,                     
    figsize=(16,24), 
    xtick_rotation=45,
    ylim=None,
    save_dir=None
):
    # Setup
    if subgroups is None: 
        # NOTE: dict is guaranteed to preserve order (as of Python 3.7) while set 
        # does not. Reference: https://stackoverflow.com/questions/1653970
        subgroups = [idx[0] for idx in df.index]
        subgroups = list(dict.fromkeys(subgroups))
    if padding is None: padding = {}
    if save_dir is not None and not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    df = df.loc[subgroups]
    df.index = df.index.remove_unused_levels()
    metrics = df.columns.levels[1].tolist()
    bar_names = [idx[1] for idx in df.index]
    bar_names = [name.split('(')[0].strip() for name in bar_names]
    bar_names[0] = subgroups[0] # Edge Case: Entire Test Cohort
    
    start_pos = 0
    x_bar_pos = []
    nrows = len(metrics)
    to_float = lambda arr: np.array([float(x) for x in arr])
    
    def get_perf(metric, subgroup_name=None):
        if subgroup_name is None:
            tmp = df[(target_event, metric)]
        else:
            tmp = df.loc[subgroup_name, (target_event, metric)]
        scores = tmp.to_numpy()
        if scores.dtype == float: return scores, None, None
        scores, ci = zip(*[score.split(' ') for score in scores])
        lower, upper = zip(*[x.strip('()').split('-') for x in ci])
        return to_float(scores), to_float(lower), to_float(upper)

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize)
    axes = [axes] if nrows == 1 else axes.flatten()
    plt.subplots_adjust(hspace=0.6)
    
    # Create the Bar Plots
    for idx, subgroup_name in enumerate(subgroups):
        N = len(df.loc[subgroup_name])
        bar_pos = np.arange(start_pos, start_pos+N).tolist()
        x_bar_pos += bar_pos
        start_pos += N + 1
        for i, metric in enumerate(metrics):
            scores, lower, upper = get_perf(metric, subgroup_name)
            yerr = None if lower is None else [scores - lower, upper - scores]
            axes[i].bar(
                bar_pos, scores, label=subgroup_name, width=0.8, yerr=yerr, 
                capsize=5
            )
        
    for i, metric in enumerate(metrics):
        scores, lower, upper = get_perf(metric)
        
        # create reference line
        axes[i].axhline(y=scores[0], color='black', linestyle='--')
        
        # adjust y-axis
        if ylim is None:
            y_bottom = 0 if min(scores) <= 0.2 else min(scores) - 0.1
            axes[i].set_ylim(bottom=y_bottom)
        else:
            axes[i].set_ylim(**ylim[metric])
        
        # adjust x-axis position of bars 
        # Alternative way: plt.xticks(x_bar_pos, bar_names, rotation=xtick_rotation)
        axes[i].set_xticks(x_bar_pos)
        axes[i].set_xticklabels(bar_names, rotation=xtick_rotation)
        
        axes[i].set_ylabel(metric)
        axes[i].legend(bbox_to_anchor=(1,0), loc='lower left', frameon=False)
        remove_top_right_axis(axes[i])
        
        # write the results
        if save_dir is not None: 
            fig.savefig(
                f'{save_dir}/{target_event}_{metric}.jpg', 
                bbox_inches=get_bbox(axes[i], fig, **padding), dpi=300
            )
    plt.show()
    
###############################################################################
# Time Interval
###############################################################################
def time_to_event_plot(time_to_event, ax, plot_type='cdf', **kwargs):
    if plot_type not in ['cdf', 'hist']:
        raise NotImplementedError('plot_type only supports cdf (Cumulative '
                                  'Distribution Function) and hist (Histogram)')
    ax.grid(zorder=0)
    if plot_type == 'hist':
        bins = int(max(time_to_event) - min(time_to_event))
        sns.histplot(
            time_to_event, ax=ax, bins=bins, zorder=2
        )
    elif plot_type == 'cdf':
        time_to_event = sorted(time_to_event)
        N = len(time_to_event)
        ax.plot(time_to_event,  np.arange(N) / float(N))
    ax.set(**kwargs)
    
###############################################################################
# Cytopenia
###############################################################################
def below_threshold_bar_plot(
    df, 
    threshold, 
    save=False, 
    filename='bar_plot', 
    color=None
):
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    n_regimens = df['regimen'].nunique()
    height = (n_regimens // 2) * 10
    fig = plt.figure(figsize=(16, height))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    idx = 1
    for regimen, group in tqdm(df.groupby('regimen')):
        days = np.arange(cycle_lengths[regimen]+1)
        get_n_patients = lambda mask: group.loc[mask, 'ikn'].nunique()
        
        n_patients = []
        n_patients_below_thresh = []
        for day in days:
            mask = group[day].notnull()
            n_patients.append(get_n_patients(mask))
            
            mask = group[day] < threshold
            n_patients_below_thresh.append(get_n_patients(mask))
            
        n_patients = np.array(n_patients)
        n_patients_below_thresh = np.array(n_patients_below_thresh)
        
        # cannot display data summary with observations less than 6, replace
        # them with 6
        mask = (n_patients < 6) & (n_patients > 0)
        n_patients[mask] = 6
        mask = (n_patients_below_thresh < 6) & (n_patients_below_thresh > 0)
        n_patients_below_thresh[mask] = 6
        
        perc_patients_below_thresh = n_patients_below_thresh / n_patients
        
        add_on = f'with blood count < {threshold}'
        n_patients_map = {
            'Number of patients': n_patients,
            f'Number of patients\n{add_on}': n_patients_below_thresh,
            f'Percentage of patients\n{add_on}': perc_patients_below_thresh
        }
        for i, (ylabel, y) in enumerate(n_patients_map.items()):
            ax = fig.add_subplot(n_regimens, 3, idx+i)
            plt.bar(days+1, y, color=color)
            plt.title(regimen)
            plt.ylabel(ylabel)
            plt.xlabel('Day')
            plt.xticks(get_day_xticks(days))
        idx += 3
    plt.text(
        0, -0.3, '*Observations < 6 are displayed as 6', 
        transform=ax.transAxes, fontsize=12
    )
    if save:
        filepath = f'{root_path}/{cyto_folder}/analysis/figures/{filename}.jpg'
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.show()

def iqr_plot(
    df, 
    unit='10^9/L', 
    show_outliers=True, 
    save=False, 
    filename='iqr_plot'
):
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    n_regimens = df['regimen'].nunique()
    height = (n_regimens // 2) * 10
    fig = plt.figure(figsize=(16, height))
    plt.subplots_adjust(hspace=0.3)
    
    nadir_dict = {}
    for idx, (regimen, group) in tqdm(enumerate(df.groupby('regimen'))):
        days = np.arange(cycle_lengths[regimen]+1).astype(int)
        data = [group[day].dropna().values for day in days]
        data = np.array(data, dtype=object)
        ax = fig.add_subplot(n_regimens, 2, idx+1)
        bp = plt.boxplot(data, labels=days+1, showfliers=show_outliers)
        plt.title(regimen)
        plt.ylabel(f'Blood Count ({unit})')
        plt.xlabel('Day')
    
        medians = [median.get_ydata()[0] for median in bp['medians']]
        min_idx = np.nanargmin(medians)
        nadir_dict[regimen] = {
            'Day of Nadir': min_idx-5, 
            'Depth of Nadir (Min Blood Count)': medians[min_idx]
        }
    
        plt.plot(days+1, medians, color='red')

    if save:
        filepath = f'{root_path}/{cyto_folder}/analysis/figures/{filename}.jpg'
        plt.savefig(filepath, bbox_inches='tight', dpi=300)    
    plt.show()
    
    nadir_df = pd.DataFrame(nadir_dict)
    return nadir_df.T

def event_rate_stacked_bar_plot(
    df, 
    regimens, 
    save_dir, 
    cytopenia='Neutropenia', 
    figsize=(16,4), 
    save=True
):
    """Plot cytopenia event rate over days since chemo administration
    """
    if cytopenia not in ['Neutropenia', 'Anemia', 'Thrombocytopenia']: 
        raise ValueError('cytopenia must be one of Neutropneia, Anemia, or '
                         'Thrombocytopenia')
        
    cycle_lengths = dict(df[['regimen', 'cycle_length']].to_numpy())
    kwargs = {'nrows': 1, 'ncols': 3, 'figsize': figsize}
    fig, axes = plt.subplots(**kwargs)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, regimen in enumerate(regimens):
        if regimen not in cycle_lengths: continue
        days = np.arange(cycle_lengths[regimen]+1)
        group = df.query('regimen == @regimen')[days]
        
        for grade, thresholds in cytopenia_grades.items():
            if cytopenia not in thresholds: continue
            thresh = thresholds[cytopenia]
            cyto_rates = [(group[day].dropna() < thresh).mean() for day in days]
            
            # set day 1 as day of administration (instead of day 0)
            axes[i].bar(
                days+1, cyto_rates, label=f'{grade} {cytopenia} (<{thresh})',
                color=np.array(thresholds['color'])/255
            ) 
            
        axes[i].set_ylabel(f'{cytopenia} Event Rate')
        axes[i].set_xlabel('Day')
        axes[i].set_xticks(get_day_xticks(days))
        axes[i].set_title(regimen.upper())
        if i == len(regimens) - 1:
            axes[i].legend(bbox_to_anchor=(1,0), loc='lower left', frameon=False)
    if save: 
        filepath = f'{save_dir}/analysis/figures/{cytopenia}_{"_".join(regimens)}.jpg'
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.show()
    
###############################################################################
# Palliative Care Consultation Service (PCCS)
###############################################################################
def epc_subgroup_plot(
    summaries, 
    target_event='365d Mortality', 
    y_top=0.4, 
    **kwargs
):
    """Plot the proportion of patients who died and received early palliative 
    care (EPC) (defined as receival of palliative care consultation service 
    (PCCS) 6 months before death) for different subgroup populations.
    
    Args:
        summaries (dict): A nested map of care type (str) and each of the 
            subgroup population's (str) service receival summary (pd.DataFrame)
            e.g. {'usual': {'Sex': pd.DataFrame, 'Age': pd.DataFrame}}
        **kwargs: keyword arguments fed into subgroup_performance_plot
    """
    # Setup the dataframe to call subgroup_performance_plot
    df = pd.DataFrame(index=twolevel, columns=twolevel)
    care_map = {'usual': 'Usual Care', 'system': 'System-Guided Care'}
    ylim = {}
    for care_name, summary in summaries.items():
        care_name = care_map[care_name]
        metric = ('Proportion of Patients Who Died\nWith Early Palliative '
                  f'Care in {care_name}')
        col = (target_event, metric)
        _epc_subgroup_helper(summary, care_name, col, df)
        ylim[metric] = {'bottom': 0, 'top': y_top}
    subgroup_performance_plot(
        df, target_event=target_event, ylim=ylim, **kwargs
    )
    return df
    
def _epc_subgroup_helper(summary, care_name, col, df):
    get_total = lambda m: m.loc[['Got PCCS', 'Not Got PCCS']].sum().sum()
    for subgroup, result in summary.items():
        for category in result.columns.levels[0]:
            counts = result[category]
            rate = counts.loc['Got Early PCCS (%)', 'Dead'] / 100
            total = get_total(counts)
            lower, upper = proportion_confint(int(total * rate), total)
            score = f'{rate:.3f} ({lower:.3f}-{upper:.3f})'
            df.loc[(subgroup, category), col] = score
        
def epc_impact_plot(impact, N, epc=True):
    """Plot the model's clinical impact on early palliative care (EPC)."""
    receival_status = 'With' if epc else 'Without'
    rate = impact.loc[f'Died {receival_status} Early PCCS'] / N
    lower_ci, upper_ci = proportion_confint(N * rate, N)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    yerr = [rate - lower_ci, upper_ci - rate]
    bar = ax.bar(
        x=rate.index, height=rate, yerr=yerr, capsize=10, 
        color=['#1f77b4', '#ff7f0e']
    )
    ax.margins(0.1)
    ax.set_ylabel(f'Proportion Of Patients Who Died\n{receival_status} Early '
                  'Palliative Care')
    remove_top_right_axis(ax)
    
    df = pd.concat([rate, upper_ci, lower_ci], axis=1).T
    df.index = [f'Died {receival_status} EPC Rate', 'Upper CI', 'Lower CI']
    diff = df['System-Guided Care'] - df['Usual Care'].to_numpy()[[0, 2, 1]]
    df['Difference'] = diff
    return df
        
def post_pccs_survival_plot(
    df,
    months_after=24,
    time_frame=None,
    verbose=True, 
    **kwargs
):
    """Plot the survival analysis of patients after they receive their first
    palliative care consulation service (PCCS).
    
    Args:
        df (pd.DataFrame): table of patients and their relevant data (PCCS 
            receival, vital status, etc) (from output of get_pccs_analysis_data)
        months_after (int): the number of months after patients' first PCCS of 
            which we report their survival probability
        **kwargs: keyword arguments fed into time_to_x_after_y
    """
    if time_frame is None: time_frame = np.arange(0, 37, 6)
    is_dead = df['status'] == 'Dead'
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    
    care_names = ['Usual Care', 'System-Guided Care']
    times, statuses, kmfs = [], [], []
    prob = pd.DataFrame(index=['Survival Probability', 'Upper CI', 'Lower CI'])
    
    for care_name in care_names:
        if verbose:
            msg = ('Analyzing time to death after initial PCCS for '
                   f'{care_name}...')
            logger.info(msg)
        
        time_to_death = time_to_x_after_y(
            df.copy(), x='last_obs', y='first_pccs', care_name=care_name, 
            **kwargs
        )
        status = is_dead.loc[time_to_death.index]
        
        kmf = KaplanMeierFitter()
        kmf.fit(time_to_death, status)
        kmf.plot(ax=ax, label=care_name, loc=slice(time_frame[-1]))
        
        # report survival probability
        ci = kmf.confidence_interval_survival_function_
        lower, upper = ci[ci.index > months_after].iloc[0]
        surv_prob = kmf.predict(months_after)
        prob[care_name] = [surv_prob, upper, lower]
        
        # clip number of patients at risk to 6 for privacy policy
        kmf.event_table['at_risk'].clip(lower=6+1, inplace=True)
        
        times.append(time_to_death)
        statuses.append(status)
        kmfs.append(kmf)

    xlabel = 'Months After First Palliative Care Consultation'
    add_at_risk_counts(
        *kmfs, ax=ax, xticks=time_frame[1:], labels=care_names, 
        rows_to_show=['At risk']
    )
    ax.set(
        xticks=time_frame, ylim=(-0.05,1.05), xlabel=xlabel, 
        ylabel='Survival Probability'
    )
    ax.legend(frameon=False)
    
    # Show p-value
    p_value = logrank_test(*times, *statuses).p_value
    logger.info(f'P-value = {p_value}')
    
    diff = prob['System-Guided Care'] - prob['Usual Care'].to_numpy()[[0, 2, 1]]
    prob['Difference'] = diff
    return prob

def pccs_graph_plot(df):
    """Visualize the number and proportion of patients who are dead, alive,
    received palliative care consultation service (PCCS), did not receive PCCS,
    alerted, not alerted, etc, as a tree-based graph.
    
    Args:
        df (pd.DataFrame): table of patients and their relevant data (PCCS receival,
            vital status, alert status, etc) (from output of get_pccs_analysis_data)
    """
    d = graphviz.Digraph(
        graph_attr={'rankdir': 'LR'}, 
        edge_attr={}, 
        node_attr={'shape': 'box'}
    )
    
    N = len(df)
    f = lambda n: f'{n} ({n/N*100:.1f}%)'
    total_str = f'Total\n{N} (100%)'

    for vital_status, vital_group in df.groupby('status'):
        # Patient vital status
        vital_str = f'{vital_status}\n{f(len(vital_group))}'
        d.edge(total_str, vital_str)

        # Patient PCCS receival status
        mask = vital_group['first_PCCS_date'].notnull()
        for receival_status, receival_group in vital_group.groupby(mask):
            name = 'Received PCCS' if receival_status else 'No PCCS'
            receival_str = f'{name}\n{f(len(receival_group))}'
            d.edge(vital_str, receival_str)

            if vital_status == 'Alive':
                # Patient Alarm Status
                mask = receival_group['predicted']
                d.edge(receival_str, f'Alerted\n{f(sum(mask))}')
                d.edge(receival_str, f'Not Alerted\n{f(sum(~mask))}')

            elif vital_status == 'Dead':
                if receival_status:
                    # Patient early PCCS receival status
                    mask = receival_group['received_early_pccs']
                    grouping = receival_group.groupby(mask)
                    for early_receival_status, early_receival_group in grouping:
                        name = 'Early' if early_receival_status else 'Late'
                        early_str = f'{name}\n{f(len(early_receival_group))}'
                        d.edge(receival_str, early_str)

                        # Patient Alarm Status
                        mask1 = early_receival_group['early_pccs_by_alert']
                        mask2 = early_receival_group['predicted']
                        d.edge(early_str, f'Alerted Early\n{f(sum(mask1))}')
                        d.edge(early_str, f'Alerted Late\n{f(sum(~mask1 & mask2))}')
                        d.edge(early_str, f'Not Alerted\n{f(sum(~mask2))}')
                else:
                    # Patient Alarm Status
                    mask1 = receival_group['early_pccs_by_alert']
                    mask2 = receival_group['predicted']
                    d.edge(receival_str, f'Alerted Early\n{f(sum(mask1))}')
                    d.edge(receival_str, f'Alerted Late\n{f(sum(~mask1 & mask2))}')
                    d.edge(receival_str, f'Not Alerted\n{f(sum(~mask2))}')
    return d

def epc_bias_mitigation_plot(
    eval_models, 
    pccs_result, 
    subgroup_masks,
    alg='ENS',
    target_event='365d Mortality',
    split='Test',
    colors=None,
    save_path=None,
    padding=None,
):
    """Plot the bias among subgroups for receival of early palliative care (EPC)
    and how the model performs comparably among these subgroups (indicating 
    the model can potentially mitigate the bias)
    
    Args:
        subgroup_masks: A nested map of subgroup category (str) and each of 
            their subgroup's (str) boolean aligned series (pd.Series), which 
            indicates which samples in the original data belongs to the subgroup
            e.g. {'Area of Residence': {'Urban': pd.Series, 'Rural': pd.Series}}
    """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
    if padding is None: 
        padding = {}
        
    Y_true = eval_models.labels[split][target_event]
    Y_pred_prob = eval_models.preds[alg][split][target_event]

    N = len(subgroup_masks)
    fig, axes = plt.subplots(nrows=N, ncols=5, figsize=(30,6*N))
    if N == 1: axes = [axes]

    def get_rates(df):
        total = df.loc[['Got PCCS', 'Not Got PCCS'], ['Alive', 'Dead']].sum().sum()
        rate = df.loc['Got Early PCCS (%)', 'Dead'] / 100
        lower_ci, upper_ci = proportion_confint(int(total * rate), total)
        return [rate, lower_ci, upper_ci]

    results = {}
    for i, (catname, masks) in enumerate(subgroup_masks.items()):
        masks = {'Overall': Y_true.notnull(), **masks}
        data = []
        for j, (subgroup, mask) in enumerate(masks.items()):
            # Plot AUPRC and AUROC
            for k, curve_type in enumerate(['pr', 'roc']):
                eval_models.plot_auc_curve(
                    axes[i][2+k], Y_true[mask], Y_pred_prob[mask], 
                    curve_type=curve_type, color=colors[j], 
                    label_prefix=f'{subgroup}\n', 
                    ci_name=f'{alg}_{split}_{target_event}_{subgroup}'
                )

            # Plot calibration
            last = bool(j == len(masks) - 1)
            eval_models.plot_calib(
                axes[i][4], Y_true[mask], Y_pred_prob[mask], 
                color=colors[j], show_perf_calib=last, 
                label_prefix=f'{subgroup}\n',
            )
            
            # Compute rate (95% CI) of death without EPC
            if subgroup == 'Overall': 
                usual_rates = get_rates(pccs_result['usual'][f'Entire {split} Cohort'][''])
                system_rates = get_rates(pccs_result['system'][f'Entire {split} Cohort'][''])
            else:
                usual_rates = get_rates(pccs_result['usual'][catname][subgroup])
                system_rates = get_rates(pccs_result['system'][catname][subgroup])
            
            # Some long subgroup names need to be split up for visualization
            subgroup = subgroup.replace(' ', '\n')
            data.append([subgroup] + usual_rates + system_rates)

        # Plot rate of death with EPC
        cols = [
            'subgroup', 'usual_rate', 'usual_lower_ci', 'usual_upper_ci', 
            'system_rate', 'system_lower_ci', 'system_upper_ci'
        ]
        df = pd.DataFrame(data=data, columns=cols)
        for x, care in enumerate(['usual', 'system']):
            rate_col = f'{care}_rate'
            yerr = [
                df[rate_col] - df[f'{care}_lower_ci'],
                df[f'{care}_upper_ci'] - df[rate_col]
            ]
            # sns.barplot(data=df, x='subgroup', y=f'{care}_rate', ax=axes[i][0], yerr=yerr)
            axes[i][x].bar(
                df['subgroup'], df[rate_col], yerr=yerr, capsize=5, color=colors
            )
            remove_top_right_axis(axes[i][x])
            if min(df[rate_col]) > 0.2: 
                axes[i][x].set_ylim(bottom=min(df[rate_col])-0.2)
            axes[i][x].set_xlabel(f'Subgroup Population - {catname}')
            care_name = 'System-Guided' if care == 'system' else 'Usual'
            axes[i][x].set_ylabel('Proportion of Patients Who Died\nWith Early '
                                  f'Palliative Care in {care_name} Care')
        # set ylim of usual care to be same as system care
        axes[i][0].set_ylim(axes[i][1].get_ylim())
        
        # write the results
        if save_path is not None: 
            for idx, ax in enumerate(axes[i]):
                fig.savefig(
                    f'{save_path}/{catname}_bias{idx}.jpg', 
                    bbox_inches=get_bbox(ax, fig, **padding), dpi=300
                )
        results[catname] = df     
    return results
        
###############################################################################
# Misc Cytopenia Plots
###############################################################################
def blood_count_dist_plot(df, include_sex=True, bins=51):
    fig = plt.figure(figsize=(20,5))
    for idx, blood_type in enumerate(blood_types):
        ax = fig.add_subplot(1,3,idx+1)
        col = f'baseline_{blood_type}_value'
        kwargs = {'data': df, 'x': col, 'ax': ax, 'kde': True, 'bins': bins}
        if include_sex: kwargs['hue'] = 'sex'
        sns.histplot(**kwargs)
        plt.xlabel('Blood Count Value')
        plt.title(blood_type)
        
def day_dist_plot(df, regimens):
    kwargs = {'nrows': int(len(regimens)/3), 'ncols': 3, 'figsize': (16,10)}
    fig, axes = plt.subplots(**kwargs)
    axes = axes.flatten()
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    for i, regimen in enumerate(regimens):
        if regimen not in cycle_lengths: continue
        days = np.arange(cycle_lengths[regimen]+1)
        group = df.loc[df['regimen'] == regimen, days]
        dist = [len(group[day].dropna()) for day in days]
        # set day 1 as day of administration (instead of day 0)
        axes[i].bar(days+1, dist)
        axes[i].set_xticks(get_day_xticks(days))
        axes[i].set_title(regimen)

def regimen_dist_plot(df, by='patient'):
    if by == 'patients':
        # number of patients per cancer regiment
        n_patients = df.groupby('regimen')['ikn'].nunique()
        dist = n_patients.sort_values()
        ylabel = 'Number of Patients'
    elif by == 'blood_counts':
        # number of blood counts per regimen
        cycle_lengths = dict(df[['regimen', 'cycle_length']].to_numpy())
        def func(group):
            cycle_length = int(cycle_lengths[group.name])
            days_range = range(-5,cycle_length)
            mask = group[days_range].notnull()
            return mask.sum(axis=1).sum()
        n_blood_counts = df.groupby('regimen').apply(func)
        dist = n_blood_counts.sort_values()
        ylabel = 'Number of Blood Measurements'
    elif by == 'sessions':
        # number of treatment sessions per regimen
        n_sessions = df.groupby('regimen').apply(len)
        dist = n_sessions.sort_values()
        ylabel = 'Number of Sessions'
    fig = plt.figure(figsize=(15,5))
    plt.bar(dist.index, dist.values) 
    plt.xlabel('Treatment Regiments')
    plt.ylabel(ylabel)
    plt.xticks(rotation=90, fontsize=7)
    plt.show()

def scatter_plot(df, unit='10^9/L', save=False, filename="scatter_plot"):
    n_regimens = df['regimen'].nunique()
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    height = (n_regimens // 2) * 10
    fig = plt.figure(figsize=(10,height))
    for idx, (regimen, group) in tqdm(enumerate(df.groupby('regimen'))):
        days = np.arange(cycle_lengths[regimen]+1)
        y = group[days].values.flatten()
        x = np.array(list(days+1)*len(group))

        ax = fig.add_subplot(n_regimens,2,idx+1)
        plt.subplots_adjust(hspace=0.3)
        plt.scatter(x, y, alpha=0.03)
        plt.title(regimen)
        plt.ylabel(f'Blood Count ({unit})')
        plt.xlabel('Day')
        plt.xticks(get_day_xticks(days))
    if save: 
        filepath = f'{root_path}/{cyto_folder}/analysis/figures/{filename}.jpg'
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.show()

def violin_plot(df, unit='10^9/L', save=False, filename='violin_plot'):
    n_regimens = df['regimen'].nunique()
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    height = (n_regimens // 2) * 10
    fig = plt.figure(figsize=(16, height))
    plt.subplots_adjust(hspace=0.3)
    for idx, (regimen, group) in tqdm(enumerate(df.groupby('regimen'))):
        chemo_days = np.arange(cycle_lengths[regimen]+1).astype(int)
        ax = fig.add_subplot(n_regimens,2,idx+1)
        blood_counts = []
        observation_days = []
        for day in chemo_days:
            values = group[day].dropna().tolist()
            blood_counts += values
            observation_days += [day+1,]*len(values)
        data = pd.DataFrame(
            zip(blood_counts, observation_days), 
            columns=[f'Blood Count ({unit})', 'Day']
        )
        sns.violinplot(x='Day', y=f'Blood Count ({unit})', data=data, ax=ax)
        plt.title(regimen)
    if save:
        filepath = f'{root_path}/{cyto_folder}/analysis/figures/{filename}.jpg'
        plt.savefig(filepath, bbox_inches='tight', dpi=300)    

def mean_cycle_plot(df, unit='10^9/L', save=False, filename='mean_cycle_plot'):
    n_regimens = df['regimen'].nunique()
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    height = (n_regimens // 2) * 10
    fig = plt.figure(figsize=(10, height))
    plt.subplots_adjust(hspace=0.3)
    cycles = [1,2,3,4,5]
    for idx, (regimen, group) in tqdm(enumerate(df.groupby('regimen'))):
        days = np.arange(cycle_lengths[regimen]+1)
        ax = fig.add_subplot(n_regimens,2,idx+1)
    
        for cycle in cycles:
            tmp_df = group[group['chemo_cycle'] == cycle]
            medians = tmp_df[days].median().values
            plt.plot(days+1, medians)
    
        plt.title(regimen)
        plt.ylabel(f'Median Blood Count ({unit})')
        plt.xlabel('Day')
        plt.xticks(get_day_xticks(days))
        plt.legend([f'cycle{c}' for c in cycles])
    if save:
        filepath = f'{root_path}/{cyto_folder}/analysis/figures/{filename}.jpg'
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.show()
    
###############################################################################
# Pearson Correlation
###############################################################################
def pearson_plot(pearson_matrix, main_target='ACU', save_path=None):
    fig, ax = plt.subplots(figsize=(15,6))
    indices = pearson_matrix[main_target].sort_values().index
    indices = indices[~indices.str.contains('Is Missing')]
    ax.plot(pearson_matrix.loc[indices], marker='o')
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
    ax.set_xlabel('Feature Columns', fontsize=12)
    plt.legend(pearson_matrix.columns, fontsize=10)
    plt.xticks(rotation=90)
    if save_path is not None:
        filepath = f'{save_path}/figures/pearson_coefficient.jpg'
        plt.savefig(filepath, bbox_inches='tight', dpi=300)

###############################################################################
# Tree Models
###############################################################################   
def tree_plot(train, target_event='Neutropenia', alg='RF'):
    """Plot a decision tree from Random Forest model or XGBoost model as a 
    visualization/interpretation example
    """
    if alg not in ['RF', 'XGB']: 
        raise ValueError('alg must be either RF or XGB')
    
    # the model is a multioutput calibrated classifier (made up of multiple 
    # classifiers )
    model = load_pickle(train.output_path, alg)
    
    # get a single example classifier
    idx = train.target_events.index(target_event)
    cv_fold = 0
    clf = model.estimators_[idx].calibrated_classifiers_[cv_fold].estimator
    
    if alg == 'RF': 
        feature_names = clf.feature_names_in_
        clf = clf.estimators_[0]
        
    # plot the tree
    fig, ax = plt.subplots(figsize=(150,20))
    if alg == 'RF':
        tree.plot_tree(
            clf, feature_names=feature_names, 
            class_names=[f'Not {target_event}', target_event], fontsize=10, 
            filled=True, ax=ax
        )
    else:
        plot_tree(clf, ax=ax)
    
###############################################################################
# Helper Functions
###############################################################################
def remove_top_right_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def get_bbox(ax, fig, pad_x0=0.75, pad_y0=0.5, pad_x1=0.15, pad_y1=0.1):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox.x0 -= pad_x0
    bbox.y0 -= pad_y0
    bbox.x1 += pad_x1
    bbox.y1 += pad_y1
    return bbox

def get_day_xticks(days):
    xticks = days+1
    if len(days) > 22: xticks = xticks[::3] # skip every 3rd tick
    elif len(days) > 16: xticks = xticks[::2] # skip every 2nd tick
    return xticks
