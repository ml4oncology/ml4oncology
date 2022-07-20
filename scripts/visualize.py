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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn import tree
from scripts.config import (root_path, cyto_folder, cytopenia_gradings, blood_types)
from scripts.utility import (load_ml_model, get_clean_variable_names)

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

def blood_count_dist_plot(df, include_sex=True):
    fig = plt.figure(figsize=(20,5))
    for idx, blood_type in enumerate(blood_types):
        ax = fig.add_subplot(1,3,idx+1)
        kwargs = {'data': df, 'x': f'baseline_{blood_type}_count', 'ax': ax, 'kde': True, 'bins': 50}
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
        axes[i].bar(days+1, dist) # set day 1 as day of administration (instead of day 0)
        axes[i].set_xticks(get_day_xticks(days))
        axes[i].set_title(regimen)

def regimen_dist_plot(df, by='patient'):
    if by == 'patients':
        # number of patients per cancer regiment
        n_patients = df.groupby('regimen').apply(lambda group: group['ikn'].nunique())
        dist = n_patients.sort_values()
        ylabel = 'Number of Patients'
    elif by == 'blood_counts':
        # number of blood counts per regimen
        cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
        func = lambda group: (group[range(-5,int(cycle_lengths[group.name]))].notnull()).sum(axis=1).sum()
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
    plt.xlabel('Chemotherapy Regiments')
    plt.ylabel(ylabel)
    plt.xticks(rotation=90, fontsize=7)
    plt.show()

def scatter_plot(df, unit='10^9/L', save=False, filename="scatter_plot"):
    num_regimen = df['regimen'].nunique()
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    height = (num_regimen // 2) * 10
    fig = plt.figure(figsize=(10,height))
    for idx, (regimen, group) in tqdm.tqdm(enumerate(df.groupby('regimen'))):
        days = np.arange(cycle_lengths[regimen]+1)
        y = group[days].values.flatten()
        x = np.array(list(days+1)*len(group))

        ax = fig.add_subplot(num_regimen,2,idx+1)
        plt.subplots_adjust(hspace=0.3)
        plt.scatter(x, y, alpha=0.03)
        plt.title(regimen)
        plt.ylabel(f'Blood Count ({unit})')
        plt.xlabel('Day')
        plt.xticks(get_day_xticks(days))
    if save: 
        plt.savefig(f'{root_path}/{cyto_folder}/plots/{filename}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def below_threshold_bar_plot(df, threshold, save=False, filename='bar_plot', color=None):
    num_regimen = df['regimen'].nunique()
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    height = (num_regimen // 2) * 10
    fig = plt.figure(figsize=(16, height))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    idx = 1
    for regimen, group in tqdm.tqdm(df.groupby('regimen')):
        days = np.arange(cycle_lengths[regimen]+1)
        num_patients = np.array([group.loc[group[day].notnull(), 'ikn'].nunique() for day in days])
        num_patient_below_threshold = np.array([group.loc[group[day] < threshold, 'ikn'].nunique() for day in days])
        # cannot display data summary with observations less than 6, replace them with 6
        num_patients[(num_patients < 6) & (num_patients > 0)] = 6
        num_patient_below_threshold[(num_patient_below_threshold < 6) & (num_patient_below_threshold > 0)] = 6
        perc_patient_below_threshold = num_patient_below_threshold/num_patients
        
        for i, (ylabel, y) in enumerate([('Number of patients', num_patients), 
                                         (f'Number of patients\nwith blood count < {threshold}', num_patient_below_threshold),
                                         (f'Percentage of patients\nwith blood count < {threshold}', perc_patient_below_threshold)]):
            ax = fig.add_subplot(num_regimen,3,idx+i)
            plt.bar(days+1, y, color=color)
            plt.title(regimen)
            plt.ylabel(ylabel)
            plt.xlabel('Day')
            plt.xticks(get_day_xticks(days))
        idx += 3
    plt.text(0, -0.3, '*Observations < 6 are displayed as 6', transform=ax.transAxes, fontsize=12)
    if save:
        plt.savefig(f'{root_path}/{cyto_folder}/plots/{filename}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def iqr_plot(df, unit='10^9/L', show_outliers=True, save=False, filename='iqr_plot'):
    num_regimen = df['regimen'].nunique()
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    height = (num_regimen // 2) * 10
    fig = plt.figure(figsize=(16, height))
    plt.subplots_adjust(hspace=0.3)
    nadir_dict = {}
    for idx, (regimen, group) in tqdm.tqdm(enumerate(df.groupby('regimen'))):
        days = np.arange(cycle_lengths[regimen]+1).astype(int)
        data = np.array([group[day].dropna().values for day in days], dtype=object)
        ax = fig.add_subplot(num_regimen,2,idx+1)
        bp = plt.boxplot(data, labels=days+1, showfliers=show_outliers)
        plt.title(regimen)
        plt.ylabel(f'Blood Count ({unit})')
        plt.xlabel('Day')
    
        medians = [median.get_ydata()[0] for median in bp['medians']]
        min_idx = np.nanargmin(medians)
        nadir_dict[regimen] = {'Day of Nadir': min_idx-5, 'Depth of Nadir (Min Blood Count)': medians[min_idx]}
    
        plt.plot(days+1, medians, color='red')

    if save:
        plt.savefig(f'{root_path}/{cyto_folder}/plots/{filename}.jpg', bbox_inches='tight', dpi=300)    
    plt.show()
    nadir_df = pd.DataFrame(nadir_dict)
    return nadir_df.T

def violin_plot(df, unit='10^9/L', save=False, filename='violin_plot'):
    num_regimen = df['regimen'].nunique()
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    height = (num_regimen // 2) * 10
    fig = plt.figure(figsize=(16, height))
    plt.subplots_adjust(hspace=0.3)
    for idx, (regimen, group) in tqdm.tqdm(enumerate(df.groupby('regimen'))):
        chemo_days = np.arange(cycle_lengths[regimen]+1).astype(int)
        ax = fig.add_subplot(num_regimen,2,idx+1)
        blood_counts = []
        observation_days = []
        for day in chemo_days:
            values = group[day].dropna().tolist()
            blood_counts += values
            observation_days += [day+1,]*len(values)
        data = pd.DataFrame(zip(blood_counts, observation_days), columns=[f'Blood Count ({unit})', 'Day'])
        sns.violinplot(x='Day', y=f'Blood Count ({unit})', data=data, ax=ax)
        plt.title(regimen)
    if save:
        plt.savefig(f'{root_path}/{cyto_folder}/plots/{filename}.jpg', bbox_inches='tight', dpi=300)    

def mean_cycle_plot(df, unit='10^9/L', save=False, filename='mean_cycle_plot'):
    num_regimen = df['regimen'].nunique()
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    height = (num_regimen // 2) * 10
    fig = plt.figure(figsize=(10, height))
    plt.subplots_adjust(hspace=0.3)
    cycles = [1,2,3,4,5]
    for idx, (regimen, group) in tqdm.tqdm(enumerate(df.groupby('regimen'))):
        days = np.arange(cycle_lengths[regimen]+1)
        ax = fig.add_subplot(num_regimen,2,idx+1)
    
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
        plt.savefig(f'{root_path}/{cyto_folder}/plots/{filename}.jpg', bbox_inches='tight', dpi=300)
    plt.show()
    
def event_rate_stacked_bar_plot(df, regimens, save_dir, cytopenia='Neutropenia', figsize=(16,4), save=True):
    """Plot cytopenia event rate over days since chemo administration
    """
    if cytopenia not in {'Neutropenia', 'Anemia', 'Thrombocytopenia'}: 
        raise ValueError('cytopenia must be one of Neutropneia, Anemia, or Thrombocytopenia')
    kwargs = {'nrows': 1, 'ncols': 3, 'figsize': figsize}
    fig, axes = plt.subplots(**kwargs)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    for i, regimen in enumerate(regimens):
        if regimen not in cycle_lengths: continue
        days = np.arange(cycle_lengths[regimen]+1)
        group = df.loc[df['regimen'] == regimen, days]
        
        for grade, thresholds in cytopenia_gradings.items():
            if cytopenia not in thresholds: continue
            thresh = thresholds[cytopenia]
            cytopenia_rates = [(group[day].dropna() < thresh).mean() for day in days]
            
            axes[i].bar(days+1, # set day 1 as day of administration (instead of day 0)
                        cytopenia_rates, 
                        label=f'{grade} {cytopenia} (<{thresh})', 
                        color=np.array(thresholds['color'])/255) 
        axes[i].set_ylabel(f'{cytopenia} Event Rate')
        axes[i].set_xlabel('Day')
        axes[i].set_xticks(get_day_xticks(days))
        axes[i].set_title(regimen.upper())
        if i == len(regimens) - 1:
            axes[i].legend(bbox_to_anchor=(1,0), loc='lower left', frameon=False)
    if save: 
        plt.savefig(f'{save_dir}/plots/{cytopenia}_{"_".join(regimens)}.jpg', bbox_inches='tight', dpi=300)
    plt.show()
        
def pearson_plot(pearson_matrix, output_path, main_target='ACU'):
    fig, ax = plt.subplots(figsize=(15,6))
    indices = pearson_matrix[main_target].sort_values().index
    indices = indices[~indices.str.contains('Is Missing')]
    ax.plot(pearson_matrix.loc[indices], marker='o')
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
    ax.set_xlabel('Feature Columns', fontsize=12)
    plt.legend(pearson_matrix.columns, fontsize=10)
    plt.xticks(rotation='90')
    plt.savefig(f'{output_path}/figures/pearson_coefficient.jpg', bbox_inches='tight', dpi=300)
    
def tree_plot(train, target_type='Neutropenia', algorithm='RF'):
    """
    Plot a decision tree from Random Forest model or XGBoost model as a visualization/interpretation example
    """
    if algorithm not in {'RF', 'XGB'}: 
        raise ValueError('algorithm must be either RF or XGB')
        
    model = load_ml_model(train.output_path, algorithm)
    # the model is a multioutput calibrated classifier (aka made up of multiple classifier)
    # get a single example classifier
    idx = train.target_types.index(target_type)
    cv_fold = 0
    clf = model.estimators_[idx].calibrated_classifiers_[cv_fold].base_estimator
    if algorithm == 'RF': clf = clf.estimators_[0]
        
    # plot the tree
    fig, ax = plt.subplots(figsize=(150,20))
    if algorithm == 'RF':
        tree.plot_tree(clf, feature_names=get_clean_variable_names(train.X_train.columns), 
                       class_names=[f'Not {target_type}', target_type], fontsize=10, filled=True, ax=ax)
    else:
        plot_tree(clf, ax=ax)
    
def importance_plot(algorithm, target_types, save_dir, figsize, top=20, importance_by='feature', 
                    padding=None, colors=None):
    if importance_by not in {'feature', 'group'}: 
        raise ValueError('importance_by must be either "feature" or "group"')
    if padding is None: padding = {}
    # NOTE: run `python scripts/perm_importance.py` in the command line before running this function
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.3)
    N = len(target_types)
    if colors is None: colors = [None,]*N
    
    df = pd.read_csv(f'{save_dir}/perm_importance/{algorithm}_{importance_by}_importance.csv')
    df = df.set_index('index')
    df.index = get_clean_variable_names(df.index)
    
    summary = pd.DataFrame()
    for idx, target_type in tqdm.tqdm(enumerate(target_types)):
        feature_importances = df[target_type]
        feature_importances = feature_importances.sort_values(ascending=False)
        feature_importances = feature_importances[0:top] # get the top important features
        summary[target_type] = [f'{feature} ({importance})' for feature, importance in 
                                feature_importances.round(4).items()]
        ax = fig.add_subplot(N,1,idx+1)
        ax.barh(feature_importances.index, feature_importances.values, color=colors[idx])
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance Score')
        remove_top_right_axis(ax)
        fig.savefig(f'{save_dir}/figures/important_{importance_by}s/{algorithm}_{target_type}.jpg', 
                    bbox_inches=get_bbox(ax, fig, **padding), dpi=300) 
        ax.set_title(target_type) # set title AFTER saving individual figures
    plt.savefig(f'{save_dir}/figures/important_{importance_by}s/{algorithm}.jpg', bbox_inches='tight', dpi=300)
    plt.show()
    summary.index += 1
    return summary
    
def subgroup_performance_plot(df, target_type='ACU', subgroups=None, 
                              padding=None, figsize=(16,24), save=False, save_dir=None):
    if save and save_dir is None: 
        raise ValueError('Please provide save_dir if you want to save figures')
    if subgroups is None: 
        subgroups = df.index.levels[0]
    if padding is None: 
        padding = {}
    df = df.loc[subgroups]
    df.index = df.index.remove_unused_levels()
    
    def get_score(subgroup_name, metric):
        tmp = df.loc[subgroup_name, (target_type, metric)]
        tmp = tmp.astype(str).str.split('(')
        return tmp.str[0].astype(float)
    
    # Get the bar names
    bar_names = df.index.levels[1]
    bar_names = bar_names.str.split(pat='(').str[0]
    bar_names = bar_names.str.strip().str.title()
    bar_names = bar_names.tolist()
    bar_names[0] = df.index.levels[0][0] # Entire Test Cohort

    # Bar plot
    start_pos = 0
    x_bar_pos = []
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=figsize)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.6)
    metrics = df.columns.levels[1].tolist()
    for idx, subgroup_name in enumerate(df.index.levels[0]):
        auroc_scores = get_score(subgroup_name, metrics[0]) # AUROC Score
        bar_pos = np.arange(start_pos, start_pos+len(auroc_scores)).tolist()
        x_bar_pos += bar_pos
        start_pos += len(auroc_scores) + 1
        axes[0].bar(bar_pos, auroc_scores, label=subgroup_name, width=0.8)
   
        auprc_scores = get_score(subgroup_name, metrics[1]) # AUPRC Score
        axes[1].bar(bar_pos, auprc_scores, label=subgroup_name, width=0.8)
        
        ppv_scores = df.loc[subgroup_name, (target_type, metrics[2])] # PPV
        axes[2].bar(bar_pos, ppv_scores, label=subgroup_name, width=0.8)
        
        sensitivity_scores = df.loc[subgroup_name, (target_type, metrics[3])] # Sensitivity
        axes[3].bar(bar_pos, sensitivity_scores, label=subgroup_name, width=0.8)
        
        event_rate = df.loc[subgroup_name, (target_type, metrics[4])] # Event Rate
        axes[4].bar(bar_pos, event_rate, label=subgroup_name, width=0.8)
        
        if idx == 0:
            param = {'color': 'black', 'linestyle': '--'}
            for i, scores in enumerate([auroc_scores, auprc_scores, ppv_scores, sensitivity_scores, event_rate]):
                axes[i].axhline(y=scores[0], **param)
        
    for i, metric in enumerate(metrics):
        remove_top_right_axis(axes[i])
        axes[i].set_xticks(x_bar_pos) # adjust x-axis position of bars # Alternative way: plt.xticks(x_bar_pos, bar_names, rotation=45)
        axes[i].set_xticklabels(bar_names, rotation=45)
        axes[i].set_ylabel(metric)
        axes[i].legend(bbox_to_anchor=(1,0), loc='lower left', frameon=False)
        if save: fig.savefig(f'{save_dir}/subgroup_performance/{target_type}_{metric}.jpg', 
                             bbox_inches=get_bbox(axes[i], fig, **padding), dpi=300) 
    if save: 
        plt.savefig(f'{save_dir}/subgroup_performance/{target_type}.jpg', bbox_inches='tight', dpi=300)
    plt.show()
