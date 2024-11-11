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
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', '../../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from collections import defaultdict
import itertools

from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import colorcet as cc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
pd.set_option('display.precision', 3)
import seaborn as sns
import statsmodels.api as sm

from src import logger
from src.config import root_path, symp_folder, symptom_cols, observation_change_cols, DATE
from src.evaluate import EvaluateClf, CLF_SCORE_FUNCS
from src.prep_data import PrepData
from src.summarize import data_description_summary, feature_summary
from src.train import Trainer
from src.utility import (
    initialize_folders, load_pickle, save_pickle, make_log_msg,
    get_nunique_categories, get_nmissing, 
    replace_rare_categories
)
from src.visualize import remove_top_right_axis


# In[3]:


# config
processes = 64
target_keyword = 'target_'
main_dir = f'{root_path}/projects/{symp_folder}'
output_path = f'{main_dir}/models'
initialize_folders(output_path, extra_folders=['figures/output'])


# # Prepare Data for Model Training

# In[4]:


class PrepDataSYMP(PrepData):
    def __init__(self, scoring_map=None, **kwargs):
        super().__init__(**kwargs)
        self.main_dir = f'{root_path}/projects/{symp_folder}'
        self.symp_cols = [col for col in symptom_cols
                          if col not in ['ecog_grade', 'prfs_grade']]
        self.targ_cols = [f'{self.target_keyword}{col}' for col in self.symp_cols]
        # scoring criteria for assessing symptom deterioration
        if scoring_map is None: scoring_map = {col: 3 for col in self.symp_cols}
        self.scoring_map = scoring_map

    def get_data(self, verbose=False, **kwargs):
        df = self.load_data()
        df = self.prepare_targets(df, verbose=verbose)
        df = self.prepare_features(df, verbose=verbose, **kwargs)
        return df
    
    def prepare_targets(self, df, verbose=False):
        # remove target ECOG grade and target PRFS grade
        df = df.drop(columns=[f'{self.target_keyword}ecog_grade',
                              f'{self.target_keyword}prfs_grade'])
        
        # compute symptom score change
        for targ_col, base_col in zip(self.targ_cols, self.symp_cols):
            df[f'{targ_col}_change'] = df[targ_col] - df[base_col]
        
        # remove samples without any targets
        mask = ~df[pd.Index(self.targ_cols) + '_change'].isnull().all(axis=1)
        if verbose:
            context = ' without any target symptom score change'
            logger.info(make_log_msg(df, mask, context=context))
        df = df[mask]

        # convert target from symptom score change to symptom deterioration
        # label is positive if symptom deteriorates (score increases) by X points
        # (1 = positive, 0 = negative, -1 = missing/exclude)
        for targ_col, base_col in zip(self.targ_cols, self.symp_cols):
            pt = self.scoring_map[base_col]
            continuous_targ_col = f'{targ_col}_change'
            discrete_targ_col = f'{targ_col}_{pt}_change'
            missing_mask = df[continuous_targ_col].isnull()
            df[discrete_targ_col] = (df[continuous_targ_col] >= pt).astype(int)
            df.loc[missing_mask, discrete_targ_col] = -1

            # if baseline score is already high, we exclude them
            df.loc[df[base_col] > 10 - pt, discrete_targ_col] = -1

            # making sure if baseline is missing, target is always missing
            assert all(df.loc[df[base_col].isnull(), discrete_targ_col] == -1)

        return df
    
    def split_and_transform_data(
        self,
        data,
        remove_immediate_events=False,
        split_date=None,
        verbose=True,
        **kwargs
    ):
        """Split data into training, validation, and test sets based on patient
        ids (and optionally split temporally based on patient first visit dates).

        Optionally transform (one-hot encode, clip, impute, normalize) the data.

        Args:
            split_date (str): Date on which to split the data temporally.
                String format: 'YYYY-MM-DD' (e.g. 2017-06-30).
                Train/val set (aka development cohort) will contain all patients
                whose first chemo session date was on or before split_date.
                Test set (aka test cohort) will contain all patients whose first
                chemo session date was after split_date.
                If None, data won't be split temporally.
            **kwargs: keyword arguments fed into PrepData.transform_data
        """
        train_data, valid_data, test_data, external_data = self.make_splits(
            data, split_date=split_date, verbose=verbose
        )

        if remove_immediate_events:
            # Remove sessions where event occured immediately afterwards on the
            # train and valid set ONLY
            train_data = self.remove_immediate_events(train_data, verbose=verbose)
            valid_data = self.remove_immediate_events(valid_data, verbose=verbose)

        # IMPORTANT: always make sure train data is done first for one-hot 
        # encoding, clipping, imputing, scaling
        train_data = self.transform_data(
            train_data, data_name='training', verbose=verbose, **kwargs
        )
        valid_data = self.transform_data(
            valid_data, data_name='validation',verbose=verbose, **kwargs
        )
        test_data = self.transform_data(
            test_data, data_name='testing', verbose=verbose, **kwargs
        )
        external_data = self.transform_data(
            external_data, data_name='external', verbose=verbose, **kwargs
        )
            
        # create a split column and combine the data for convenienceR
        train_data[['cohort', 'split']] = ['Development', 'Train']
        valid_data[['cohort', 'split']] = ['Development', 'Valid']
        test_data[['cohort', 'split']] = 'Test'
        external_data[['cohort', 'split']] = 'External'
        data = pd.concat([train_data, valid_data, test_data, external_data])
        self.event_dates = self.event_dates.loc[data.index]

        # split into input features, output labels, and tags
        tag_cols = ['ikn', 'cohort', 'split', 'visit_hospital_number']
        cols = data.columns.drop(tag_cols)
        mask = cols.str.contains(self.target_keyword)
        target_cols, feature_cols = cols[mask], cols[~mask]
        
        X = data[feature_cols]
        Y = self.convert_labels(data[target_cols].copy())
        tag = data[tag_cols]
        
        if verbose:
            df = pd.DataFrame({
                'Number of sessions': tag.groupby('split').apply(len), 
                'Number of patients': tag.groupby('split')['ikn'].nunique()}
            ).T
            df['Total'] = df.sum(axis=1)
            logger.info(f'\n{df.to_string()}')
        
        return X, Y, tag
    
    def make_splits(self, data, split_date=None, verbose=True):
        # create internal and external cohort based on hospital site
        # PMH = princess margaret hospital
        treated_at_PMH_mask = data['visit_hospital_number'] == '4406'
        internal_data = data[treated_at_PMH_mask].copy()
        external_data = data[~treated_at_PMH_mask].copy()

        # remove patients from external cohort seen in internal cohort
        mask = ~external_data['ikn'].isin(internal_data['ikn'])
        if verbose:
            context = ' in the external cohort who were seen in internal cohort.'
            logger.info(make_log_msg(external_data, mask, context=context))
        external_data = external_data[mask]

        # readjust first visit date to its respective cohort
        for df in [internal_data, external_data]:
            first_visit_date = self.event_dates.loc[df.index, DATE].groupby(df['ikn']).min()
            self.event_dates.loc[df.index, f'first_{DATE}'] = df['ikn'].map(first_visit_date)

        # create training, validation, testing set from the internal data
        # split the development and testing set temporally based on patients first visit date
        train_data, valid_data, test_data = self.create_splits(internal_data, split_date=split_date, verbose=verbose)

        # align time frame of external data with the testing set (to simulate prospective deployment)
        _, external_data = self.create_cohort(external_data, split_date=split_date, verbose=verbose)

        return train_data, valid_data, test_data, external_data
    
def remove_immediate_events(self, df, verbose=False):
    event_dates = self.event_dates.loc[df.index]
    n_events = []
    for targ_col, base_col in zip(self.targ_cols, self.symp_cols):
        pt = self.scoring_map[base_col]
        event_date_col = f'{targ_col}_survey_date'
        discrete_targ_col = f'{targ_col}_{pt}pt_change'

        days_until_event = event_dates[event_date_col] - event_dates[DATE]
        immmediate_mask = days_until_event < pd.Timedelta(days=2)
        occured_mask = df[discrete_targ_col] == 1
        mask = immmediate_mask & occured_mask

        # if immediate event occured, set the target to -1 so it is removed
        # at a later stage
        df.loc[mask, discrete_targ_col] = -1

        n_events.append(sum(mask))

    if verbose:
        logger.info(f'About {min(n_events)}-{max(n_events)} sessions had an '
                    f'event occur in less than 2 days')
        

prep = PrepDataSYMP(target_keyword=target_keyword)
# only keep Lung, Head & Neck, and Gasto-intestinal patients (to align with AIM2REDUCE data)
cancer_topog_cds = [f'C{i:02}' for i in range(27)] + ['C30', 'C31', 'C32', 'C34', 'C37', 'C38']
model_data = prep.get_data(missing_thresh=80, cancer_sites=cancer_topog_cds, verbose=True)
X, Y, tag = prep.split_and_transform_data(model_data, remove_immediate_events=True, split_date='2017-09-30', verbose=True)
model_data = model_data.loc[tag.index]
# clean up Y
Y = Y[[col for col in Y.columns if col.endswith('pt_change')]]


# Convenience variables
train_mask = tag['split'] == 'Train'
valid_mask = tag['split'] == 'Valid'
test_mask = tag['split'] == 'Test'
external_mask = tag['split'] == 'External'
X_train, X_valid, X_test, X_external = X[train_mask], X[valid_mask], X[test_mask], X[external_mask]
Y_train, Y_valid, Y_test, Y_external = Y[train_mask], Y[valid_mask], Y[test_mask], Y[external_mask]


# # Describe Data

# In[]:

# number of cancer centers larger than the internal cancer center
internal_volume = len(X_test)
external_volume = tag[external_mask].groupby('visit_hospital_number').apply(len)
(external_volume > internal_volume).value_counts()


# In[]:


from collections import defaultdict
def get_label_distributions(Y, tag, with_respect_to='sessions'):
    if with_respect_to == 'patients':
        dists = {}
        for split, group in Y.groupby(tag['split']):
            count = defaultdict(dict)
            ikn = tag.loc[group.index, 'ikn']
            for target, label in group.items():
                count[1][target] = ikn[label == 1].nunique()
                count[0][target] = ikn.nunique() - count[1][target]
            dists[split] = pd.DataFrame(count).T
    elif with_respect_to == 'sessions':
        dists = {split: group.apply(pd.value_counts) for split, group in Y.groupby(tag['split'])}
    dists['Total'] = dists['Train'] + dists['Valid'] + dists['Test']
    dists = pd.concat(dists).T
    return dists
get_label_distributions(Y, tag, with_respect_to='sessions')


# In[]:


get_label_distributions(Y, tag, with_respect_to='patients')


# ## Study Characteristics

# In[]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location',
]
data_description_summary(
    model_data, Y, tag, save_dir=f'{output_path}/tables', partition_method='cohort', subgroups=subgroups
)


# ## Feature Characteristics

# In[]:


x = prep.ohe.encode(model_data.drop(columns=['visit_hospital_number']), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
x = x[train_mask].drop(columns=['ikn'] + Y.columns.tolist())
feature_summary(
    x, save_dir=f'{output_path}/tables', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# # Train System

# In[]:


targets = Y.columns
algs = ['LGBM']
for alg, target in itertools.product(algs, targets):
    print(f'Training {alg} for predicting {target}')
    mask = Y[target] != -1
    trainer = Trainer(X[mask], Y.loc[mask, [target]], tag[mask], output_path)
    trainer.run(
        bayesopt=True, train=True, calibrate=False, save_preds=True, pred_filename=f'{alg}_{target}_preds', algs=[alg],
        train_kwargs=dict(filename=f'{alg}_{target}'), # clip_gradients=True
        bayes_kwargs=dict(filename=f'{alg}_{target}_params')
    )


# # Evaluate System

# In[]:


targets = Y.columns
algs = ['LGBM']
results = []

# confidence interval
compute_ci = False
if compute_ci: ci = False

for alg in algs:
    result = []
    for target in targets:
        preds = load_pickle(f'{output_path}/preds', f'{alg}_{target}_preds')
        labels = {split: Y.loc[pred.index, [target]] for split, pred in preds[alg].items()}
        evaluator = EvaluateClf(output_path, preds, labels)

        if compute_ci:
            # a quick hack to retain the confidence interval results
            if ci is not None: evaluator.ci.bs_scores = ci
            scores = evaluator.get_evaluation_scores(display_ci=True, save_score=False, splits=['Valid', 'Test', 'External'])
        else:
            scores = evaluator.get_evaluation_scores(display_ci=True, load_ci=True, save_score=False, splits=['Valid', 'Test', 'External'])
        
        result.append(scores)
    results.append(pd.concat(result, axis=1))

if compute_ci: evaluator.ci.save_bootstrapped_scores()
results = pd.concat(results)
results.to_csv(f'{output_path}/tables/evaluation_scores.csv')


# In[]:


results['Valid']


# In[]:


results['Test']


# In[]:


results['External']


# In[]:


# get the mean score across targets
for alg in ['LGBM']:
    for split in ['Valid', 'Test', 'External']:
        for metric, row in result[split].loc[alg].iterrows():
            mean_score = []
            for score in row:
                val, lower, upper = score.replace('(', '').replace(')', '').replace('-', '').split(' ')
                mean_score.append([val, lower, upper])
            mean, lower, upper = np.array(mean_score).astype(float).mean(axis=0)
            print(f'{alg} - {split} Mean {metric} across all targets: {mean:.3f} ({lower:.3f}-{upper:.3f})')


# # Score by Cancer Center

# In[]:


# align with Aim2Reduce colors and labels
label_map = {
    'target_anxiety_3pt_change': 'Anxiety',
    'target_depression_3pt_change': 'Depression',
    'target_drowsiness_3pt_change': 'Drowsiness',
    'target_lack_of_appetite_3pt_change': 'Appetite',
    'target_nausea_3pt_change': 'Nausea',
    'target_pain_3pt_change': 'Pain',
    'target_shortness_of_breath_3pt_change': 'Dyspnea',
    'target_tiredness_3pt_change': 'Fatigue',
    'target_wellbeing_3pt_change': 'Well-being',
}
legend_order = ['Nausea', 'Appetite', 'Pain', 'Dyspnea', 'Fatigue', 'Drowsiness', 'Depression', 'Anxiety', 'Well-being']
colors = {symp: cc.glasbey_light[i] for i, symp in enumerate(legend_order)}


# In[]:


# get the results
def get_cancer_center_analysis_data(df, label, output_path, alg='LGBM', label_map=None, bs_var=False):
    """
    Args:
        bs_var (bool): If True, compute the bootstrapped variance for each cancer center metric
    """
    if label_map is None: label_map = {targ: targ for targ in label.columns}

    result = []
    for target in tqdm(label.columns):
        # load prediction
        preds = load_pickle(f'{output_path}/preds', f'{alg}_{target}_preds')
        pred = pd.concat([preds[alg]['Test'], preds[alg]['External']])

        # setup required data
        meta = df.loc[pred.index].copy()
        # N = sum(meta.goupby('visit_hospital_number')['ikn'].nunique() < 40)
        # print(f'Collapsing {N} cancer centers with less than 40 patients into Other')
        meta = replace_rare_categories(meta, catcols=['visit_hospital_number'], n=40)

        # compute center-level characteristics and metrics
        for cancer_center, group in meta.groupby('visit_hospital_number'):
            y_true = label.loc[group.index, target]
            y_pred = pred.loc[group.index, target]
            center_data = {
                'Cancer Center': cancer_center,
                'Targets': label_map[target],
                'Number of Patients': group['ikn'].nunique(),
                'Number of Sessions': len(group),
                'Urban Proportion': (~group['rural']).mean(),
                'Recent Immigrant Proportion': (group['neighborhood_income_quintiles'] == 1).mean(),
                'Target Rate': y_true.mean()
            }
            for metric, score_func in CLF_SCORE_FUNCS.items():
                center_data[metric] = score_func(y_true, y_pred)

            if bs_var:
                # compute bootstrapped variance
                bootstrap_scores = compute_bootstrap_scores(y_true, y_pred, CLF_SCORE_FUNCS, n_bootstraps=1000, processes=4)
                for metric, var in bootstrap_scores.items(): center_data[f'{metric}_variance'] = var
            
            result.append(center_data)

        return pd.DataFrame(result)


def get_baseline_missingness(filepath, result, label, tag, label_map=None, convert_to_targ=None):
    """Insert baseline missingness proportion to the result

    Args:
        filepath (str): path to the file containing the missingness info for each cancer center
        result (pd.DataFrame): table to insert the missingness info to
        label (pd.DataFrame): table of target labels
        tag (pd.DataFrame): table containing the hospital numbers for each visit
    """
    if label_map is None: label_map = {targ: targ for targ in label.columns}
    if convert_to_targ is None: convert_to_targ = lambda base_col: f'target_{base_col}_3pt_change'

    miss_df = pd.read_csv(filepath, index_col=[0], dtype={'visit_hospital_number': str})
    miss_df = miss_df[miss_df.index.isin(tag['visit_hospital_number'])] # keep only centers seen in this cohort
    nsessions = miss_df.pop('nsessions')
    for base_col, missing_ratio in miss_df.items():
        target = convert_to_targ(base_col)
        # retrieve which cancer centers were grouped to 'Other'
        mask = label[target] != -1
        meta = replace_rare_categories(tag[mask].copy(), catcols=['visit_hospital_number'], n=40)
        mask = miss_df.index.isin(meta['visit_hospital_number'])
        # combine and recompute the missingness for the 'Other' cancer centers
        total = sum(nsessions[~mask])
        nmissing = sum(nsessions[~mask] * missing_ratio[~mask])
        # remove the cancer centers that belong to 'Other'
        missing_ratio = missing_ratio[mask]
        missing_ratio['Other'] = nmissing / total
        # add to results
        mask = result['Targets'] == label_map[target]
        result.loc[mask, 'Missing Baseline Proportion'] = result.loc[mask, 'Cancer Center'].map(missing_ratio)

    return result


# In[]:


cols = ['ikn', 'visit_hospital_number', 'rural', 'recent_immigrants', 'neighborhood_income_quintile']
result = get_cancer_center_analysis_data(model_data[cols], Y, output_path, alg='LGBM', label_map=label_map, bs_var=True)

# include baseline ESAS missingness to results
filepath = f'{main_dir}/data/baseline_missingness_per_cancer_center.csv'
result = get_baseline_missingness(filepath, result, Y, tag, label_map=label_map)

result.to_csv(f'{output_path}/tables/cancer_center.csv')


# In[]:


result = pd.read_csv(f'{output_path}/tables/cancer_center.csv')

# split into internal validation and external validation results
test_cancer_center = tag.query('split == "Test"')['visit_hospital_number'].unique()
assert len(test_cancer_center) == 1
mask = result['Cancer Center'] == test_cancer_center[0]
internal_test_result, external_validation_result = result[mask].set_index('Targets'), result[~mask]


# In[]:


# see number of patients for each cancer center for each symptom
df = external_validation_result.pivot(index='Cancer Center', columns='Targets', values='Number of Patients')
df['MEAN'] = df.mean(axis=1).round(2)
print(f'Number of cancer centers = {len(df) - 1} + (36 to 37 small cancer centers collapsed in Other)')
df.sort_values(by='MEAN')


# In[]:


# see AUROC range for each symptom
auroc_range = {'range': {}, 'interquartile_range': {}}
for symp, auroc in external_validation_result.groupby('Targets')['AUROC']:
    quantiles = auroc.quantile([0.25, 0.75])
    auroc_range['range'][symp] = f'{auroc.min():.3f"}-{auroc.max():.3f}'
    auroc_range['interquartile_range'][symp] = f'{quantiles.min():.3f"}-{quantiles.max():.3f}'
auroc_range = pd.DataFrame(auroc_range).loc[legend_order]
auroc_range.to_csv(f'{output_path}/tables/auroc_range.csv')
auroc_range


# In[]:


# correlation heatmap
corr_map, p_val_map = defaultdict(dict), defaultdict(dict)
characteristics = [
    'Number of Sessions', 'Urban Proportion', 'Recent Immigrant Proportion', 'Low Income Proportion',
    'Missing Baseline Proportion', 'Target Rate'
]
metric = 'AUROC'
mask = external_validation_result['Cancer Center'] != 'Other' # do not include the other small centers when computing correlation
for target, group in external_validation_result[mask].groupby('Targets'):
    for col in characteristics:
        corr, p_value = spearmanr(group[col], group[metric])
        corr_map[target][col] = p_value
corr = pd.DataFrame(corr_map)[legend_order].T
p_val = pd.DataFrame(p_val_map)[legend_order].T

# adjust the p-value - Berfonni method
p_val *= np.prod(p_val.shape)

fig, ax = plt.subplot(figsize=(8,6))
sns.heatmap(
    corr, center=0, annot=True, annot_kws={'fontsize': 9, 'color': 'black'}, fmt='.3f', ax=ax,
    cmap='coolwarm', vmin=-1, vmax=1
)
ax.set_xticklabels([col.repalce(' ', '\n') for col in corr.columns], rotation=0)
# make significant cells bold and add a star
sig_mask = p_val.to_numpy().flatten() < 0.05
for i in np.where(sig_mask)[0]: ax.texts[i].set(weight='bold', text=f'*{ax.texts[i]._text}') # style='italic'
plt.savefig(f'{output_path}/figures/output/corr_heatmap.jpg', bbox_inches='tight', dpi=300)


# In[]:


# reformat R-computed heterogenity
f = open(f'{output_path}/logs/hetero.txt')
hetero = f.read()
hetero = hetero.split('"####################################################"') # split by dividers
hetero = hetero[1:] # remove the first item
result = {}
for i in range(0, len(hetero), 2):
    target = hetero[i]
    output = hetero[i+1]

    target = target.strip('\n[1]" ')
    i_squared, estimates = output.split('Model Results:\n\n')
    estimates = estimates.split('\n')
    cols, vals = estimates[0].strip(), estimates[1].strip('* ')

    result[target] = pd.Series(data=vals.split(), index=cols.split())

    for line in i_squared.strip().split('\n'):
        if 'I^2' in line or 'H^2' in line:
            col, val = line.split(': ')
            result[target][col] = val
        elif 'Q' in line:
            Q, p = line.split(',')
            col, val = Q.replace('Q(df = ', 'Q(df=').split(' = ')
            result[target][col] = val
            result[target][f'{col} p-value'] = p.replace('p-val', '')
hetero = pd.DataFrame(result).T
hetero.to_csv(f'{output_path}/tables/hetero.csv')
hetero


# In[]:


# violin plot
fig, ax = plt.subplots(figsize=(12,6))
sns.violinplot(
    x='Targets', y='AUROC', data=external_validation_result, ax=ax, order=legend_order, palette=colors, inner=None
)
plt.setp(ax.collections, alpha=.3)

# create a horizontal dashed line for PM and other cancer centers for each target
xmin, line_length = 0, 1 / len(targets)
adj_int = [0.035, 0.0275, 0.030, 0.0275, 0.0175, 0.0175, 0.0275, 0.030, 0.0225]
adj_ext = [0.030, 0.0275, 0.0325, 0.020, 0.0125, 0.0175, 0.025, 0.025, 0.0175]
for i, symp in enumerate(legend_order):
    # PM - internal
    auroc = internal_test_result.loc[symp, 'AUROC']
    ax.axhline(
        auroc, xmin=xmin+adj_int[i], xmax=xmin+line_length-adj_int[i], color='black', linewidth=1.2, linestyle='dashed',
        label=('Princess Margaret Cancer Center' if i == 0 else None)
    )
    # other cancer centers - external
    auroc = float(hetero.loc[symp, 'estimate'])
    ax.axhline(
        auroc, xmin=xmin+adj_ext[i], xmax=xmin+line_length-adj_ext[i], color='black', linewidth=1.2, linestyle='solid',
        label=('Other Cancer Centers' if i == 0 else None)
    )
    xmin += line_length
ax.legend(frameon=False, loc='lower right', fontsize=9)

sns.stripplot(
    x='Targets', y='AUROC', hue='Targets', data=external_validation_result,
    ax=ax, order=legend_order, palette=colors, zorder=1, legend=False
)
ax.set(xlabel=None, ylim=(0.3-0.01, 0.88+0.01))
ax.grid(axis='y', linewidth=0.2)
remove_top_right_axis(ax)
plt.savefig(f'{output_path}/figures/output/violin_plot.jpg', bbox_inches='tight', dpi=300)


# In[]:


def cancer_center_scatter_plot(
    ax,
    df,
    x='Number of Sessions',
    y='AUROC',
    color=None,
    dot_size=30,
    refs=None, 
    p_adj=None
):
    """Plot the metric and attribute across different cancer centers
    """
    if regs is None: refs = []

    sns.scatterplot(data=df, x=x, y=y, ax=ax, color=color, s=dot_size)
    ax.set_ylim(df[y].min()-0.1, df[y].max()+0.05)

    # create a reference line
    for ref in refs:
        if reg['type'] == 'horizontal_line': ax.axhline(**refs['kwargs'])
        elif reg['type'] == 'vertical_line': ax.axvline(**refs['kwargs'])
        elif reg['type'] == 'point': sns.scatterplot(ax=ax, **refs['kwargs'])
        else: raise NotSupportedError(f'Referece type {ref_type} is not supported yet')

    # display the spearman correlation coefficient
    mask = df['Cancer Center'] != 'Other' # do not include the other small centers when computing correlation
    corr, p_value = spearmanr(df.loc[mask, x], df.loc[mask, y])
    if p_adj is not None:
        p_value = min(1, p_value * p_adj)
    if p_value > 0.99:
        p_value = '>0.99'
    elif p_value < 0.001:
        p_value = '<0.001'
    else:
        p_value = f'={p_value:.3f}' if p_value < 0.01 else f'={p_value:.2f}'
    ax.text(0.02, 0.03, f'Spearman Coefficient={corr:.3f}, P{p_value}', transform=ax.transAxes)

    remove_top_right_axis(ax)
    ax.legend(loc='lower right', frameon=False)


def cancer_center_scatter_plots(
    axes,
    internal_test_df,
    external_validation_df,
    metric='AUROC',
    attribute='Number of Sessions',
    colors=None,
    p_adj=None
):
    """Plot the metric and attribute across different cancer centers for each target

    PM - Princess Margaret cancer center
    no_PM - All other cancer centers
    """
    if colors is None: colors = {}
    for i, (targ, group) in enumerate(external_validation_df.groupby('Targets')):
        color = colors.get(targ, None)
        # build the arguments for referense lines and points
        PM_score = internal_test_df.loc[targ, metric]
        non_PM_mean_score = group[metric].mean()
        refs = [
            {
                'type': 'horizontal_line',
                'kwargs': {
                    'y': non_PM_mean_score,
                    'label': f"Non-PM\nMean {metric} = {non_PM_mean_score:.3f}",
                    'color': color,
                    'linestyle': 'dashed'
                }
            },
            {
                'type': 'point',
                'kwargs': {
                    'data': internal_test_df.loc[[targ]],
                    'x': attribute,
                    'y': metric,
                    'label': f"PM\n{metric} = {PM_score:.3f}",
                    'color': 'black',
                    'marker': '*',
                    's': 90
                }
            },
            {
                'type': 'point',
                'kwargs': {
                    'data': group[group['Cancer Center'] == 'Other'],
                    'x': attribute,
                    'y': metric,
                    'label': "Other Small Centers",
                    'color': 'black',
                    'marker': 'D',
                    's': 20
                }
            },
        ]

        # create the scatter plot
        cancer_center_scatter_plot(axes[i], group, ax=attribute, y=metric, color=color, refs=refs, p_adj=p_adj)
        axes[i].set_title(targ)


def cancer_center_performance_comparison_plot(
    internal_test_df,
    external_validation_df,
    metrics=None, 
    attributes=None,
    colors=None,
    legend_order=None
):
    """Plot model performance across different cancer centers, showcasing their different attributes
    """
    if metrics is None: metrics = ['AUROC', 'AUPRC']
    if attributes is None: attributes = ['Number of Sessions']

    sns.set_context('paper', rc={'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 9})
    nrows, ncols = external_validation_df['Targets'].nunique() + 1, len(metrics)

    p_adj = len(attributes) * len(internal_test_df)
    for attribute in attributes:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 6*nrows))
        if ncols == 1: axes = axes[..., np.newaxis]
        for i, metric in enumerate(metrics):
            # plot for single targets
            cancer_center_scatter_plots(
                axes[:, i], internal_test_df, external_validation_df, metric=metric, attribute=attribute,
                colors=colors, p_adj=p_adj
            )

            # plot all targets
            ax = axes[-1, i]
            sns.scatterplot(
                data=external_validation_df, x=attribute, y=metric, hue='Targets', palette=colors, alpha=0.5, ax=ax
            )
            if legend_order is not None:
                handles, labels = ax.get_legend_handles_labels()
                legend_map = dict(zip(labels, handles))
                ax.legend([legend_map[symp] for symp in legend_order], legend_order, loc='lower right', frameon=False)
            remove_top_right_axis(ax)

        filename = f'cancer_center_plot_{attribute.replace(" ", "_").lower()}'
        plt.savefig(f'{output_path}/figures/output/{filename}.jpg', bbox_inches='tight', dpi=300)


# In[]:


attributes = [
    'Number of Sessions', 'Urban Proportion', 'Recent Immigrant Proportion', 'Low Income Proportion',
    'Missing Baseline Proportion', 'Target Rate'
]
cancer_center_performance_comparison_plot(
    internal_test_result, external_validation_result, metrics=['AUROC'], attributes=attributes, colors=colors,
    legend_order=legend_order
)