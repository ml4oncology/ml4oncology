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

# # Supplementary Analysis
# From reveiwer's comments and recommendation

# In[1]:


get_ipython().run_line_magic('cd', '../../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.config import root_path, death_folder, split_date, DATE, variable_groupings_by_keyword
from src.evaluate import EvaluateClf
from src.impact import get_pccs_analysis_data
from src.prep_data import PrepDataEDHD
from src.preprocess import compute_chemo_cycle
from src.summarize import CharacteristicsSummary
from src.train import LASSOTrainer, Ensembler, Trainer
from src.utility import (
    load_pickle, save_pickle, initialize_folders, 
    equal_rate_pred_thresh, 
    get_clean_variable_name,
    get_clean_variable_names,
    make_log_msg,
    month_diff,
)
from src.visualize import remove_top_right_axis


# In[3]:


main_dir = f'{root_path}/projects/{death_folder}'
output_path = f'{main_dir}/models'
target_keyword = 'Mortality'
date_cols = ['visit_date', 'death_date', 'first_PCCS_date', 'first_visit_date']


# In[4]:


prep = PrepDataEDHD(adverse_event='death', target_keyword=target_keyword)
model_data = prep.get_data(missing_thresh=80, treatment_intents=['P'], verbose=False)
X, Y, tag = prep.split_and_transform_data(
    model_data, split_date=split_date, remove_immediate_events=True, ohe_kwargs={'verbose': False}
)
model_data = model_data.loc[tag.index] # remove sessions in model_data that were excluded during split_and_transform

train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]
Y_train, Y_valid, Y_test = Y[train_mask], Y[valid_mask], Y[test_mask]


# In[5]:


# combine predictions
preds = {} # load_pickle(f'{output_path}/preds', 'all_pred')
for alg in ['LR', 'XGB', 'RF', 'NN', 'RNN']:
    preds.update(load_pickle(f'{output_path}/preds', f'{alg}_preds'))

ensembler = Ensembler(X, Y, tag, output_path, preds)
ensembler.run(bayesopt=False, calibrate=False)
preds, labels = ensembler.preds.copy(), ensembler.labels.copy()
preds.update(load_pickle(f'{output_path}/preds', 'LASSO_preds'))


# In[6]:


evaluator = EvaluateClf(output_path, preds, labels)
evaluator.orig_data = model_data
year_mortality_thresh = equal_rate_pred_thresh(
    evaluator, prep.event_dates, split='Test', alg='ENS', target_event='365d Mortality'
)


# In[7]:


def get_data(split='Test', alg='ENS', target_event='365d Mortality', pred_thresh=None):
    pred = preds[alg][split][target_event]
    label = labels[split][target_event].astype(int)
    time = prep.event_dates.loc[pred.index, DATE]
    patient_id = tag.loc[pred.index, 'ikn']
    df = pd.concat([label, pred, time, patient_id], keys=['Label', 'Pred', 'Time', 'Subject'], axis=1)
    if pred_thresh is not None: df['Pred'] = df['Pred'] > pred_thresh 
    return df


# # First Treatment Per Patient Only

# In[9]:


df = model_data.reset_index().groupby('ikn').first()


# In[11]:


first_trt_preds = {}
for alg, values in preds.items():
    first_trt_preds[alg] = {split: pred[pred.index.isin(df['index'])] for split, pred in values.items()}
first_trt_labels = {split: label[label.index.isin(df['index'])] for split, label in labels.items()}


# In[13]:


"""
AUROC dropped by 0.08, AUPRC rose by 0.028
"""
evaluator = EvaluateClf(output_path, first_trt_preds, first_trt_labels)
evaluator.get_evaluation_scores(
    algs=['ENS'], target_events=['365d Mortality'], splits=['Test'], 
    display_ci=True, load_ci=False, save_ci=False, save_score=False
)


# # One Random Treatment Per Patient Only

# In[14]:


df = model_data.reset_index().groupby('ikn').sample(n=1, random_state=4)


# In[15]:


rand_trt_preds = {}
for alg, values in preds.items():
    rand_trt_preds[alg] = {split: pred[pred.index.isin(df['index'])] for split, pred in values.items()}
rand_trt_labels = {split: label[label.index.isin(df['index'])] for split, label in labels.items()}


# In[17]:


"""
AUROC increased by 0.022, AUPRC rose by 0.113
"""
evaluator = EvaluateClf(output_path, rand_trt_preds, rand_trt_labels)
evaluator.get_evaluation_scores(
    algs=['ENS'], target_events=['365d Mortality'], splits=['Test'], 
    display_ci=True, load_ci=False, save_ci=False, save_score=False
)


# # AUROC / AUPRC Over Time

# In[18]:


test_df = get_data(split='Test')
time = test_df['Time'] - prep.event_dates.loc[test_df.index, f'first_{DATE}']
time = (time.dt.days / 30).astype(int)
time.hist()
test_df['Time'] = time.clip(upper=time.quantile(0.99))


# In[21]:


from src.conf_int import ScoreConfidenceInterval
from src.evaluate import CLF_SCORE_FUNCS

ci = ScoreConfidenceInterval(output_path, CLF_SCORE_FUNCS)
ci.load_bootstrapped_scores(filename='bootstrapped_scores_over_time')

scores_over_time = {}
for time, group in test_df.groupby('Time'):
    Y_true, Y_pred = group['Label'], group['Pred']
    scores = ci.get_score_confidence_interval(
        Y_true, Y_pred, name=f'{time}_ENS_Test_365d Mortality', store=True, verbose=True
    )
    data = {'AUROC': roc_auc_score(Y_true, Y_pred), 'AUPRC': average_precision_score(Y_true, Y_pred)}
    for name, (lower, upper) in scores.items():
        data[f'{name}_lower'] = lower 
        data[f'{name}_upper'] = upper
    scores_over_time[time] = data
scores_over_time = pd.DataFrame(scores_over_time).T
# auroc_over_time = test_df.groupby('Time').apply(lambda g: roc_auc_score(g['Label'], g['Pred']))
# auprc_over_time = test_df.groupby('Time').apply(lambda g: average_precision_score(g['Label'], g['Pred']))

ci.save_bootstrapped_scores(filename='bootstrapped_scores_over_time')


# In[20]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
for i, name in enumerate(['AUROC', 'AUPRC']):
    axes[i].plot(scores_over_time[name])
    axes[i].fill_between(
        scores_over_time.index, scores_over_time[f'{name}_lower'], scores_over_time[f'{name}_upper'], alpha=0.2
    )
    axes[i].set(xticks=np.arange(0, 25, 3), xlabel='Months after First Treatment', ylabel=f'{name} Score')


# # Intra-patient Correlation Analysis

# In[22]:


test_df = get_data(split='Test', pred_thresh=year_mortality_thresh)
new_patient = ~test_df['Subject'].duplicated()
test_df['Time'] = compute_chemo_cycle(new_patient)
test_df['Time'].hist()

r_data = test_df.copy()
r_data.to_csv(f'{main_dir}/data/R_data.csv', index=False)
# cut it half and clip at 95th percentile to allow unstructured correlation to work in R
r_data2 = r_data.query('Time % 2 == 1').copy()
r_data2['Time'] = r_data2['Time'].clip(upper=r_data2['Time'].quantile(0.95))
r_data2.to_csv(f'{main_dir}/data/R_data2.csv', index=False)


# ## Marginal Model (Generalized Estimating Equation)

# In[14]:


def gee_fit(df, struct='Autoregressive', formula='Label ~ Pred + Time', maxiter=60, upper_clip=0.95):
    # Ref: https://towardsdatascience.com/an-introduction-to-generalized-estimating-equations-bc7dee570478
    fam = sm.families.Binomial()
    time = df['Time']
    time = time.clip(upper=time.quantile(upper_clip)) # YOU NEED TO CLIP OUTLIERS OR ELSE GEE THROWS A TANTRUM
    # NOTE: for exchangeable, you can try adjusting clip quantile and maxiter to get non-garbage numbers
    struct = {
        'Independence': sm.cov_struct.Independence(), 
        'Exchangeable': sm.cov_struct.Exchangeable(), 
        'Autoregressive': sm.cov_struct.Autoregressive(grid=True),
        'Unstructured': sm.cov_struct.Unstructured(),
    }[struct]
    model = sm.GEE.from_formula(
        formula=formula, groups='Subject', data=df, time=time, cov_struct=struct, family=fam
    )
    result = model.fit(maxiter=maxiter)
    
    return result, model

def gee_eval(result, model):
    print(result.summary())
    
    # odds ratio = ratio of the odds of true label in the presence of positive prediction and odds of true label in the
    # absence of positive prediction
    odds_ratio = np.exp(result.params['Pred[T.True]'])
    
    # R2 score is not appropriate for GEE. We can consider other goodness-of-fit measures such as QIC
    # QIC = quasi-information criterion (quasi-likelihood under the independence model information criterion)
    # Smaller QIC is "better"
    # qic, qicu = result.qic(model.estimate_scale())
    ql, qic, qicu = result.model.qic(result.params, model.estimate_scale(), result.cov_params())
    
    # CIC = correlation information critierion, the penalty term of QIC. Supposedly it's better than using QIC
    # We can get CIC by reverse engineering the QIC formula: qic = -2 * ql + 2 * cic
    cic = (qic + 2*ql) / 2
    return {'QIC': qic, 'CIC': cic, 'Alarm odds ratio': odds_ratio}


# In[20]:


get_ipython().run_cell_magic('time', '', "output = {}\nscores = {}\nfor struct in ['Independence', 'Exchangeable', 'Autoregressive', 'Unstructured']:\n    output[struct] = gee_fit(test_df, struct=struct, formula='Label ~ Pred + Time')\n    print('##########################################################################')\n    print(f'# {struct}')\n    print('##########################################################################')\n    scores[struct] = gee_eval(*output[struct])\n# save_pickle(output['Unstructured'], output_path, 'GEE')\n# On average the odds of event increased by Alarm Odds Ratio after an alarm\npd.DataFrame(scores).T")


# ## Conditional Model (Generalized Linear Mixed-Effects Models)

# In[26]:


def glmm_fit(df, formula):
    model = sm.genmod.BinomialBayesMixedGLM.from_formula(
        formula=formula, vc_formulas={'Subject': '0 + Subject'}, data=df
    )
    result = model.fit()
    return result, model


# In[27]:


# something about singular matrix error on np.linalg.inv(hess)
# output = glmm_fit(test_df, formula='Label ~ Pred + Time + (1 | Subject)')


# # SHAP
# Takes ages to compute. Can only compute SHAP for a subset of data

# In[56]:


from src.config import min_chemo_date


# In[57]:


min_date = pd.Timestamp(min_chemo_date)
visit_dates = prep.event_dates.loc[test_mask, 'visit_date']
test_ikns = tag.loc[test_mask, 'ikn']
data = pd.concat([X_test.astype(float), (visit_dates - min_date).dt.days.astype(int), test_ikns], axis=1)
sampled_ikns = np.random.choice(data['ikn'].unique(), size=1000)
mask = data['ikn'].isin(sampled_ikns)
data, bg_dist = data[mask], data[~mask] # use the latter half as background distribution
# data = data.sample(2000, random_state=42)
# data, bg_dist = data[:1000], data[1000:] # use the latter half as background distribution
drop_cols = ['ikn', 'visit_date']
bg_dist[drop_cols] = -1
idx = Y.columns.tolist().index('365d Mortality')


# In[10]:


lr_model = load_pickle(output_path, 'LR')
rf_model = load_pickle(output_path, 'RF')
xgb_model = load_pickle(output_path, 'XGB')
nn_model = load_pickle(output_path, 'NN')
rnn_model = load_pickle(output_path, 'RNN')
rnn_model.model.rnn.flatten_parameters()
ensemble_weights = load_pickle(f'{output_path}/best_params', 'ENS_params')
ensemble_weights = {alg: w for alg, w in ensemble_weights.items() if w > 0}
trainer = Trainer(X, Y, tag, output_path)


# In[11]:


def _rnn_predict(X):
    # Reformat to sequential data
    X = X.copy()
    N = len(X)
    # get the ikn and visit date for this sample row
    res = {}
    for col in ['ikn', 'visit_date']:
        mask = X[col] != -1
        val = X.loc[mask, col].unique()
        assert len(val) == 1
        res[col] = val[0]
    # get patient's historical data
    # NOTE: historical data is NOT permuted
    hist = data.query(f'ikn == {res["ikn"]} & visit_date < {res["visit_date"]}').copy()
    n = len(hist)
    hist = pd.concat([hist] * N) # repeat patient historical data for each sample row
    # set up new ikn for each sample row
    ikns = np.arange(0, N, 1) + res['ikn']
    hist['ikn'] = np.repeat(ikns, n)
    X['ikn'] = ikns
    # combine historical data and sample rows togethers
    X['visit_date'] = res['visit_date']
    X = pd.concat([X, hist]).sort_values(by=['ikn', 'visit_date'], ignore_index=True)

    # Get the RNN predictions
    trainer.ikns = X['ikn']
    trainer.labels['Test'] = pd.DataFrame(True, columns=Y.columns, index=X.index) # dummy variable
    X = X.drop(columns=drop_cols)
    trainer.datasets['Test'] = X # set the new input
    pred = trainer.predict(rnn_model, 'Test', 'RNN', calibrated=True)
    pred = pred.to_numpy()[n::n+1, idx] # only take the predictions for sample rows
    return pred

def _predict(alg, X):
    if alg == 'RNN':
        return _rnn_predict(X)
    
    X = X.drop(columns=drop_cols)
    if alg == 'LR':
        return lr_model.model.estimators_[idx].predict_proba(X)[:, 1]
    elif alg == 'XGB':
        return xgb_model.model.estimators_[idx].predict_proba(X)[:, 1]
    elif alg == 'RF':
        return rf_model.model.estimators_[idx].predict_proba(X)[:, 1]
    elif alg == 'NN':
        return nn_model.predict(X)[:, idx].cpu().detach().numpy()
    else:
        raise ValueError(f'{alg} not supported')
    
def predict(X):
    # NOTE: X is just a single sample, duplicated multiple times with different feature permutations
    weights, preds = [], []
    for alg, weight in ensemble_weights.items():
        weights.append(weight)
        preds.append(_predict(alg, X))
    pred = np.average(preds, axis=0, weights=weights)
    return pred


# In[36]:


get_ipython().run_cell_magic('time', '', "# compute shap values for the ENS model\n# NOTE: the explainer will loop through each sample row, and create multiple versions of the sample row\n# with different feature permutations, where the values are replaced with the background distribution values\nexplainer = shap.Explainer(predict, bg_dist, seed=42)\nshap_values = explainer(x, max_evals=800)\nsave_pickle(shap_values, f'{output_path}/feat_importance', 'shap_values_1000_sample')")


# In[58]:


shap_values = load_pickle(f'{output_path}/feat_importance', 'shap_values_1000_patient')

# ensure ikns are the same (if not, restart kernel to refresh numpy random seed)
ikn_idx = list(data.columns).index('ikn')
assert all(shap_values.data[:, ikn_idx] == data['ikn'])

data = data.drop(columns=drop_cols)
shap_values = shap_values[:, list(data.columns)]
# set display version of data (unnormalized)
norm_cols = prep.scaler.feature_names_in_
data[norm_cols] = prep.scaler.inverse_transform(data[norm_cols])
shap_values.data = data.to_numpy()


# In[59]:


# Group Importances
group_imp = {}
for group, keyword in variable_groupings_by_keyword.items():
    mask = pd.Index(shap_values.feature_names).str.contains(keyword)
    abs_mean_shap_vals = shap_values.abs.mean(axis=0).values[mask]
    group_imp[f'Sum of {group.title()} Features'] = sum(abs_mean_shap_vals)
group_imp = sorted(group_imp.items(), key=lambda x: x[1])
name, vals = list(zip(*group_imp))
group_imp


# In[24]:


save_path = f'{output_path}/figures/important_groups'
fig, ax = plt.subplots(figsize=(6,4))
ax.barh(name, vals)
remove_top_right_axis(ax)
ax.set_xlabel('mean(|SHAP Value|)')
fig.savefig(f'{save_path}/ENS_365d Mortality_SHAP_bar.jpg', bbox_inches='tight', dpi=300)
plt.show()


# In[11]:


# Feature Importnaces - show only the top 10 features
top = 10
idxs = np.argsort(-shap_values.abs.mean(axis=0).values)
top_ten_feats = shap_values[:, idxs].feature_names[:top]
shap_values = shap_values[:, top_ten_feats]
shap_values.feature_names = get_clean_variable_names(shap_values.feature_names)
shap_values = shap_values[:int(shap_values.shape[0]/top)*top] # make the shape a multiple of `top` by removing some samples


# In[12]:


dict(zip(shap_values.feature_names, shap_values.abs.mean(axis=0).values))


# In[51]:


save_path = f'{output_path}/figures/important_features'
shap.plots.bar(shap_values, show=False)
plt.savefig(f'{save_path}/ENS_365d Mortality_SHAP_bar.jpg', bbox_inches='tight', dpi=300)
plt.show()
shap.plots.waterfall(shap_values[50])
shap.plots.beeswarm(shap_values, show=False)
plt.savefig(f'{save_path}/ENS_365d Mortality_SHAP_beeswarm.jpg', bbox_inches='tight', dpi=300)
plt.show()
shap.plots.violin(shap_values, plot_type='layered_violin', show=False)
plt.savefig(f'{save_path}/ENS_365d Mortality_SHAP_violin.jpg', bbox_inches='tight', dpi=300)
# shap.plots.heatmap(shap_values)
# shap.plots.scatter(shap_values[:, 'Chemotherapy Cycle'], color=shap_values[:, 'Neutrophil'])


# # Prediction Correlation

# In[25]:


result = {te: pd.DataFrame() for te in ensembler.target_events}
for alg in ['LR', 'RF', 'XGB', 'NN', 'RNN']:
    for target_event, pred in preds[alg]['Test'].items():
        result[target_event][alg] = pred
        
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30,6))
for i, (target_event, df) in enumerate(result.items()):
    corr = df.corr()
    sns.heatmap(corr, annot=True, ax=axes[i], fmt='.2f', linewidth=.05, cmap='coolwarm', mask=np.triu(corr))
    axes[i].set_title(target_event)
fig.savefig(f'{output_path}/figures/output/prediction_correlation.jpg', bbox_inches='tight', dpi=300)


# # Survival Curve for Dev and Test Cohort

# In[246]:


from lifelines import KaplanMeierFitter
def plot_surv_curve_by_cohort(tag, event_dates, censor_dev=False, filename=None):
    fig, ax = plt.subplots(figsize=(6,6))
    kmf = KaplanMeierFitter()
    df = pd.DataFrame()
    max_duration = 0
    for cohort, group in tag.groupby('cohort'):
        group_dates = event_dates.loc[group.index]

        visit_dates = group_dates[DATE]
        print(f'{cohort} cohort: {visit_dates.min()} - {visit_dates.max()}')

        start = group_dates['first_visit_date']
        end = group_dates['death_date'].fillna(group_dates['death_date'].max())
        status = group_dates['death_date'].notnull()
        
        if cohort == 'Development' and censor_dev:
            # censor after one-year follow up
            max_date = visit_dates.max() + pd.Timedelta(days=365)
            mask = end > max_date
            end[mask] = max_date
            status[mask] = False # patient still alive at one-year follow up mark
            
        duration = month_diff(end, start)
        max_duration = max(max_duration, int(duration.max()))
        kmf.fit(duration, status, label=f'{cohort} Cohort')
        kmf.plot_survival_function(ax=ax)

        # report survival probability
        ci = kmf.confidence_interval_survival_function_
        for months_after in [6, 12, 18, 24]:
            lower, upper = ci[ci.index > months_after].iloc[0]
            surv_prob = kmf.predict(months_after)
            df.loc[cohort, f'{months_after} Months'] = f'{surv_prob:.2f} ({lower:.2f}-{upper:.2f})' 

    ax.set(xticks=np.arange(0, max_duration+6, 6), xlabel='Months', ylabel='Survival Probability')
    ax.legend(frameon=False)
    if filename is not None:
        fig.savefig(f'{output_path}/figures/output/{filename}.jpg', bbox_inches='tight', dpi=300)
    
    plt.show()
    return df


# In[247]:


# last seen date as max observed death date
filename = 'dev_test_cohort_survival_curve'
plot_surv_curve_by_cohort(tag, prep.event_dates, censor_dev=True, filename=f'{filename}_dev_censored')
plot_surv_curve_by_cohort(tag, prep.event_dates, censor_dev=False, filename=filename)


# # Characteristics of Lost Patients
# Patients that Lost Early PCCS in System-Guided Care

# In[8]:


from src.config import symptom_cols
from src.preprocess import Symptoms, combine_symptom_data
symp_cols = [col for col in symptom_cols if col in model_data.columns]


# In[9]:


pccs_df = get_pccs_analysis_data(
    evaluator, prep.event_dates[date_cols], 
    days_before_death=180, pred_thresh=year_mortality_thresh, 
    alg='ENS', split='Test', target_event='365d Mortality'
)
pccs_df = pd.concat([pccs_df, model_data.loc[pccs_df.index, symp_cols]], axis=1)


# In[10]:


display = lambda desc, N: print(f'Number of {desc} patients: {N} / {len(pccs_df)} ({N/len(pccs_df):.3f}%)')
# patients received Early PCCS in usual care but not in system-guided care
lost_df = pccs_df.query('received_early_pccs & ~early_pccs_by_alert')
display('lost', len(lost_df))

# patients received Early PCCS in system-guided care but not in usual care - for comparison
gained_df = pccs_df.query('~received_early_pccs & early_pccs_by_alert')
display('gained', len(gained_df))


# In[11]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_location', 'target'
]
lost_cs = CharacteristicsSummary(lost_df, Y_test.loc[lost_df.index], subgroups=subgroups)
lost_summary = lost_cs.get_summary()
top_cats = lost_cs.top_categories
gained_cs = CharacteristicsSummary(gained_df, Y_test.loc[gained_df.index], subgroups=subgroups, top_categories=top_cats)
gained_summary = gained_cs.get_summary()


# In[30]:


df = pd.concat([lost_summary, gained_summary], axis=1, keys=['Lost', 'Gained'])
df.to_csv(f'{output_path}/tables/lost_patient_characteristic_summary.csv')
df


# In[35]:


def plot_symptom_dist_between_two_groups(df1, df2, group_names=None, save_path=None):
    if group_names is None: group_names = ['Group 1', 'Group 2']
    nrows = 2
    ncols = int(len(symp_cols) / nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    stats = {}
    for i, col in enumerate(symp_cols):
        ax = axes[i]
        remove_top_right_axis(ax)
        title = 'PRFS Grade' if col == 'prfs_grade' else col.replace('_', ' ').title()
        kwargs = dict(stat='proportion', ax=ax, element='step', fill=False)
        sns.histplot(df1[col].astype("category"), color=sns.color_palette()[0], **kwargs)
        sns.histplot(df2[col].astype("category"), color=sns.color_palette()[1], **kwargs)
        ax.set(title=title, xlabel='Score', xticks=range(int(df1[col].max()) + 1))
        
        # get t-value and p-value
        result = scipy.stats.ranksums(df1[col], df2[col], nan_policy='omit')
        stats[col] = {'stat': result.statistic, 'p-value': result.pvalue}
        # stats[f'{group_names[0]} Missingness Rate'] = 1 - mask1.mean()
        # stats[f'{group_names[1]} Missingness Rate'] = 1 - mask2.mean()
        
        # get median with interquartile range
        for name, df in zip(group_names, [df1, df2]):
            median = int(df[col].median())
            q25, q75 = np.nanpercentile(df[col], [25, 75]).astype(int)
            stats[col][f'median - {name}'] = f'{median} ({q25}-{q75})'
        
    ax.legend(group_names, loc='upper left', bbox_to_anchor=(1, 0.5), frameon=False)
    if save_path is not None: fig.savefig(save_path, bbox_inches='tight', dpi=300)
    return pd.DataFrame(stats).T.round(4).sort_index()


# In[32]:


# check symptom scores for lost patients at the date closest preceding the usual care first PCCS date
# if the patients' symptom scores are low, the patients may not have necssarily needed PC consultation
df = lost_df[['ikn', 'first_PCCS_date']].copy()
df = df.rename(columns={'first_PCCS_date': DATE})
df['ikn'] = df['ikn'].astype(str).str.zfill(10) # insert leading zeros
symp = Symptoms(processes=32)
symp_df = symp.run(df, verbose=False)
lost_df = combine_symptom_data(df, symp_df)


# In[68]:


"""
Null hypothesis - the two groups are the same (in terms of central tendency)
Wilcoxon ranksum statistic: difference between the two groups
    - if negative, df1 median is smaller than df2 median
p-value: statistical significance of the difference
    - if p-value > 0.05, result is not significant
    - if p-value <= 0.05, result is significant and null hypothesis should be rejected 
"""
plot_symptom_dist_between_two_groups(
    lost_df, gained_df, 
    group_names=['Lost Early PC', 'Gained Early PC'], 
    save_path=f'{output_path}/figures/output/symp_dist_lost_vs_gained_epc.jpg',
)


# In[29]:


# number of patients for each score in lost_df
res = {}
for symp, scores in lost_df[symp_cols].items():
    res[symp] = {i: lost_df.loc[scores == i, 'ikn'].nunique()
                 for i in scores.unique()
                 if ~np.isnan(i)}
pd.DataFrame(res).sort_index()


# In[51]:


# number of patients for each score in gained_df
res = {}
for symp, scores in gained_df[symp_cols].items():
    res[symp] = {i: gained_df.loc[scores == i, 'ikn'].nunique()
                 for i in scores.unique()
                 if ~np.isnan(i)}
pd.DataFrame(res).sort_index()


# # Label Rate For Alarm vs No Alarm

# In[ ]:


preds, labels = ensembler.preds.copy(), ensembler.labels.copy()


# In[36]:


alarm = preds['ENS']['Test'] > year_mortality_thresh
label = labels['Test']


# In[38]:


result = {'with_alarm': {}, 'no_alarm': {}}
for target_event, pred in alarm.items():
    # With alarm
    mask = label[target_event][pred]
    result['with_alarm'][target_event] = f'{mask.mean():.3f} (N={len(mask)})'
    # No alarm
    mask = label[target_event][~pred]
    result['no_alarm'][target_event] = f'{mask.mean():.3f} (N={len(mask)})'
pd.DataFrame(result)


# # LASSO Model - More Metrics

# In[79]:


from sklearn.metrics import roc_auc_score, average_precision_score
from src.evaluate import CLF_SCORE_FUNCS
from src.conf_int import ScoreConfidenceInterval


# In[81]:


ci = ScoreConfidenceInterval(output_path, CLF_SCORE_FUNCS)
lasso_trainer = LASSOTrainer(X, Y, tag, output_path, target_event='365d Mortality')
C_search_space = [3.9e-5, 4.3e-5, 4.52e-5, 4.8e-5, 5.9e-5, 8.4e-5, None]
for C in C_search_space:
    if C is None:
        model = load_pickle(output_path, 'LASSO')
        estimator_idx = list(Y.columns).index(lasso_trainer.target_event)
        lasso_trainer.labels = lasso_trainer._labels
    else:
        print(f'Parameter C: {C:.2E}.')
        model = lasso_trainer.train_model(alg='LR', save=False, penalty='l1', n_jobs=-1, inv_reg_strength=C)
        estimator_idx = None
    
    for split in ['Valid', 'Test']:
        print(f'Split: {split}.')
        
        pred_prob = lasso_trainer.predict(model, split=split, alg='LR')[lasso_trainer.target_event]
        label = lasso_trainer.labels[split][lasso_trainer.target_event]
        # Get score
        roc_score = roc_auc_score(label, pred_prob)
        ap_score = average_precision_score(label, pred_prob)
        
        # Get 95% CI
        score_ci = ci.get_score_confidence_interval(label, pred_prob,store=False, verbose=False)

        lower, upper = score_ci['AUROC']
        print(f'AUROC Score: {roc_score:.3f} ({lower:.3f}-{upper:.3f}).')
        lower, upper = score_ci['AUPRC']
        print(f'AUPRC Score: {ap_score:.3f} ({lower:.3f}-{upper:.3f}).')
        
    # Get coefficients and intercept
    coef = lasso_trainer.get_coefficients(model, estimator_idx=estimator_idx)
    intercept = lasso_trainer.get_intercept(model, estimator_idx=estimator_idx)
    
    n_features = len(coef)
    top_ten = coef[:10].round(3).to_dict()
    top_ten['intercept'] = np.round(intercept, 3)
    print(f'Number of non-zero weighted features: {n_features}. ')
    print(f'Top 10 Features: {top_ten}')
