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
#!/usr/bin/env python
# coding: utf-8

# # Supplementary Analysis
# From reveiwer's comments and recommendation

# In[1]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import shap
import statsmodels.api as sm

from src.config import root_path, death_folder, split_date, DATE, variable_groupings_by_keyword
from src.evaluate import EvaluateClf
from src.impact import get_pccs_analysis_data
from src.prep_data import PrepDataEDHD
from src.summarize import CharacteristicsSummary
from src.train import TrainML, TrainENS, TrainRNN
from src.utility import (
    load_pickle, save_pickle, initialize_folders, 
    equal_rate_pred_thresh, 
    get_clean_variable_names,
    month_diff,
)
from src.visualize import remove_top_right_axis


# In[3]:


main_dir = f'{root_path}/{death_folder}'
output_path = f'{main_dir}/models'
target_keyword = 'Mortality'


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


preds = load_pickle(f'{output_path}/preds', 'ML_preds')
preds_rnn = load_pickle(f'{output_path}/preds', 'RNN_preds')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred

train_ens = TrainENS(X, Y, tag, output_path, preds)
train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)
preds, labels = train_ens.preds.copy(), train_ens.labels.copy()

preds_lasso = load_pickle(f'{output_path}/preds', filename='LASSO_preds')
for split, pred in preds_lasso.items(): preds[split]['LASSO'] = pred['LR']


# In[6]:


eval_models = EvaluateClf(output_path, preds, labels)
eval_models.orig_data = model_data
year_mortality_thresh = equal_rate_pred_thresh(
    eval_models, prep.event_dates, split='Test', alg='ENS', target_event='365d Mortality'
)


# In[7]:


def get_data(split='Test', alg='ENS', target_event='365d Mortality', pred_thresh=None):
    pred = preds[split][alg][target_event]
    label = labels[split][target_event].astype(int)

    dates = prep.event_dates.loc[pred.index]
    time = (dates[DATE] - dates[f'first_{DATE}']).dt.days / 365 # years

    patient_id = tag.loc[pred.index, 'ikn']

    df = pd.concat([label, pred, time, patient_id], keys=['Label', 'Pred', 'Time', 'Subject'], axis=1)
    if pred_thresh is not None: df['Pred'] = df['Pred'] > pred_thresh 
    return df


# In[8]:


train_df = get_data(split='Train', pred_thresh=year_mortality_thresh)
test_df = get_data(split='Test', pred_thresh=year_mortality_thresh)


# # Generalized Estimating Equations

# In[9]:


def gee_fit(df, struct='Autoregressive'):
    # Ref: https://towardsdatascience.com/an-introduction-to-generalized-estimating-equations-bc7dee570478
    fam = sm.families.Binomial()
    time = 'Time'
    if struct == 'Unstructured':
        time = (df[time] * 365 / 7).astype(int)
        time = time.clip(upper=time.quantile(0.99))
    struct = {
        'Independence': sm.cov_struct.Independence(), 
        'Exchangeable': sm.cov_struct.Exchangeable(), 
        'Autoregressive': sm.cov_struct.Autoregressive(grid=True),
        'Unstructured': sm.cov_struct.Unstructured(),
    }[struct]
    model = sm.GEE.from_formula(
        formula='Label ~ Pred', groups='Subject', data=df, time=time, cov_struct=struct, family=fam
    )
    result = model.fit()
    return {'result': result, 'model': model}

def gee_eval(result, model):
    # odds ratio = ratio of the odds of true label in the presence of positive prediction and odds of true label in the
    # absence of positive prediction
    odds_ratio = np.exp(result.params['Pred[T.True]'])
    
    # R2 score is not appropriate for GEE. We can consider other goodness-of-fit measures such as QIC
    # QIC = quasi-information criterion (quasi-likelihood under the independence model information criterion)
    # Smaller QIC is "better"
    qic, qicu = result.qic(model.estimate_scale())
    
    # Cox-Snell likelihood ratio pseudo R-squared
    psuedo_rsquared = result.pseudo_rsquared(kind='cs')
    
    return {'Odds Ratio': odds_ratio, 'QIC': qic, 'Psuedo R-squared': psuedo_rsquared}


# In[249]:


get_ipython().run_cell_magic('time', '', "structs = ['Independence', 'Exchangeable', 'Autoregressive', 'Unstructured']\nresults = {struct: gee_fit(test_df, struct) for struct in structs}\nsave_pickle(results['Unstructured'], output_path, 'GEE')\npd.DataFrame({struct: gee_eval(*gee_output.values()) for struct, gee_output in results.items()}).T")


# Interpretation
# - Higher odds ratio, stronger the association between prediction and label
# - Higher psuedo r-squared, the better the model performed compared to a model with no predictors (just the intercept)

# In[252]:


# results = {struct: gee_fit(test_df, struct) for struct in ['Independence', 'Exchangeable', 'Autoregressive']}
# results['Unstructured'] = load_pickle(output_path, 'GEE')
gee = load_pickle(output_path, 'GEE')
result, model = gee['result'], gee['model']
result.summary()


# In[100]:


result.summary2()


# In[26]:


for struct in ['Exchangeable', 'Autoregressive', 'Unstructured']:
    model = results[struct]['model']
    dep_params = model.cov_struct.dep_params
    if struct == 'Unstructured':
        # show where the size of correlation matrix (aka dep_params) came from
        assert len(np.unique(model.time)) == len(dep_params)
        
        # The correlation matrix shows common/average correlations between pairs of observations within a subject
        print('Unstructured strucutre dependency params = ')
        ax = sns.heatmap(pd.DataFrame(dep_params))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xticks(rotation=90)
        ax.set(xlabel='Weeks', ylabel='Weeks')
        plt.savefig(f'{output_path}/figures/output/gee_unstructured_corr_matrix.jpg', bbox_inches='tight', dpi=300)
    else:
        print(f'{struct} structure dependency params = {dep_params:.3f}')


# In[253]:


# Sanity Check
import scipy.stats as stats
pos_pred_label = test_df.query('Pred')['Label'].astype(bool)
neg_pred_label = test_df.query('~Pred')['Label'].astype(bool)
table = [[sum(pos_pred_label), sum(~pos_pred_label)], 
         [sum(neg_pred_label), sum(~neg_pred_label)]]
odds_ratio, p_value = stats.fisher_exact(table)
print(f'Odds ratio = {odds_ratio:.2f}, P-value = {p_value}')


# In[256]:


# take a random time point for each patient
rand_df = test_df.groupby('Subject').apply(lambda g: g.sample(1, random_state=42))
gee = gee_fit(rand_df) # NOTE: obvious but it would be same result regardless of struct
rand_result, rand_model = gee['result'], gee['model']
print(gee_eval(rand_result, rand_model))
rand_result.summary()


# In[257]:


rand_result.summary2()


# In[ ]:


# Ref: https://statology.org/likelihood-ratio-test-in-python
# I am not using likelihood ratio correctly here...likelihood ratio is only valid for comparing nested models
likelihood_ratio = -2 * (results['Independence'].llf - results['Unstructured'].llf)
dof = 0 # difference in number of parameters between the two models
p_val = scipy.stats.chi2.sf(likelihood_ratio, dof)
print(f'likelihood ratio between the two models = {likelihood_ratio:.2f}. Correspond p-value = {p_val}')
# If p-value is not less than 0.05, we fail to reject the null hypothesis. 
# This means there is no difference between the two models other than due to chance.


# # SHAP
# Takes ages to compute. Can only compute SHAP for a subset of data

# In[85]:


x = pd.concat([X_test.astype(float), Y_test.astype(float), tag.loc[test_mask, 'ikn']], axis=1)
x = x.sample(2000, random_state=42)
x, bg_dist = x[:1000], x[1000:] # use the latter half as background distribution
idx = Y.columns.tolist().index('365d Mortality')
drop_cols = ['ikn'] + Y_test.columns.tolist()


# In[9]:


lr_model = load_pickle(output_path, 'LR')
rf_model = load_pickle(output_path, 'RF')
xgb_model = load_pickle(output_path, 'XGB')
nn_model = load_pickle(output_path, 'NN')

ensemble_weights = load_pickle(f'{output_path}/best_params', 'ENS_params')

rnn_param = load_pickle(f'{output_path}/best_params', 'RNN_params')
del rnn_param['learning_rate']
train_rnn = TrainRNN(X, Y, tag, output_path)
rnn_model = train_rnn.get_model(load_saved_weights=True, **rnn_param)


# In[10]:


def _predict(alg, X):
    if alg == 'RNN':
        train_rnn.ikns = X['ikn']
        Y = X[train_rnn.target_events] # dummy variable
        X = X.drop(columns=drop_cols)
        train_rnn.tensor_datasets['Test'] = train_rnn.transform_to_tensor_dataset(X, Y)
        pred, index_arr = train_rnn._get_model_predictions(rnn_model, 'Test')
        return pred[:, idx]
    
    X = X.drop(columns=drop_cols)
    if alg == 'LR':
        return lr_model.estimators_[idx].predict_proba(X)[:, 1]
    elif alg == 'XGB':
        return xgb_model.estimators_[idx].predict_proba(X)[:, 1]
    elif alg == 'RF':
        return rf_model.estimators_[idx].predict_proba(X)[:, 1]
    elif alg == 'NN':
        return nn_model.predict_proba(X)[:, idx]
    else:
        raise ValueError(f'{alg} not supported')
    
def predict(X):
    weights, preds = [], []
    for alg, weight in ensemble_weights.items():
        weights.append(weight)
        preds.append(_predict(alg, X))
    pred = np.average(preds, axis=0, weights=weights)
    return pred


# In[36]:


get_ipython().run_cell_magic('time', '', "# compute shap values for the ENS model\nexplainer = shap.Explainer(predict, bg_dist)\nshap_values = explainer(x, max_evals=800)\nsave_pickle(shap_values, f'{output_path}/feat_importance', 'shap_values')")


# In[86]:


x = x.drop(columns=drop_cols)
shap_values = load_pickle(f'{output_path}/feat_importance', 'shap_values')
shap_values = shap_values[:, list(x.columns)]
# set display version of data (unnormalized)
norm_cols = prep.scaler.feature_names_in_
x[norm_cols] = prep.scaler.inverse_transform(x[norm_cols])
shap_values.data = x.to_numpy()


# In[145]:


# Group Importances
group_imp = {}
for group, keyword in variable_groupings_by_keyword.items():
    mask = pd.Index(shap_values.feature_names).str.contains(keyword)
    abs_mean_shap_vals = shap_values.abs.mean(axis=0).values[mask]
    group_imp[f'Sum of {group.title()} Features'] = sum(abs_mean_shap_vals)
group_imp = sorted(group_imp.items(), key=lambda x: x[1])
name, vals = list(zip(*group_imp))


# In[148]:


save_path = f'{output_path}/figures/important_groups'
fig, ax = plt.subplots(figsize=(6,4))
ax.barh(name, vals)
remove_top_right_axis(ax)
ax.set_xlabel('mean(|SHAP Value|)')
fig.savefig(f'{save_path}/ENS_365d Mortality_SHAP_bar.jpg', bbox_inches='tight', dpi=300)
plt.show()


# In[80]:


# Feature Importnaces - show only the top 10 features
top = 10
idxs = np.argsort(-shap_values.abs.mean(axis=0).values)
top_ten_feats = shap_values[:, idxs].feature_names[:top]
shap_values = shap_values[:, top_ten_feats]
shap_values.feature_names = get_clean_variable_names(shap_values.feature_names)


# In[72]:


save_path = f'{output_path}/figures/important_features'
shap.plots.bar(shap_values, show=False)
plt.savefig(f'{save_path}/ENS_365d Mortality_SHAP_bar.jpg', bbox_inches='tight', dpi=300)
plt.show()
shap.plots.waterfall(shap_values[50])
shap.plots.beeswarm(shap_values, show=False)
plt.savefig(f'{save_path}/ENS_365d Mortality_SHAP_beeswarm.jpg', bbox_inches='tight', dpi=300)
plt.show()
shap.plots.heatmap(shap_values)
shap.plots.scatter(shap_values[:, 'Chemotherapy Cycle'], color=shap_values[:, 'Neutrophil'])


# # Prediction Correlation

# In[149]:


result = {te: pd.DataFrame() for te in train_ens.target_events}
for alg in ['LR', 'RF', 'XGB', 'NN', 'RNN']:
    for target_event, pred in preds['Test'][alg].items():
        result[target_event][alg] = pred
        
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30,6))
for i, (target_event, df) in enumerate(result.items()):
    corr = df.corr()
    sns.heatmap(corr, annot=True, ax=axes[i], fmt='.2f', linewidth=.05, cmap='coolwarm', mask=np.triu(corr))
    axes[i].set_title(target_event)
fig.savefig(f'{output_path}/figures/output/prediction_correlation.jpg', bbox_inches='tight', dpi=300)


# # Survival Curve for Dev and Test Cohort

# In[156]:


from lifelines import KaplanMeierFitter
fig, ax = plt.subplots(figsize=(6,6))
kmf = KaplanMeierFitter()
df = pd.DataFrame()
for cohort, group in tag.groupby('cohort'):
    event_dates = prep.event_dates.loc[group.index]
    start = event_dates['first_visit_date']
    end = event_dates['death_date'].fillna(event_dates['last_seen_date'])
    duration = month_diff(end, start)
    status = event_dates['death_date'].notnull()
    kmf.fit(duration, status, label=f'{cohort} Cohort')
    kmf.plot_survival_function(ax=ax)
    
    # report survival probability
    ci = kmf.confidence_interval_survival_function_
    for months_after in [6, 12, 18, 24]:
        lower, upper = ci[ci.index > months_after].iloc[0]
        surv_prob = kmf.predict(months_after)
        df.loc[cohort, f'{months_after} Months'] = f'{surv_prob:.2f} ({lower:.2f}-{upper:.2f})' 
ax.set(xticks=np.arange(0, 80, 6), xlabel='Months', ylabel='Survival Probability')
ax.legend(frameon=False)
fig.savefig(f'{output_path}/figures/output/dev_test_cohort_survival_curve.jpg', bbox_inches='tight', dpi=300)
df


# # Characteristics of Lost Patients
# Patients that Lost Early PCCS in System-Guided Care

# In[157]:


from src.config import symptom_cols
symp_cols = [col for col in symptom_cols if col in model_data.columns]
date_cols = ['visit_date', 'death_date', 'first_PCCS_date', 'first_visit_date', 'last_seen_date']


# In[158]:


pccs_df = get_pccs_analysis_data(
    eval_models, prep.event_dates[date_cols], 
    days_before_death=180, pred_thresh=year_mortality_thresh, 
    alg='ENS', split='Test', target_event='365d Mortality'
)
pccs_df = pd.concat([pccs_df, model_data.loc[pccs_df.index, symp_cols]], axis=1)


# In[208]:


display = lambda desc, N: print(f'Number of {desc} patients: {N} / {len(pccs_df)} ({N/len(pccs_df):.3f}%)')
# patients received Early PCCS in usual care but not in system-guided care
lost_df = pccs_df.query('received_early_pccs & ~early_pccs_by_alert')
display('lost', len(lost_df))

# patients retained Early PCCS in system-guided care for comparison
retained_df = pccs_df.query('received_early_pccs & early_pccs_by_alert')
display('retained', len(retained_df))

# patients received Early PCCS in usual care for comparison
usual_df = pccs_df.query('received_early_pccs')
display('usual', len(usual_df))


# In[209]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_location', 'target'
]
lost_cs = CharacteristicsSummary(lost_df, Y_test.loc[lost_df.index], subgroups=subgroups)
retained_cs = CharacteristicsSummary(retained_df, Y_test.loc[retained_df.index], subgroups=subgroups)
usual_cs = CharacteristicsSummary(usual_df, Y_test.loc[usual_df.index], subgroups=subgroups)


# In[210]:


df = pd.concat([lost_cs.get_summary(), retained_cs.get_summary(), usual_cs.get_summary()], 
               axis=1, keys=['Lost', 'Retained', 'Usual'])
df.to_csv(f'{output_path}/tables/lost_patient_characteristic_summary.csv')
df


# In[211]:


lost_df[symp_cols].describe().round(2)
