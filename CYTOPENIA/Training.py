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

# In[1]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pandas as pd
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import root_path, cyto_folder, split_date, blood_types
from src.evaluate import EvaluateClf
from src.model import SimpleBaselineModel
from src.prep_data import PrepDataCYTO
from src.train import TrainML, TrainRNN, TrainENS
from src.summarize import data_description_summary, feature_summary
from src.utility import initialize_folders, load_pickle, get_nunique_categories, get_nmissing
from src.visualize import importance_plot, subgroup_performance_plot


# In[3]:


# config
processes = 64
output_path = f'{root_path}/{cyto_folder}/models'
initialize_folders(output_path)


# # Prepare Data for Model Training

# In[4]:


prep = PrepDataCYTO()
model_data = prep.get_data(verbose=True)
model_data


# In[5]:


sorted(model_data.columns.tolist())


# In[6]:


prep.event_dates['first_visit_date'].dt.year.value_counts()


# In[7]:


get_nunique_categories(model_data)


# In[8]:


get_nmissing(model_data, verbose=True)


# In[4]:


prep = PrepDataCYTO() # need to reset
model_data = prep.get_data(missing_thresh=80, verbose=True)
X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[10]:


prep.get_label_distribution(Y, tag, with_respect_to='sessions')


# In[11]:


prep.get_label_distribution(Y, tag, with_respect_to='patients')


# In[5]:


# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]
Y_train, Y_valid, Y_test = Y[train_mask], Y[valid_mask], Y[test_mask]


# ## Study Characteristics

# In[13]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location', 'target', 'gcsf'
]
data_description_summary(
    model_data, Y, tag, save_dir=f'{output_path}/tables', partition_method='cohort', target_event='Thrombocytopenia', subgroups=subgroups
)


# ## Feature Characteristics

# In[14]:


df = prep.ohe.encode(model_data.copy(), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
df = df[train_mask].drop(columns=['ikn'])
feature_summary(
    df, save_dir=f'{output_path}/tables', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# # Train Models

# ## Main ML Models

# In[14]:


train_ml = TrainML(X, Y, tag, output_path, n_jobs=processes)
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True)


# ## RNN Model

# In[171]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby(tag.loc[train_mask, 'ikn']).apply(len)
dist_seq_lengths = dist_seq_lengths.clip(upper=dist_seq_lengths.quantile(q=0.999))
fig, ax = plt.subplots(figsize=(15, 3))
ax.grid(zorder=0)
sns.histplot(dist_seq_lengths, ax=ax, zorder=2, bins=int(dist_seq_lengths.max()))


# In[37]:


train_rnn = TrainRNN(X, Y, tag, output_path)
train_rnn.tune_and_train(run_bayesopt=False, run_training=True, run_calibration=True, save_preds=True)


# ## ENS Model 
# Find Optimal Ensemble Weights

# In[13]:


# combine rnn and ml predictions
preds = load_pickle(f'{output_path}/preds', 'ML_preds')
preds_rnn = load_pickle(f'{output_path}/preds', 'RNN_preds')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(X, Y, tag, output_path, preds)


# In[14]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# # Evaluate Models

# In[15]:


# setup the final prediction and labels
preds, labels = train_ens.preds, train_ens.labels
base_cols = ['regimen'] + [f'baseline_{bt}_value' for bt in blood_types]
base_model = SimpleBaselineModel(model_data[base_cols], labels)
base_preds = base_model.predict()
for split, pred in base_preds.items(): preds[split].update(pred)


# In[17]:


eval_models = EvaluateClf(output_path, preds, labels)
eval_models.get_evaluation_scores(display_ci=True, load_ci=True, save_ci=False)


# In[28]:


eval_models.plot_curves(curve_type='pr', legend_loc='lower left', figsize=(12,18))
eval_models.plot_curves(curve_type='roc', legend_loc='lower right', figsize=(12,18))
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18)) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_loc='upper left', figsize=(12,18)) 
# eval_models.plot_calibs(
#     include_pred_hist=True, legend_loc='upper left', figsize=(12,28),
#     padding={'pad_y1': 0.3, 'pad_y0': 3.0}, save=False, 
# )


# # Post-Training Analysis

# ## Most Important Features/Feature Groups

# In[ ]:


get_ipython().system('python scripts/feat_imp.py --adverse-event CYTO --output-path {output_path}')


# In[101]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot(
    'ENS', eval_models.target_events, output_path, figsize=(6,15), top=10, importance_by='feature', 
    padding={'pad_x0': 2.6}, colors=['#1f77b4', '#ff7f0e', '#2ca02c']
)


# In[102]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot(
    'ENS', eval_models.target_events, output_path, figsize=(6,15), top=10, importance_by='group', 
    padding={'pad_x0': 2.6}, colors=['#1f77b4', '#ff7f0e', '#2ca02c']
)


# In[19]:


importance = pd.read_csv(f'{output_path}/perm_importance/ENS_feature_importance.csv').set_index('index')
auroc_scores = eval_models.get_evaluation_scores(save_score=False).loc[('ENS', 'AUROC Score'), 'Test']
pd.concat([importance.sum(), 1 - auroc_scores], axis=1, keys=['Cumulative Importance', '1 - AUROC'])


# ## All the Plots

# In[55]:


for blood_type, blood_info in blood_types.items():
    target_event = blood_info['cytopenia_name']
    print(f'Displaying all the plots for {target_event}')
    eval_models.all_plots_for_single_target(alg='ENS', target_event=target_event, calib_strategy='uniform', save=False)


# ## Threshold Operating Points

# In[25]:


pred_thresholds = np.arange(0.05, 1.01, 0.05)
perf_metrics = ['warning_rate', 'precision', 'recall', 'NPV', 'specificity']
thresh_df = eval_models.operating_points(points=pred_thresholds, op_metric='threshold', perf_metrics=perf_metrics)
thresh_df


# ## Precision Operating Points

# In[26]:


desired_precisions = [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.75]
perf_metrics = ['warning_rate', 'threshold', 'recall', 'NPV', 'specificity']
eval_models.operating_points(points=desired_precisions, op_metric='precision', perf_metrics=perf_metrics)


# ## Threshold Selection

# In[22]:


neutro_thresh = 0.25
anemia_thresh = 0.25
thromb_thresh = 0.15 # NOTE: the maximum prediction threshold for thrombocytopenia in the regimen subgroup ac-pacl(dd) was 0.17
pred_thresholds = [neutro_thresh, anemia_thresh, thromb_thresh]


# ## Performance on Subgroups

# In[20]:


subgroups = [
    'all', 'age', 'sex', 'immigrant', 'language', 'arrival', 'income', 'world_region_of_birth',
    'area_density', 'regimen', 'cancer_location', 'days_since_starting'
]


# In[23]:


perf_kwargs = {'perf_metrics': ['precision', 'recall', 'event_rate']}
subgroup_performance = eval_models.get_perf_by_subgroup(
    model_data, subgroups=subgroups, pred_thresh=pred_thresholds, alg='ENS', 
    save=True, display_ci=True, load_ci=True, perf_kwargs=perf_kwargs
)
subgroup_performance


# In[25]:


subgroup_performance = pd.read_csv(f'{output_path}/tables/subgroup_performance.csv', index_col=[0,1], header=[0,1])
subgroup_plot_groupings = {
    'Demographic': [
        'Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Language', 'Neighborhood Income Quintile', 
        'Immigrant World Region of Birth', 'Area of Residence'
    ],
    'Treatment': [
        'Entire Test Cohort', 'Regimen', 'Topography ICD-0-3', 'Days Since Starting Regimen'
    ]
}
subgroup_plot_padding = {'pad_y0': 1.2, 'pad_x1': 2.8, 'pad_y1': 0.2}
subgroup_plot_width = {'Demographic': 18, 'Treatment': 12}


# ## Subgroup Performance Plot

# In[163]:


for name, grouping in subgroup_plot_groupings.items():
    for blood_type, blood_info in blood_types.items():
        target_event = blood_info['cytopenia_name']
        print(f'Plotted {name} Subgroup Performance Plot for {target_event}')
        subgroup_performance_plot(
            subgroup_performance, target_event=target_event, subgroups=grouping, padding=subgroup_plot_padding,
            figsize=(subgroup_plot_width[name],30), save_dir=f'{output_path}/figures/subgroup_perf/{name}'
        )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ## Decision Curve Plot

# In[121]:


result = eval_models.plot_decision_curves('ENS', xlim=(-0.05, 1.05))
result['Neutropenia'].tail(n=100)


# ## Randomized Individual Patient Performance

# In[21]:


from src.config import cytopenia_grades


# In[45]:


def get_patient_info(df):
    age = int(df['age'].mean())
    sex = 'Female' if df['sex'].iloc[0] =='F' else 'Male'
    regimen = df['regimen'].iloc[0]
    return f"{age} years old {sex} patient under regimen {regimen}"

def line_plot(x, y, ax, thresh, colors=None):
    """Do a simple line plot, but color changes when value crosses a threshold"""
    if colors is None: colors = ['g', 'r']
    
    # interpolate the values in between, so color change is more granular
    new_x = np.linspace(x.min(), x.max(), 10000)
    new_y = np.interp(new_x, x, y)
    ax.plot(new_x, new_y, color=colors[0])
    
    under_thresh = new_y.copy()
    under_thresh[under_thresh > thresh] = np.nan
    ax.plot(new_x, under_thresh, color=colors[1])
    
    ax.axhline(y=thresh, color='b', alpha=0.5)
    ax.grid(axis='x')

def plot_patient_prediction(eval_models, pred_thresholds, split='Test', alg='XGB', num_ikn=3, seed=0, save=False):
    """
    Args:
        num_ikn (int): the number of random patients to analyze
    """
    np.random.seed(seed)
    
    pred_prob = eval_models.preds[split][alg]
    orig_data = eval_models.orig_data.loc[pred_prob.index]
    label = eval_models.labels[split]

    # only consider patients who had more than 3 chemo cycles AND had at least one cytopenia event
    ikn_count = orig_data.loc[label.any(axis=1), 'ikn'].value_counts()
    ikns = ikn_count[ikn_count > 3].index

    for _ in range(num_ikn):
        ikn = np.random.choice(ikns) # select a random patient from the consideration pool
        idxs = orig_data.query('ikn == @ikn').index # get the indices corresponding with the selected patient
        pred = pred_prob.loc[idxs]
        df = orig_data.loc[idxs]
        patient_info = get_patient_info(df)
        print(patient_info)

        fig = plt.figure(figsize=(18, 20))
        
        # days since admission at target date
        days_since_admission = df['days_since_last_chemo'].shift(-1).cumsum().to_numpy()
        # last target date will be "unknown", use default cycle length
        days_since_admission[-1] = days_since_admission[-2] + int(df['cycle_length'].iloc[-1]) 
        
        for i, (bt, blood_info) in enumerate(blood_types.items()):
            true_count = df[f'target_{bt}_value']
            name = blood_info['cytopenia_name']
            unit = blood_info['unit']
            thresh = cytopenia_grades['Grade 2'][name]

            ax1 = fig.add_subplot(6, 3, i+1) # 3 blood types * 2 subplots each
            line_plot(days_since_admission, true_count, ax1, thresh, colors=['g', 'r'])
            ax1.tick_params(labelbottom=False)
            ax1.set(ylabel=f"Blood count ({unit})", title=f"Patient {bt} measurements")

            ax2 = fig.add_subplot(6, 3, i+1+3, sharex=ax1)
            line_plot(days_since_admission, pred[name], ax2, pred_thresholds[i], colors=['r', 'g'])
            ax2.set(
                xticks=days_since_admission, yticks=np.arange(0, 1.01, 0.2), 
                xlabel='Days since admission', ylabel=f"Risk of {name}", 
                title=f"{alg} Model Prediction for {name}"
            )
        if save:
            plt.savefig(f'{output_path}/figures/patients/{ikn}_performance.jpg', bbox_inches='tight') #dpi=300
        plt.show()


# In[46]:


plot_patient_prediction(eval_models, pred_thresholds, alg='ENS', num_ikn=8, seed=1, save=False)


# # SCRATCH NOTES

# ## XGB as txt file

# In[57]:


XGB_model = load_pickle(output_path, 'XGB')
for idx, blood_type in enumerate(blood_types):
    estimator = XGB_model.estimators_[0].calibrated_classifiers_[0].estimator
    estimator.get_booster().dump_model(f'{output_path}/interpret/XGB_{blood_type}.txt')
    estimator.save_model(f'{output_path}/interpret/XGB_{blood_type}.model')


# ## Graph Visualization

# In[60]:


from src.visualize import tree_plot


# In[63]:


tree_plot(train_ml, target_event='Neutropenia', alg='RF')


# ## More Senstivity/Error Analysis

# In[159]:


df = eval_models.get_perf_by_subgroup(
    model_data, subgroups=['all', 'cycle_length'], pred_thresh=neutro_thresh, alg='ENS', display_ci=False, 
    perf_kwargs=perf_kwargs
)
subgroup_performance_plot(df, target_event='Neutropenia', padding=subgroup_plot_padding, figsize=(4,20))


# In[26]:


# analyze regimen subgroups with the worst performance
perf_kwargs = {'perf_metrics': ['precision', 'recall', 'event_rate', 'count']}
df = eval_models.get_perf_by_subgroup(
    model_data, subgroups=['regimen'], pred_thresh=neutro_thresh, alg='ENS', target_events=['Neutropenia'], 
    save=False, display_ci=False, perf_kwargs=perf_kwargs, top=150
)
df[('Neutropenia', 'N')] = df.pop(('Neutropenia', 'N')).astype(int)
df.sort_values(by=('Neutropenia', 'AUROC'))


# In[27]:


# analyze cancer subgroups with the worst performance
df = eval_models.get_perf_by_subgroup(
    model_data, subgroups=['cancer_location'], pred_thresh=neutro_thresh, alg='ENS', target_events=['Neutropenia'], 
    save=False, display_ci=False, perf_kwargs=perf_kwargs, top=100
)
df[('Neutropenia', 'N')] = df.pop(('Neutropenia', 'N')).astype(int)
df.sort_values(by=('Neutropenia', 'AUROC'))


# ## Hyperparameters

# In[188]:


from src.utility import get_hyperparameters
get_hyperparameters(output_path)


# ## Data Summary of Filtered Data

# In[80]:


from src.utility import twolevel
from src.summarize import DataPartitionSummary
chemo_df = prep.load_data(dtypes=prep.chemo_dtypes)
summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
dps = DataPartitionSummary(model_data, model_data, 'Included Data') # Both target and baseline blood count for hemoglobin, neutrophil, platelet required
dps.get_summary(summary_df, include_target=False)

mask = ~chemo_df.index.isin(model_data.index)
chemo_df['neighborhood_income_quintile_is_missing'] = chemo_df['neighborhood_income_quintile'].isnull()
dps = DataPartitionSummary(chemo_df[mask], chemo_df[mask], 'Excluded Data', top_category_items=dps.top_category_items)
dps.get_summary(summary_df, include_target=False)
summary_df


# ## Missingness By Splits

# In[189]:


from src.utility import get_nmissing_by_splits
missing = get_nmissing_by_splits(model_data, eval_models.labels)
missing.sort_values(by=(f'Test (N={len(Y_test)})', 'Missing (N)'), ascending=False)


# ## LOESS Baseline Model

# In[6]:


from sklearn.preprocessing import StandardScaler
from src.train import TrainLOESSModel, TrainPolynomialModel
from src.evaluate import EvaluateBaselineModel


# In[7]:


def run(X, Y, tag, base_vals, output_path, alg='LOESS', split='Test', task_type='C', scale_func=None):
    Trains = {'LOESS': TrainLOESSModel, 'SPLINE': TrainPolynomialModel, 'POLY': TrainPolynomialModel}
    train = Trains[alg](X, Y, tag, output_path, base_vals.name, alg, task_type=task_type)
    best_param = train.bayesopt(alg=alg, verbose=0, filename=f'{alg}_{task_type}_params')
    model = train.train_model(**best_param)
    Y_preds, Y_preds_min, Y_preds_max = train.predict(model, split=split)
    if scale_func is not None:
        f = scale_func
        Y_preds[:], Y_preds_min[:], Y_preds_max[:] = f(Y_preds), f(Y_preds_min), f(Y_preds_max)
    mask = tag['split'] == split
    preds, pred_ci, labels = {split: {alg: Y_preds}}, {split: {alg: (Y_preds_min, Y_preds_max)}}, {split: Y[mask]}
    eval_base = EvaluateBaselineModel(base_vals[mask], preds, labels, output_path, pred_ci=pred_ci)
    if task_type == 'C': 
        eval_base.all_plots(alg=alg)
    elif task_type =='R':
        fig, ax = plt.subplots(figsize=(6,6))
        target_event = Y.columns[0]
        eval_base.plot_prediction(ax, alg, target_event=target_event, split=split)
        filename = f'{output_path}/figures/baseline/{target_event}_{alg}.jpg'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    return Y_preds, Y_preds_min, Y_preds_max


# In[21]:


for bt in blood_types:
    print(f'Training LOESS for {bt}')
    y = Y[[blood_types[bt]['cytopenia_name']]]
    preds, preds_min, preds_max = run(
        X, y, tag, model_data[f'baseline_{bt}_value'], output_path, task_type='C'
    )


# In[ ]:


for bt in blood_types:
    print(f'Training LOESS for {bt}')
    y = model_data[[f'target_{bt}_value']].copy()
    scaler = StandardScaler()
    y[train_mask] = scaler.fit_transform(y[train_mask])
    y[valid_mask] = scaler.transform(y[valid_mask])
    y[test_mask] = scaler.transform(y[test_mask])
    preds, preds_min, preds_max = run(
        X, y, tag, model_data[f'baseline_{bt}_value'], output_path, task_type='R', scale_func=scaler.inverse_transform
    )
