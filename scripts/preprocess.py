import tqdm
import itertools
import pandas as pd
import numpy as np
import multiprocessing as mp

from scripts.config import (root_path, sas_folder,
                            all_observations, english_lang_codes,
                            din_exclude, cancer_location_exclude,
                            systemic_cols, olis_cols, immigration_cols)

manager = mp.Manager()
shared_dict = manager.dict()
    
# Multiprocessing
def parallelize(generator, worker, processes=16):
    pool = mp.Pool(processes=processes)
    result = pool.map(worker, generator)
    pool.close()
    pool.join() # wait for all threads
    result = list(itertools.chain(*result))
    return result

def split_and_parallelize(data, worker, split_by_ikn=False, processes=16):
    if split_by_ikn: # split by patients
        generator = []
        ikn_groupings = np.array_split(data['ikn'].unique(), processes)
        for ikn_grouping in ikn_groupings:
            generator.append(data[data['ikn'].isin(ikn_grouping)])
    else:
        # splits df into x number of partitions, where x is number of processes
        generator = np.array_split(data, processes)
    return parallelize(generator, worker, processes=processes)

# sas7bdat files
def clean_sas(df, name):
    # remove empty rows
    df = df[~df.iloc[:,1:].isnull().all(axis=1)]
    if name == 'systemic':
        # create cleaned up regimen column
        df = create_clean_regimen(df.copy())
    return df

def create_clean_regimen(df):
    df['regimen'] = df['cco_regimen'].astype(str)
    df['regimen'] = df['regimen'].str[2:-1]
    df.loc[df['regimen'].str.contains('NCT'), 'regimen'] = 'TRIALS'
    df['regimen'] = df['regimen'].str.replace("*", "")
    df['regimen'] = df['regimen'].str.replace(" ", "")
    df['regimen'] = df['regimen'].str.lower()
    return df

def sas_to_csv(name, transfer=False, chunk_load=False, chunksize=10**6):
    filename = f'transfer20210810/{name}' if transfer else name
    if not chunk_load:
        df = pd.read_sas(f'{root_path}/{sas_folder}/{filename}.sas7bdat')
        df = clean_sas(df, name)
        df.to_csv(f"{root_path}/data/{name}.csv", index=False)
    else:
        chunks = pd.read_sas(f'{root_path}/{sas_folder}/{filename}.sas7bdat', chunksize=chunksize)
        for i, chunk in tqdm.tqdm(enumerate(chunks)):
            chunk = clean_sas(chunk, name)
            header = True if i == 0 else False
            chunk.to_csv(f"{root_path}/data/{name}.csv", header=header, mode='a', index=False)

# Helper functions
def clean_string(df, cols):
    # remove first two characters "b'" and last character "'"
    for col in cols:
        df.loc[:, col] = df[col].str[2:-1].values
    return df

def replace_rare_col_entries(df, cols, with_respect_to='patients', n=6, verbose=False):
    for col in cols:
        if with_respect_to == 'patients':
            # replace unique values with less than n patients to 'Other'
            counts = df.groupby(col).apply(lambda group: len(set(group['ikn'])))
        elif with_respect_to == 'rows':
            # replace unique values that appears less than n number of rows in the dataset to 'Other'
            counts = df[col].value_counts()
        replace_values = counts.index[counts < n]
        if verbose: 
            print(f'The following entries have less than {n} {with_respect_to} and will be replaced with "Other": {replace_values.tolist()}')
        df.loc[df[col].isin(replace_values), col] = 'Other'
    return df

def pandas_ffill(df):
    # uh...unfortunately pandas ffill is super slow when scaled up
    return df.ffill(axis=1)[0].values

def numpy_ffill(df):
    # courtesy of stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    arr = df.values.astype(float)
    num_rows, num_cols = arr.shape
    mask = np.isnan(arr)
    indices = np.where(~mask, np.arange(num_cols), 0)
    np.maximum.accumulate(indices, axis=1, out=indices)
    arr[mask] = arr[np.nonzero(mask)[0], indices[mask]]
    return arr[:, -1]

def group_observations(observations, freq_map):
    grouped_observations = {}
    for obs_code, obs_name in observations.items():
        if obs_name in grouped_observations:
            grouped_observations[obs_name].append(obs_code)
        else:
            grouped_observations[obs_name] = [obs_code]

    for obs_name, obs_codes in grouped_observations.items():
        # sort observation codes based on their number of occurences / observation frequencies
        grouped_observations[obs_name] = freq_map[obs_codes].sort_values(ascending=False).index.tolist()
   
    return grouped_observations
    
# Systemic (chemo) data
def filter_systemic_data(df, regimens, mapping):
    df = clean_string(df, ['ikn', 'din', 'intent_of_systemic_treatment', 'inpatient_flag'])
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df = df.sort_values(by='visit_date') # order the data dataframe by date
    df = df[df['inpatient_flag'] == 'N'] # remove chemo treatment recieved as an inpatient
    df = df[df['regimen'].isin(regimens)] # keep only selected reigments
    df.loc[:, 'regimen'] = df['regimen'].map(mapping).values # change regimen name to the correct mapping
    df = df[~df['din'].isin(din_exclude)] # remove dins in din_exclude
    df = df[systemic_cols] # keep only selected columns
    df = df.drop_duplicates()
    return df

def merge_intervals(df):
    # Merges small intervals into a 4 day cycle, or to the row below/above that has interval greater than 4 days
    df = df.reset_index(drop=True)
    remove_indices = []
    for i in range(len(df)):
        if df.loc[i, 'chemo_interval'] > pd.Timedelta('3 days') or pd.isnull(df.loc[i, 'chemo_interval']):
            continue
        if i == len(df)-1:
            # merge with the row above if last entry of a regimen
            if i == 0: # if its the very first entry of the whole dataframe, leave it as is
                continue 
            df.loc[i-1, 'visit_date'] = df.loc[i, 'visit_date']
            df.loc[i-1, 'chemo_interval'] = df.loc[i, 'chemo_interval'] + df.loc[i-1, 'chemo_interval'] 
        elif df.loc[i, 'regimen'] != df.loc[i+1, 'regimen']:
            # merge with the row above if last entry of an old regimen
            if i == 0: # if its the very first entry of the whole dataframe, leave it as is
                continue 
            df.loc[i-1, 'visit_date'] = df.loc[i, 'visit_date']
            df.loc[i-1, 'chemo_interval'] = df.loc[i, 'chemo_interval'] + df.loc[i-1, 'chemo_interval'] 
        else:
            # merge with the row below
            df.loc[i+1, 'prev_visit'] = df.loc[i, 'prev_visit']
            df.loc[i+1, 'chemo_interval'] = df.loc[i, 'chemo_interval'] + df.loc[i+1, 'chemo_interval']
        remove_indices.append(i)
    df = df.drop(index=remove_indices)
    return df

def systemic_worker(partition):
    chemo = []
    num_chemo_sess_eliminated = 0
    for ikn, df in tqdm.tqdm(partition.groupby('ikn')):
        # include prev visit and chemo interval
        df['prev_visit'] = df['visit_date'].shift()
        df.loc[~df['regimen'].eq(df['regimen'].shift()), 'prev_visit'] = pd.NaT # break off when chemo regimen changes
        df['chemo_interval'] = df['visit_date'] - df['prev_visit']

        # Merges small intervals into a 4 day cycle, or to the row below/above that has interval greater than 4 days
        # NOTE: for patient XXXXXXXX (same with XXXXXXX), they have single chemo sessions that gets eliminated. 
        num_chemo_sess_eliminated += len(df[df['prev_visit'].isnull() & ~df['regimen'].eq(df['regimen'].shift(-1))])
        df = df[~df['chemo_interval'].isnull()]
        if df.empty:
            continue
        df = merge_intervals(df)
        if df.empty:
            # most likely patient (e.g. XXXXXXXX) had consecutive 1 day interval chemo sessions 
            # that totaled less than 5 days
            continue

        # identify chemo cycle number (resets when patient undergoes new chemo regimen or there is a 60 day gap)
        mask = df['regimen'].eq(df['regimen'].shift()) & (df['chemo_interval'].shift() < pd.Timedelta('60 days'))
        group = (mask==False).cumsum()
        df['chemo_cycle'] = mask.groupby(group).cumsum()+1

        # identify if this is the first chemo cycle of a new regimen immediately after the old one
        # WARNING: currently does not account for those single chemo sessios (that gets eliminated above )
        mask = ~df['regimen'].eq(df['regimen'].shift()) & \
                (df['prev_visit']-df['visit_date'].shift() < pd.Timedelta('60 days'))
        mask.iloc[0] = False
        df['immediate_new_regimen'] = mask

         # convert to list and combine for faster performance
        chemo.extend(df.values.tolist())
        
    # print(f'Number of chemo sessions eliminated: {num_chemo_sess_eliminated}')
    return chemo

def clean_cancer_and_demographic_data(chemo_df, col_arrangement, verbose=False):
    # do not include patients under 18
    num_patients_under_18 = len(set(chemo_df.loc[chemo_df['age'] < 18, 'ikn']))
    if verbose:
        print(f"Removing {num_patients_under_18} patients under 18")
    chemo_df = chemo_df[~(chemo_df['age'] < 18)]

    # clean up some features
    chemo_df.loc[:, 'body_surface_area'] = chemo_df['body_surface_area'].replace(0, np.nan)
    chemo_df.loc[:, 'body_surface_area'] = chemo_df['body_surface_area'].replace(-99, np.nan)

    # clean up morphology and topography code features
    cols = ['curr_morph_cd', 'curr_topog_cd']
    chemo_df.loc[:, cols] = chemo_df[cols].replace('*U*', np.nan)
    for col in cols: chemo_df.loc[:, col] = chemo_df[col].str[:3] # only keep first three characters - the rest are for specifics
                                                           # e.g. C50 Breast: C501 Central portion, C504 Upper-outer quadrant, etc
    mask = chemo_df['curr_topog_cd'].isin(cancer_location_exclude) # remove entries in cancer_location_exclude
    if verbose:
        print(f"Removing {chemo_df.loc[mask, 'ikn'].nunique()} patients with blood cancers")
    chemo_df = chemo_df[~mask]
    chemo_df = replace_rare_col_entries(chemo_df, cols, verbose=verbose)
                                                       
    # rearrange the columns
    chemo_df = chemo_df[col_arrangement]
    return chemo_df

def load_chemo_df(main_dir):
    dtype = {'ikn': str, 'lhin_cd': str, 'curr_morph_cd': str}
    chemo_df = pd.read_csv(f'{main_dir}/data/chemo_processed.csv', dtype=dtype)
    chemo_df['visit_date'] = pd.to_datetime(chemo_df['visit_date'])
    chemo_df['prev_visit'] = pd.to_datetime(chemo_df['prev_visit'])
    chemo_df['chemo_interval'] = pd.to_timedelta(chemo_df['chemo_interval'])
    return chemo_df

# Olis (blood work/lab test) data
def filter_olis_data(olis):
    # convert string column into timestamp column
    olis.loc[:, 'ObservationDateTime'] = pd.to_datetime(olis['ObservationDateTime']).values
    olis.loc[:, 'ObservationDateTime'] = olis['ObservationDateTime'].dt.floor('D').values # keep only the date, not time
    olis.loc[:, 'ObservationReleaseTS'] = pd.to_datetime(
                                            olis['ObservationReleaseTS'], errors='coerce').values # there are out of bound timestamp errors
    
    # remove rows with blood count null or neg values
    olis['value'] = olis['value'].astype(float)
    olis = olis[~olis['value'].isnull() | ~(olis['value'] < 0)]
    
    # remove duplicate rows
    subset = ['ikn','ObservationCode', 'ObservationDateTime', 'value']
    olis = olis.drop_duplicates(subset=subset) 
    
    # if only the patient id, blood, and observation timestamp are duplicated (NOT the blood count value), 
    # keep the most recently RELEASED row
    olis = olis.sort_values(by='ObservationReleaseTS')
    subset = ['ikn','ObservationCode', 'ObservationDateTime']
    olis = olis.drop_duplicates(subset=subset, keep='last')
    
    return olis

def olis_worker(partition, main_dir):
    chemo_df = load_chemo_df(main_dir)
    
    # only keep rows where patients exist in both dataframes
    chemo_df = chemo_df[chemo_df['ikn'].isin(partition['ikn'])]
    partition = partition[partition['ikn'].isin(chemo_df['ikn'])]

    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        olis_subset = partition[partition['ikn'] == ikn]
        
        for chemo_idx, chemo_row in chemo_group.iterrows():
            
            # see if there any blood count data within the target dates
            earliest_date = chemo_row['prev_visit'] - pd.Timedelta('5 days')
            # set limit to 28 days after chemo administration or the day of next chemo administration, 
            # whichever comes first
            latest_date = min(chemo_row['visit_date'], chemo_row['prev_visit'] + pd.Timedelta('28 days'))
            tmp = olis_subset[(earliest_date <= olis_subset['ObservationDateTime']) & 
                              (latest_date >= olis_subset['ObservationDateTime'])]
            tmp['chemo_idx'] = chemo_idx
            tmp['days_after_chemo'] = (tmp['ObservationDateTime'] - chemo_row['prev_visit']).dt.days
            result.extend(tmp[['ObservationCode', 'chemo_idx', 'days_after_chemo', 'value']].values.tolist())
                
    return result

def prefilter_olis_data(chunk, chemo_ikns, select_blood_types=None):
    """
    Args:
        keep_blood_types (list or None): list of observation codes associated with specific blood work tests we want to keep
    """
    # organize and format columns
    chunk = chunk[olis_cols]
    chunk = clean_string(chunk, ['ikn', 'ObservationCode', 'ReferenceRange', 'Units'])
    
    # filter patients not in chemo_df
    chunk = chunk[chunk['ikn'].isin(chemo_ikns)]
    
    if select_blood_types:
        # filter rows with excluded blood types
        chunk = chunk[chunk['ObservationCode'].isin(select_blood_types)]
    
    chunk = filter_olis_data(chunk)
    return chunk
                          
def postprocess_olis_data(chemo_df, olis_df, observations=all_observations, days_range=range(-5,29)):
    # fill up the blood count dataframes for each blood type
    mapping = {obs_code: pd.DataFrame(index=chemo_df.index, columns=days_range) for obs_code in observations}
    olis_df = olis_df[olis_df['observation_code'].isin(observations)] # exclude obs codes not in selected observations
    for obs_code, obs_group in tqdm.tqdm(olis_df.groupby('observation_code')):
        for day, day_group in obs_group.groupby('days_after_chemo'):
            chemo_indices = day_group['chemo_idx'].values.astype(int)
            obs_count_values = day_group['observation_count'].values.astype(float)
            mapping[obs_code].loc[chemo_indices, int(day)] = obs_count_values
            
    # group together obs codes with same obs name
    freq_map = olis_df['observation_code'].value_counts()
    grouped_observations = group_observations(observations, freq_map)
    
    # get baseline observation counts, combine it to chemo_df 
    num_rows = {'chemo_df': len(chemo_df)} # Number of non-missing rows
    for obs_name, obs_codes in tqdm.tqdm(grouped_observations.items()):
        col = f'baseline_{obs_name}_count'
        for i, obs_code in enumerate(obs_codes):
            # forward fill observation counts from day -5 to day 0
            values = numpy_ffill(mapping[obs_code][range(-5,1)])
            values = pd.Series(values, index=chemo_df.index)
            """
            Initialize using observation counts with the most frequent observations
            This takes priority when there is conflicting count values among the grouped observations
            Fill any missing counts using the observations counts with the next most frequent observations, and so on

            e.g. an example of a group of similar observations
            'alanine_aminotransferase': ['1742-6',  #                    (n=BIG) <- initalize
                                         '1744-2',  # without vitamin B6 (n=MED) <- fill na, does not overwrite '1742-6'
                                         '1743-4'], # with vitamin B6    (n=SMALL)  <- fill na, does not overwrite '1742-6', '1744-2'
            """
            chemo_df[col] = values if i == 0 else chemo_df[col].fillna(values)
        num_rows[obs_name] = sum(~chemo_df[col].isnull())
    missing_df = pd.DataFrame(num_rows, index=['num_rows_not_null'])
    missing_df = missing_df.T.sort_values(by='num_rows_not_null')
       
    return mapping, missing_df

# Esas (questionnaire) data
def preprocess_esas(chemo_ikns):
    esas = pd.read_csv(f"{root_path}/data/esas.csv")
    esas = esas.rename(columns={'esas_value': 'severity', 'esas_resourcevalue': 'symptom'})
    esas = clean_string(esas, ['ikn', 'severity', 'symptom'])
    
    # filter out patients not in chemo_df
    esas = esas[esas['ikn'].isin(chemo_ikns)]
    
    # remove duplicate rows
    subset = ['ikn', 'surveydate', 'symptom', 'severity']
    esas = esas.drop_duplicates(subset=subset) 
    
    # remove Activities & Function as they only have 4 samples
    esas = esas[~(esas['symptom'] == 'Activities & Function:')]
    
    return esas

def esas_worker(partition):
    esas = shared_dict['esas_chunk']
    result = []
    for ikn, group in partition.groupby('ikn'):
        esas_specific_ikn = esas[esas['ikn'] == ikn]
        for idx, chemo_row in group.iterrows():
            visit_date = chemo_row['visit_date']
            # even if survey date is like a month ago, we will be forward filling the entries to the visit date
            esas_most_recent = esas_specific_ikn[esas_specific_ikn['surveydate'] < visit_date]
            if not esas_most_recent.empty:
                symptoms = list(esas_most_recent['symptom'].unique())
                for symptom in symptoms:
                    esas_specific_symptom = esas_most_recent[esas_most_recent['symptom'] == symptom]
                    # last item is always the last observed grade (we've already sorted by date)
                    result.append((idx, symptom, esas_specific_symptom['severity'].iloc[-1]))
    return result

def get_esas_responses(chemo_df, esas_chunks, len_chunks=18):
    result = [] 
    for i, chunk in tqdm.tqdm(enumerate(esas_chunks), total=len_chunks):
        # organize and format columns
        chunk['surveydate'] = pd.to_datetime(chunk['surveydate'])
        chunk['severity'] = chunk['severity'].astype(int)
        chunk = chunk.sort_values(by='surveydate') # sort by date

        # filter out patients not in esas chunk
        filtered_chemo_df = chemo_df[chemo_df['ikn'].isin(chunk['ikn'])]

        # get results
        shared_dict['esas_chunk'] = chunk
        chunk_result = split_and_parallelize(filtered_chemo_df, esas_worker, split_by_ikn=True, processes=32)
        result += chunk_result
    
    return result

def postprocess_esas_responses(esas_df):
    esas_df = esas_df.sort_values(by=['index', 'symptom', 'severity'])

    # remove duplicates - keep the more severe entry
    # TODO: remove duplicates by using surveydate
    mask = esas_df[['index', 'symptom']].duplicated(keep='last')
    print(f'{sum(mask)} duplicate entries among total {len(esas_df)} entries')
    esas_df = esas_df[~mask]

    # make each symptom its own column, with severity as its entry value
    esas_df = esas_df.pivot(index='index', columns='symptom')['severity']
    
    return esas_df

# ECOG (body function grade) data
def filter_ecog_data(ecog, chemo_ikns):
    # organize and format columns
    ecog = ecog.drop(columns=['ecog_resourcevalue'])
    ecog = ecog.rename(columns={'ecog_value': 'ecog_grade'})
    ecog = clean_string(ecog, ['ikn', 'ecog_grade'])
    ecog['ecog_grade'] = ecog['ecog_grade'].astype(int)
    ecog['surveydate'] = pd.to_datetime(ecog['surveydate'])
    
    # filter patients not in chemo_df
    ecog = ecog[ecog['ikn'].isin(chemo_ikns)]
    
    # sort by date
    ecog = ecog.sort_values(by='surveydate')
    
    return ecog

def ecog_worker(partition):
    ecog = shared_dict['ecog']
    result = []
    for ikn, group in tqdm.tqdm(partition.groupby('ikn'), position=0):
        ecog_specific_ikn = ecog[ecog['ikn'] == ikn]
        for idx, chemo_row in group.iterrows():
            visit_date = chemo_row['visit_date']
            ecog_most_recent = ecog_specific_ikn[ecog_specific_ikn['surveydate'] < visit_date]
            if not ecog_most_recent.empty:
                # last item is always the last observed grade (we've already sorted by date)
                result.append((idx, ecog_most_recent['ecog_grade'].iloc[-1]))
    return result

# Immigration
def filter_immigration_data(immigration):
    immigration = clean_string(immigration, ['ikn', 'official_language'])
    immigration['speaks_english'] = immigration['official_language'].isin(english_lang_codes) # able to speak english
    immigration['is_immigrant'] = True
    immigration = immigration[immigration_cols]
    return immigration

# ED/H/D - get datasets and define workers
def get_y3():
    y3 = pd.read_csv(f'{root_path}/data/y3.csv', dtype=str) # Alive/Dead status and death date for ~XXXXXX patients
    print('Completed Loading Y3 Dataset')
    y3 = clean_string(y3, ['vital_status_cd', 'ikn'])
    y3['d_date'] = pd.to_datetime(y3['dthdate'])

    # remove rows with conflicting information
    mask1 = (y3['vital_status_cd'] == 'A') & ~y3['dthdate'].isnull() # vital status says Alive but deathdate exists
    mask2 = (y3['vital_status_cd'] == 'D') & y3['dthdate'].isnull() # vital status says Dead but deathdate does not exist
    y3 = y3[~(mask1 | mask2)]
    return y3

def chemo_worker(partition):
    values = []
    for ikn, group in tqdm.tqdm(partition.groupby('ikn')):
        start_date = group['visit_date'].iloc[0]
        group['days_since_starting_chemo'] = (group['visit_date'] - start_date).dt.days
        group['days_since_true_prev_chemo'] = (group['visit_date'] - group['visit_date'].shift()).dt.days
        
        # keep only the first chemo visit of a given week
        keep_indices = []
        tmp = group
        while not tmp.empty:
            first_chemo_idx = tmp.index[0]
            keep_indices.append(first_chemo_idx)
            first_visit_date = group['visit_date'].loc[first_chemo_idx]
            tmp = tmp[tmp['visit_date'] >= first_visit_date + pd.Timedelta('7 days')]
        group = group.loc[keep_indices]
        group['days_since_prev_chemo'] = (group['visit_date'] - group['visit_date'].shift()).dt.days
        
        values.extend(group.values.tolist())
    return values

def get_systemic(regimens_keep=None, replace_rare_regimens=False):
    regimen_name_mapping = {'paclicarbo': 'crbppacl'}

    df = pd.read_csv(f'{root_path}/data/systemic.csv', dtype=str)
    print('Completed Loading Systemic Dataset')
    df = clean_string(df, ['ikn', 'din', 'intent_of_systemic_treatment', 'inpatient_flag'])
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df = df.sort_values(by='visit_date') # order the dataframe by date

    # filter regimens
    col = 'regimen'
    df = df[~df[col].isnull()] # filter out rows with no regimen data 
    for old_name, new_name in regimen_name_mapping.items():
        df.loc[df[col] == old_name, col] = new_name # change regimen name to the correct mapping
    if regimens_keep:
        df = df[df[col].isin(regimens_keep)] # keep only selected reigments
    if replace_rare_regimens:
        # replace regimens with less than 6 patients to 'Other'
        df = replace_rare_col_entries(df, ['regimen'])

    # keep only one chemo per week, get days since chemo
    days_cols = ['days_since_starting_chemo', 'days_since_true_prev_chemo', 'days_since_prev_chemo']
    values = split_and_parallelize(df, chemo_worker, split_by_ikn=True)
    cols = df.columns.tolist() + days_cols
    df = pd.DataFrame(values, columns=cols)

    # remove chemo treatment recieved as an inpatient
    print(f"There are {sum(df['inpatient_flag'] == 'Y')} inpatient chemo treatment and {sum(df['inpatient_flag'].isnull())} unknown inpatient status out of {len(df)} total chemo treatments")
    df = df[df['inpatient_flag'] == 'N']

    df = df[systemic_cols + days_cols] # keep only selected columns
    df = df.drop_duplicates(subset=systemic_cols)
    return df

def observation_worker(partition, main_dir, days_ago=5, filename='chemo_processed'):
    chemo_df = pd.read_csv(f'{main_dir}/data/{filename}.csv', dtype=str)
    chemo_df = chemo_df.set_index('index')
    chemo_df['visit_date'] = pd.to_datetime(chemo_df['visit_date'])
    
    # only keep rows where patients exist in both dataframes
    chemo_df = chemo_df[chemo_df['ikn'].isin(partition['ikn'])]
    partition = partition[partition['ikn'].isin(chemo_df['ikn'])]

    result = []
    for ikn, chemo_group in tqdm.tqdm(chemo_df.groupby('ikn')):
        olis_subset = partition[partition['ikn'] == ikn]
        
        for chemo_idx, chemo_row in chemo_group.iterrows():
            # see if there any blood count data since prev chemo up to x days ago 
            days_since_prev_chemo = float(chemo_row['days_since_prev_chemo'])
            if not pd.isnull(days_since_prev_chemo):
                days_ago = min(days_ago, days_since_prev_chemo)
            latest_date = chemo_row['visit_date']
            earliest_date = latest_date - pd.Timedelta(days=days_ago)
            tmp = olis_subset[(olis_subset['ObservationDateTime'] >= earliest_date) & 
                              (olis_subset['ObservationDateTime'] <= latest_date)]
            tmp['chemo_idx'] = chemo_idx
            tmp['days_after_chemo'] = (tmp['ObservationDateTime'] - latest_date).dt.days
            result.extend(tmp[['ObservationCode', 'chemo_idx', 'days_after_chemo', 'value']].values.tolist())
                
    return result