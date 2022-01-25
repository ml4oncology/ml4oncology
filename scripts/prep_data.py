import pandas as pd

from scripts.config import (root_path, blood_types,
                            observation_cols, esas_ecog_cols)
from scripts.preprocess import (replace_rare_col_entries)

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

class PrepData:
    """
    Prepare the data for model training
    """
    def __init__(self):
        self.observation_cols = observation_cols
        self.imp_obs = None # observation imputer
        self.imp_ee = None # esas ecog imputer
        self.scaler = None # normalizer
        self.bsa_mean = None # body surface area mean (imputer)
    
    def clip_outliers(self, data, cols=None, lower_percentile=0.01, upper_percentile=0.99):
        """
        clip the upper and lower percentiles for the columns indicated below
        """
        if not cols:
            cols = data.columns
            cols = cols[cols.str.contains('count') & ~cols.str.contains('is_missing')]
            cols = cols.drop([f'target_{bt}_count' for bt in blood_types], errors='ignore')
            cols = cols.tolist() + ['body_surface_area']

        thresh = data[cols].quantile([lower_percentile, upper_percentile])
        data[cols] = data[cols].clip(lower=thresh.loc[lower_percentile], upper=thresh.loc[upper_percentile], axis=1)
        return data, thresh

    def dummify_data(self, data):
        # make categorical columns into one-hot encoding
        return pd.get_dummies(data)

    def replace_missing_body_surface_area(self, data):
        # replace missing body surface area with the mean based on sex
        bsa = 'body_surface_area'
        mask_F = data['sex'] == 'F' if 'sex' in data.columns else data['sex_F'] == 1
        mask_M = data['sex'] == 'M' if 'sex' in data.columns else data['sex_M'] == 1
        
        if self.bsa_mean is None:
            self.bsa_mean = {'female': data.loc[mask_F, bsa].mean(), 'male': data.loc[mask_M, bsa].mean()}
            
        data.loc[mask_F, bsa] = data.loc[mask_F, bsa].fillna(self.bsa_mean['female'])
        data.loc[mask_M, bsa] = data.loc[mask_M, bsa].fillna(self.bsa_mean['male'])

        return data

    def impute_observations(self, data):
        # mean impute missing observation data
        if self.imp_obs is None:
            self.imp_obs = SimpleImputer() 
            data[self.observation_cols] = self.imp_obs.fit_transform(data[self.observation_cols])
        else:
            data[self.observation_cols] = self.imp_obs.transform(data[self.observation_cols])
        return data

    def impute_esas_ecog(self, data):
        # mode impute missing esas and ecog data
        if self.imp_ee is None:
            self.imp_ee = SimpleImputer(strategy='most_frequent') 
            data[esas_ecog_cols] = self.imp_ee.fit_transform(data[esas_ecog_cols])
        else:
            data[esas_ecog_cols] = self.imp_ee.transform(data[esas_ecog_cols])
        return data
    
    def extra_norm_cols(self):
        """
        You can overwrite this to use custom columns used for GRU, ED-H-D events, etc
        The following columns are used for ML models for cytopenia detection
        """
        return ['chemo_cycle', 'chemo_interval', 'cycle_lengths']
        
    def normalize_data(self, data):
        norm_cols = ['age', 'body_surface_area', 'line_of_therapy'] + esas_ecog_cols + self.observation_cols
        norm_cols += self.extra_norm_cols()

        if self.scaler is None:
            self.scaler = MinMaxScaler() 
            data[norm_cols] = self.scaler.fit_transform(data[norm_cols])
        else:
            data[norm_cols] = self.scaler.transform(data[norm_cols])
        return data
    
    def clean_feature_target_cols(self, feature_cols, target_cols):
        """
        You can overwrite this to do custom operations for GRU, ED-H-D, etc
        """
        feature_cols = feature_cols.drop('ikn')
        return feature_cols, target_cols

    def split_data(self, data, target_keyword='target', impute=True, normalize=True, convert_to_float=True, verbose=True):
        """
        Split data into training, validation and test sets based on patient ids
        Impute and Normalize the datasets
        """
        # convert dtype object to float
        if convert_to_float: data = data.astype(float)

        # create training set
        gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
        train_idxs, test_idxs = next(gss.split(data, groups=data['ikn']))
        train_data = data.iloc[train_idxs]
        test_data = data.iloc[test_idxs]

        # crate validation and testing set
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        valid_idxs, test_idxs = next(gss.split(test_data, groups=test_data['ikn']))

        valid_data = test_data.iloc[valid_idxs]
        test_data = test_data.iloc[test_idxs]

        # sanity check - make sure there are no overlap of patients in the splits
        assert(sum(valid_data['ikn'].isin(set(train_data['ikn']))) + sum(valid_data['ikn'].isin(set(test_data['ikn']))) + 
               sum(train_data['ikn'].isin(set(valid_data['ikn']))) + sum(train_data['ikn'].isin(set(test_data['ikn']))) + 
               sum(test_data['ikn'].isin(set(train_data['ikn']))) + sum(test_data['ikn'].isin(set(valid_data['ikn']))) == 0)
        if verbose:
            print(f'Size of splits: Train:{len(train_data)}, Val:{len(valid_data)}, Test:{len(test_data)}')
            print(f"Number of patients: Train:{len(set(train_data['ikn']))}, Val:{len(set(valid_data['ikn']))}, Test:{len(set(test_data['ikn']))}")

        if impute:
            # IMPORTANT: always make sure train data is done first, so imputer is fit to the training set
            # mean impute the body surface area based on sex
            train_data = self.replace_missing_body_surface_area(train_data.copy())
            valid_data = self.replace_missing_body_surface_area(valid_data.copy())
            test_data = self.replace_missing_body_surface_area(test_data.copy())
            if verbose:
                print(f"Body Surface Area Mean - Female:{self.bsa_mean['female'].round(4)}, Male:{self.bsa_mean['male'].round(4)}")

            # mean impute the blood work data
            train_data = self.impute_observations(train_data)
            valid_data = self.impute_observations(valid_data)
            test_data = self.impute_observations(test_data)

            # mode impute the esas and ecog data
            train_data = self.impute_esas_ecog(train_data)
            valid_data = self.impute_esas_ecog(valid_data)
            test_data = self.impute_esas_ecog(test_data)

        if normalize:
            # normalize the splits based on training data
            train_data = self.normalize_data(train_data.copy())
            valid_data = self.normalize_data(valid_data.copy())
            test_data = self.normalize_data(test_data.copy())

        # split into input features and target labels
        cols = data.columns
        feature_cols = cols[~cols.str.contains(target_keyword)]
        target_cols = cols[cols.str.contains(target_keyword)]
        feature_cols, target_cols = self.clean_feature_target_cols(feature_cols, target_cols)
        X_train, X_valid, X_test = train_data[feature_cols], valid_data[feature_cols], test_data[feature_cols]
        Y_train, Y_valid, Y_test = train_data[target_cols], valid_data[target_cols], test_data[target_cols]

        return [(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)]
    
    def regression_to_classification(self, target, threshold):
        """
        Convert regression labels (last blood count value) to 
        classification labels (if last blood count value is below corresponding threshold)
        """
        for blood_type in blood_types:
            target[f'{blood_type} < {threshold[blood_type]}'] = target[f'target_{blood_type}_count'] < threshold[blood_type]
            target = target.drop(columns=[f'target_{blood_type}_count'])
        return target

    def upsample(self, X, Y, target_col='platelet < 75', n=4):
        # upsample the data by increasing target_col positive examples by n folds
        indices = Y.index.tolist() + Y[Y[target_col]].index.tolist()*(n-1)
        Y = Y.loc[indices]
        X = X.loc[indices]
        return X, Y

    def get_label_distribution(self, Y_train, Y_valid=None, Y_test=None, save=False, save_path=''):
        distributions = [pd.DataFrame([Y_train[col].value_counts() for col in Y_train.columns])]
        cols = ['Train']
        if Y_valid is not None:
            distributions.append(pd.DataFrame([Y_valid[col].value_counts() for col in Y_valid.columns]))
            cols.append('Valid')
        if Y_test is not None:
            distributions.append(pd.DataFrame([Y_test[col].value_counts() for col in Y_test.columns]))
            cols.append('Test')
        Y_distribution = pd.concat(distributions, axis=1)
        cols = pd.MultiIndex.from_product([cols, ['False', 'True']])
        Y_distribution.columns = cols
        if save:
            Y_distribution.to_csv(save_path, index=False)
        return Y_distribution
    
class PrepDataEDHD(PrepData):   
    
    def extra_norm_cols(self):
        return ['visit_month', 'num_prior_EDs', 'num_prior_Hs',
                'days_since_prev_H', 'days_since_prev_ED', 
                'days_since_starting_chemo', 'days_since_prev_chemo']
    
    def get_data(self, main_dir, target_keyword, rem_no_obs_visits=False, rem_days_since_prev_chemo=True, verbose=False):
        """
        input:                               -->        MODEL         -->            target:
        regimen                                                                      ED/H/D Events within next 3-x days              
        intent of systemic treatment, line of therapy                                - exclude 1st (day of admin) and 2nd day
        lhin cd, curr morth cd, curr topog cd, age, sex                              ED/H event causes
        body surface area, esas/ecog features
        blood work prior to visit date
        days since ED/H event occured
        cause for prev event
        
        Args:
            rem_no_obs_visits (bool): if True, remove chemo visits with no blood work/labaratory test observations
            rem_days_since_prev_chemo (bool): if True, remove the predictor days_since_prev_chemo, using only days_since_true_prev_chemo
        """
        df = pd.read_csv(f'{main_dir}/data/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
        df = df.set_index('index')
        df.index.name = ''
        
        # remove sessions where no observations were measured
        if rem_no_obs_visits:
            mask = df[observation_cols].isnull().all(axis=1)
            df = df[~mask]

        # fill null values with 0 or max value
        df['line_of_therapy'] = df['line_of_therapy'].fillna(0) # the nth different chemotherapy taken
        df['num_prior_EDs'] = df['num_prior_EDs'].fillna(0)
        df['num_prior_Hs'] = df['num_prior_Hs'].fillna(0)
        for col in ['days_since_prev_chemo', 'days_since_true_prev_chemo']:
            df[col] = df[col].fillna(df[col].max())

        # reduce sparse matrix by replacing rare col entries with less than 6 patients with 'Other'
        cols = ['regimen', 'curr_morph_cd', 'curr_topog_cd']
        df = replace_rare_col_entries(df, cols, verbose=verbose)
        
        # get visit month 
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        df['visit_month'] = df['visit_date'].dt.month
        
        # create features for missing entries
        cols_with_nan = df.columns[df.isnull().any()]
        df[cols_with_nan + '_is_missing'] = df[cols_with_nan].isnull()
        
        # create column for acute care (ACU = ED + H) and treatment related acute care (TR_ACU = TR_ED + TR_H)
        df['ACU'+target_keyword] = df['ED'+target_keyword] | df['H'+target_keyword] 
        df['TR_ACU'+target_keyword] = df['TR_ED'+target_keyword] | df['TR_H'+target_keyword] 

        cols = df.columns
        drop_columns = cols[cols.str.contains('within')]
        drop_columns = drop_columns[~drop_columns.str.contains(target_keyword)]
        drop_columns = ['visit_date'] + drop_columns.tolist()
        df = df.drop(columns=drop_columns)
        
        if rem_days_since_prev_chemo:
            df['days_since_prev_chemo'] = df['days_since_true_prev_chemo']
            df = df.drop(columns='days_since_true_prev_chemo')
            
        return df
