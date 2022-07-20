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
# Paths
root_path = 'XXXXX'
share_path = 'XXXXX'
sas_folder = 'XXXXX'
sas_folder2 = 'XXXXX'
regiments_folder = 'XXXXX'
cyto_folder = 'CYTOPENIA' # Cytopenia folder = 'CYTOPENIA'
acu_folder = 'PROACCT' # Acute care use folder = 'PROACCT' (PRediction of Acute Care use during Cancer Treatment)
can_folder = 'CAN' # Cisplatin-associated nephrotoxicity folder = 'CAN'
symp_folder = 'SYMPTOM' # Symptom Deteroriation folder = 'SYMPTOM'
death_folder = 'DEATH' # Death folder = 'DEATH'

# Date to temporally split data into developement and test cohort
split_date = '2017-06-30'

# Main Blood Types and Low Blood Count Thresholds
blood_types = {'neutrophil': {'cytopenia_threshold': 1.5,  # cytopenia = low blood count
                              'cytopenia_name': 'Neutropenia',
                              'unit': '10^9/L'},
               'hemoglobin': {'cytopenia_threshold': 100,
                              'cytopenia_name': 'Anemia',
                              'unit': 'g/L'}, 
               'platelet': {'cytopenia_threshold': 75,
                            'cytopenia_name': 'Thrombocytopenia',
                            'unit': '10^9/L'}}

cytopenia_gradings = {'Grade 2': {'Neutropenia': 1.5, 'Anemia': 100, 'Thrombocytopenia': 75, 'color': (236, 112, 20)},
                      'Grade 3': {'Neutropenia': 1.0, 'Anemia': 80, 'Thrombocytopenia': 50, 'color': (204, 76, 2)},
                      'Grade 4': {'Neutropenia': 0.5, 'Thrombocytopenia': 25, 'color': (102, 37, 6)}}

# All Lab Tests and Blood Work
all_observations = { '1742-6': 'alanine_aminotransferase',
                     '1744-2': 'alanine_aminotransferase', # without vitamin B6
                     '1743-4': 'alanine_aminotransferase', # with vitamin B6

                     '704-7': 'basophil', # automated count
                     '705-4': 'basophil', # manual count
                     '26444-0': 'basophil',

                     '14629-0': 'bilirubin_direct', # (direct bilirubin = glucoronidated bilirubin + albumin bound bilirubin)
                     '29760-6': 'bilirubin_direct', # glocoronidated only
                     '14630-8': 'bilirubin_indirect',
                     '14631-6': 'bilirubin', # (total bilirubin = direct bilirubin + indirect bilirubin)

                     '711-2': 'eosinophil', # automated count
                     '712-0': 'eosinophil', # manual count
                     '26449-9': 'eosinophil',

                     '788-0': 'erythrocyte_distribution_width_ratio', # automated count
                     '30385-9': 'erythrocyte_distribution_width_ratio',
                     '21000-5': 'erythrocyte_distribution_width_entitic_volume', # automated count
                     '30384-2': 'erythrocyte_distribution_width_entitic_volume',
                     '789-8': 'erythrocyte', # automated count
                     '790-6': 'erythrocyte', # manual count
                     '26453-1': 'erythrocyte',
                     '787-2': 'erythrocyte_MCV', # (MCV: mean corpuscular volume) automated count
                     '30428-7': 'erythrocyte_MCV',
                     '786-4': 'erythrocyte_MCHC', # (MCHC: mean corpuscular hemoglobin concentration) automated count
                     '28540-3': 'erythrocyte_MCHC',
                     '785-6': 'erythrocyte_MCH', # (MCH: mean corpuscular hemoglobin) automated count
                     '28539-5': 'erythrocyte_MCH', 

                     '4544-3': 'hematocrit', # automated count
                     '71833-8': 'hematocrit', # pure fractioon (aka no units), automated count
                     '20570-8': 'hematocrit', 

                     '718-7': 'hemoglobin',
                     '20509-6': 'hemoglobin', # calculation
                     '4548-4': 'hemoglobin_A1c_total_hemoglobin_ratio',
                     '71875-9': 'hemoglobin_A1c_total_hemoglobin_ratio', # pure mass fraction
                     '17855-8': 'hemoglobin_A1c_total_hemoglobin_ratio', # calculation
                     '17856-6': 'hemoglobin_A1c_total_hemoglobin_ratio', # HPLC
                     '59261-8': 'hemoglobin_A1c_total_hemoglobin_ratio', # IFCC

                     '6690-2': 'leukocyte', # automated count',
                     '804-5': 'leukocyte', # manual count',
                     '26464-8': 'leukocyte',

                     '731-0': 'lymphocyte', # automated count
                     '732-8': 'lymphocyte', # manual count
                     '26474-7': 'lymphocyte', 

                     '742-7': 'monocyte', # automated count
                     '743-5': 'monocyte', # manual count
                     '26484-6': 'monocyte', 

                     '751-8': 'neutrophil', # automated count
                     '753-4': 'neutrophil', # manual count
                     '26499-4': 'neutrophil',

                     '32623-1': 'platelet_mean_volume', # automated count
                     '28542-9': 'platelet_mean_volume',
                     '777-3': 'platelet', # in blood, automated count
                     '26515-7': 'platelet', # in blood
                     '13056-7': 'platelet', # in plasma, automated count

                     '2823-3': 'potassium', # in serum/plasma
                     '6298-4': 'potassium', # in blood
                     '39789-3': 'potassium', # in venous blood

                     '1751-7': 'albumin',
                     '1920-8': 'asparate',
                     '2000-8': 'calcium',
                     '14682-9': 'creatinine', # in serum/plasma
                     '2157-6': 'creatinine_kinase',
                     '14196-0': 'reticulocyte',
                     '2951-2': 'sodium'}
observation_cols = [f'baseline_{observation}_count' for observation in set(all_observations.values())]

# Cancer Location/Cancer Type
# TODO: remove this legacy code
cancer_location_mapping = { 'C421': 'Bone Marrow',
                            'C209': 'Rectum',
                            'C341': 'Lung (Upper Lobe)',
                            'C779': 'Lymph Node',
                            'C569': 'Ovary',
                            'C187': 'Colon (Sigmoid)', # sigmoid colon: last section of the bowel - the part that attaches to the rectum
                            'C619': 'Prostate Gland',
                            'C502': 'Breast (Upper-inner Quadrant)',
                            'C504': 'Breast (Upper-outer Quadrant)',
                            'C508': 'Breast (Overlapping Lesion)', # lesion: an area of tissue that has been damaged through injury/disease
                            'C180': 'Colon (Cecum)', # cecum: pouch that forms the first part of the large intestine
                            'C541': 'Endometrium', # mucous membrane lining the uterus
                            'C250': 'Pancreas (Head)'}

cancer_type_mapping = {'81403': 'Adenocarcinoma', # originate in mucous glands inside of organs (e.g. lungs, colon, breasts)
                       '85003': 'Breast Cancer (IDC)', # IDC: Invasive Ductal Carcinoma
                                                       # originate in milk duct and invade breast tissue outside the duct
                       '96803': 'Malignant Lymphoma (DLBCL)', # DLBCL: Diffuse Large B-cell Lymphoma
                                                              # originate  in white blood cell called lymphocytes
                       '97323': 'Plasma Cell Cancer (MM)', # MM: Multiple Myeloma
                       '84413': 'Ovarian/Pancreatic Cancer (SC)', # SC: Serous Cystadenocarcinoma
                                                                  # primarily in the ovary, rarely in the pancreas
                       '80413': 'Epithelial Cancer (Small Cell)', # primarily in the lung, sometimes in cervix, prostate, GI tract 
                       '84803': 'Colon Cancer (MA)', # MA: Mucinous Adenocarcinoma
                       '83803': 'Uterine Cancer (EA)', # EA: Endometrioid Adenocarcinoma
                       '80703': 'Skin Cancer (SCC)', # SCC: Squamous Cell Carcinoma
                       '80103': 'Epithelial Cancer (NOS)', # NOS: Not Otherwise Specified
                       '94403': 'Brain/Spinal Cancer (GBM)', # GBM: Glioblastoma
                       '81203': 'Tansitional Cell Cancer'} # Can occur in kidney, bladder, ureter, urethra, urachus

cancer_code_mapping = {'C16': 'Stomach',
                       'C18': 'Colon',
                       'C19': 'Rectosigmoid Junction',
                       'C20': 'Rectum',
                       'C25': 'Pancreas',
                       'C34': 'Lung/Bronchus',
                       'C50': 'Breast', 
                       'C53': 'Cervix Uteri',
                       'C54': 'Corpus Uteri', # body of uterus
                       'C56': 'Ovary',
                       'C61': 'Prostate Gland',
                       'C67': 'Bladder',
                       '804': 'Epithelial',
                       '807': 'Squamous Cell',
                       '814': 'Adenoma',
                       '844': 'Cystic',
                       '850': 'Ductal',
                       '852': 'Lobular'}
cancer_location_exclude = ['C77', 'C42'] # exclude blood cancers

# Drugs
din_exclude = ['02441489', '02454548', '01968017', '02485575', '02485583', '02485656', '02485591',
               '02484153', '02474565', '02249790', '02506238', '02497395'] # drug exclusion for neutrophil
cisplatin_dins = ['02403188', '02355183', '02126613', '02366711'] # aka Platinol, CDDP
cisplatin_cco_drug_code = ['003902']

# Official Language Codes
# NOTE: refer to datadictionary.ices.on.ca, Library: CIC, Member: CIC_IRCC
english_lang_codes = ['1', '15220', '15222', '3']

# Intent of Systemic Treatment
# NOTE: refer to datadictionary.ices.on.ca, Library: ALR, Member: SYSTEMIC 
intent_mapping = {'A': 'adjuvant', # applied after initial treatment for cancer (suppress secondary tumor formation)
                  'C': 'curative', # promote recovery and cure disease
                  'N': 'neoadjuvant', # shrink a tumor before main treatment 
                  'P': 'palliative'} # afford relief, but not cure

# Columns
systemic_cols = ['ikn', 'regimen', 'visit_date', 
                 'body_surface_area', # m^2
                 'intent_of_systemic_treatment',
                 'line_of_therapy', # the nth different chemotherapy regimen taken
                ]

y3_cols = ['ikn', 'sex', 'bdate',
           'lhin_cd', # local health integration network
           'curr_morph_cd', # cancer type
           'curr_topog_cd', # cancer location
          ] # 'pstlcode'

drug_cols = ['din', # DIN: Drug Identification Number
             'cco_drug_code', # CCO: Cancer Care Ontario
             'dose_administered', 'measurement_unit'] 
    
olis_cols = ['ikn', 'ObservationCode', 'ObservationDateTime', 'ObservationReleaseTS', 'ReferenceRange', 'Units', 
             'value_recommended_d']

symptom_cols = ['ecog_grade', 'prfs_grade', 'Wellbeing','Tiredness', 'Pain', 'Shortness of Breath', 'Drowsiness', 
                'Lack of Appetite', 'Depression', 'Anxiety', 'Nausea']

immigration_cols = ['ikn', 'is_immigrant', 'speaks_english']

diag_cols = [f'dx10code{i}' for i in range(1, 11)]
event_main_cols = ['ikn', 'arrival_date', 'depart_date']

# Model Training
# categorical hyperparam options
nn_solvers = ['adam', 'sgd']
nn_activations = ['tanh', 'relu', 'logistic']

# calibration params
calib_param = {'method': 'isotonic', 'cv': 3}
calib_param_logistic = {'method': 'sigmoid', 'cv': 3}

# Diagnostic Codes
# fever and infection (INFX)
fever_codes = ['R508', 'R509']
A_exclude = ['10', '11', '12', '13', '14', '29', '91']
B_exclude = ['10', '11', '12', '13', '14', '84']
infectious_and_parasitic_disease_codes = [f'A{i}{j}' for i in range(0, 10) for j in range(0, 10) if f'{i}{j}' not in A_exclude] + \
                                         [f'B{i}{j}' for i in range(0, 10) for j in range(0, 10) if f'{i}{j}' not in B_exclude]
post_op_would_infection_codes = ['T813', 'T814']
line_associated_infection_codes = ['T827']
bronchitis_codes = ['J20', 'J21', 'J22']
pneumonia_codes = ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18']
flu_codes = ['J09', 'J10', 'J11']
kidney_infection_codes = ['N10', 'N390']
acute_cystitis_codes = ['N300']
cellulitis_codes = ['L00', 'L01', 'L02', 'L03', 'L04', 'L05', 'L08']
empyema_codes = ['J86']
abscess_of_lung_codes = ['J85']

# gi toxicity (GI)
diarrhea_codes = ['K52', 'K591']
nausea_codes = ['R11']
abdominal_pain_codes = ['R10']
heartburn_codes = ['R12']
constipation_codes = ['K590']
obstruction_codes = ['K56']
stomatitis_codes = ['K12']
cachexia_codes = ['R640']
anorexia_codes = ['R630']

# other systemic treatment related
hyponatremia_codes = ['E871']
hypokalemia_codes = ['E876']
electrolyte_disorder_codes = ['E870', 'E872', 'E873', 'E874', 'E875', 'E877', 'E878']
magnesium_disorder_codes = ['E834']
dehydration_codes = ['E86']
malaise_codes = ['R53']
syncope_codes = ['R55']
dizziness_codes = ['R42']
hypotension_codes = ['I959']
anaemia_codes = ['D50', 'D51', 'D52', 'D53', 'D60', 'D61', 'D62', 'D63', 'D64']
thrombocytopenia_codes = ['D695', 'D696']
thrombophlebitis_codes = ['I802', 'I803']
pe_codes = ['I26']
other_venous_embolism_codes = ['I82']
rash_codes = ['R21']
hyperglycemia_codes = ['R73']
vascular_device_related_complication_codes = ['Z452', 'T825', 'T828', 'T856', 'Y712']
phlebitis_codes = ['I808']

# neutropenia
agranulocytosis_codes = ['D70']

diag_code_mapping = {'INFX': fever_codes + 
                             infectious_and_parasitic_disease_codes + 
                             post_op_would_infection_codes + 
                             line_associated_infection_codes + 
                             bronchitis_codes + 
                             pneumonia_codes + 
                             flu_codes + 
                             kidney_infection_codes + 
                             acute_cystitis_codes + 
                             cellulitis_codes + 
                             empyema_codes + 
                             abscess_of_lung_codes,
                    'GI': diarrhea_codes + 
                          nausea_codes + 
                          abdominal_pain_codes + 
                          heartburn_codes + 
                          constipation_codes + 
                          obstruction_codes + 
                          stomatitis_codes + 
                          cachexia_codes + 
                          anorexia_codes}
# treatment related encompasses everything!
diag_code_mapping['TR'] = diag_code_mapping['INFX'] + \
                          diag_code_mapping['GI'] + \
                          hyponatremia_codes + \
                          hypokalemia_codes + \
                          electrolyte_disorder_codes + \
                          magnesium_disorder_codes + \
                          dehydration_codes + \
                          malaise_codes + \
                          syncope_codes + \
                          dizziness_codes + \
                          hypotension_codes + \
                          anaemia_codes + \
                          thrombocytopenia_codes + \
                          thrombophlebitis_codes + \
                          pe_codes + \
                          other_venous_embolism_codes + \
                          rash_codes + \
                          hyperglycemia_codes + \
                          vascular_device_related_complication_codes + \
                          phlebitis_codes + \
                          agranulocytosis_codes
# ED/H Event
event_map = {'H':  {'event_name': 'hospitalization',
                    'date_col_name': ('admdate', 'ddate'),
                    'database_name': 'dad',
                    'event_cause_cols': [f'{cause}_H' for cause in diag_code_mapping]},
             'ED': {'event_name': 'emergency department visit',
                    'date_col_name': ('regdate', 'regdate'),
                    'database_name': 'nacrs',
                    'event_cause_cols': [f'{cause}_ED' for cause in diag_code_mapping]}}

# Clean Variable Names (ORDER MATTERS!)
clean_variable_mapping = {'chemo': 'chemotherapy', 
                          'prev': 'previous', 
                          'baseline_': '', 
                          'num': 'number_of', 
                          '_count': '',
                          'lhin_cd': 'local health integration network', 
                          'curr_topog_cd': 'cancer_location', 
                          'curr_morph_cd': 'cancer_type', 
                          'prfs': 'patient_reported_functional_status',
                          'eGFR': 'estimated_glomerular_filtration_rate',
                          'ODBGF': 'growth_factor',
                          'MCV': 'mean_corpuscular_volume',
                          'MCHC': 'mean_corpuscular_hemoglobin_concentration',
                          'MCH': 'mean_corpuscular_hemoglobin',
                          'INFX': 'due_to_fever_and_infection', 
                          'TR': 'due_to_treatment_related', 
                          'GI': 'due_to_gastrointestinal_toxicity',
                          'OTH': 'Other',
                          'H': 'hospitalization', 
                          'ED': 'ED_visit'}

# Variable Groupings
# {group: keywords} - Any variables whose name contains these keywords are assigned into that group
variable_groupings_by_keyword = {'Acute care use': 'INFX|GI|TR|prev_H|prior_H|prev_ED|prior_ED',
                                 'Cancer': 'curr_topog_cd|curr_morph_cd', 
                                 'Demographic': 'age|body|immigrant|lhin|sex|english',
                                 'Laboratory': 'baseline',
                                 'Treatment': 'visit_month|regimen|intent|chemo|therapy|cycle',
                                 'Symptoms': '|'.join(symptom_cols)}

# Acute Kidney Injury
# serum creatinine (SCr) levels
SCr_max_threshold = 132.63 # umol/L (1.5mg/dL)
SCr_rise_threshold = 26.53 # umol/L (0.3mg/dL)
SCr_rise_threshold2 = 353.68 # umol/L (4.0mg/dL)

# Chronic Kdiney Disease 
# glomerular filtration rate (eGFR) - https://www.kidney.org/professionals/kdoqi/gfr_calculator/formula
eGFR_params = {'F': {'K': 0.7, 'a': -0.241, 'multiplier': 1.012}, # Female
               'M': {'K': 0.9, 'a': -0.302, 'multiplier': 1.0}} # Male
