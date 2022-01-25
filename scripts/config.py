# Paths & Symbols
root_path = 'XXXXXXXX'
share_path = 'XXXXXXXX'
sas_folder = 'XXXXXXXX'
regiments_folder = 'XXXXXXXX'
plus_minus = u'\u00B1'

# Main Blood Types and Low Blood Count Thresholds
cytopenia_thresholds = {'neutrophil': 1.5, 'hemoglobin': 100, 'platelet': 75} # cytopenia = low blood count
blood_types = cytopenia_thresholds.keys()

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
                     '14682-9': 'creatinine',
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

cancer_code_mapping = {'C50': 'Breast', 
                       'C34': 'Lung/Bronchus',
                       'C18': 'Colon',
                       'C56': 'Ovary',
                       'C20': 'Rectum',
                       'C61': 'Prostate Gland',
                       '850': 'Ductal',
                       '814': 'Adenoma',
                       '807': 'Squamous Cell',
                       '804': 'Epithelial',
                       '852': 'Lobular'}

# Official Language Codes
english_lang_codes = ['1', '15220', '15222', '3']

# Exclusions
din_exclude = ['02441489', '02454548', '01968017', '02485575', '02485583', '02485656', '02485591',
               '02484153', '02474565', '02249790', '02506238', '02497395'] # drug exclusion for neutrophil
cancer_location_exclude = ['C77', 'C42'] # exclude blood cancers

# Columns
y3_cols = ['ikn', 'sex', 'bdate',
           'lhin_cd', # local health integration network
           'curr_morph_cd', # cancer type
           'curr_topog_cd', # cancer location
          ] # 'pstlcode'

systemic_cols = ['ikn', 'regimen', 'visit_date', 
                 'body_surface_area', # m^2
                 'intent_of_systemic_treatment', # A - after surgery, C - chemo, N - before surgery, P - incurable
                 'line_of_therapy', # nth number of different chemotherapy regimen 
                ] 
                # 'din', 'cco_drug_code', 'dose_administered', 'measurement_unit']
    
olis_cols = ['ikn', 'ObservationCode', 'ObservationDateTime', 'ObservationReleaseTS', 'ReferenceRange', 'Units', 'value']

esas_ecog_cols = ['ecog_grade', 'Wellbeing','Tiredness', 'Pain', 'Shortness of Breath', 'Drowsiness', 
                  'Lack of Appetite', 'Depression', 'Anxiety', 'Nausea']

immigration_cols = ['ikn', 'is_immigrant', 'speaks_english']

chemo_df_cols = ['ikn', 'regimen', 'visit_date', 'prev_visit', 'chemo_interval', 'chemo_cycle', 'immediate_new_regimen',
        'intent_of_systemic_treatment', 'line_of_therapy', 'lhin_cd', 'curr_morph_cd', 'curr_topog_cd',
        'age', 'sex', 'body_surface_area']

diag_cols = [f'dx10code{i}' for i in range(1, 11)]
event_main_cols = ['ikn', 'arrival_date', 'depart_date']

# Model Training
# categorical hyperparam options
nn_solvers = ['adam', 'sgd']
nn_activations = ['tanh', 'relu', 'logistic']

# calibration params
calib_param = {'method': 'isotonic', 'cv': 3}
calib_param_logistic = {'method': 'sigmoid', 'cv': 3}

# ml models
ml_models = ['LR', 'XGB', 'RF', 'NN']

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

# Clean Variable Names
clean_variable_mapping = {'chemo': 'chemotherapy', 
                          'curr_topog_cd': 'cancer_location', 
                          'curr_morph_cd': 'cancer_type', 
                          'H': 'hospitalization', 
                          'ED': 'ED_visit',
                          'prev': 'previous', 
                          'baseline_': '', 
                          'lhin_cd': 'local health integration network', 
                          'num': 'number_of', 
                          'INFX': 'due_to_fever_and_infection', 
                          'TR': 'due_to_treatment_related', 
                          'GI': 'due_to_gastrointestinal_toxicity'}
