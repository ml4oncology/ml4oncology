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
# Paths
root_path = 'XXXXX'
share_path = 'XXXXX'
sas_folder = 'XXXXX'
sas_folder2 = 'XXXXX'
regimens_folder = 'XXXXX'
cyto_folder = 'CYTOPENIA' # Cytopenia folder = 'CYTOPENIA'
acu_folder = 'PROACCT' # Acute care use folder = 'PROACCT' (PRediction of Acute Care use during Cancer Treatment)
can_folder = 'CAN' # Cisplatin-associated nephrotoxicity folder = 'CAN'
death_folder = 'DEATH' # Death folder = 'DEATH'
symp_folder = 'SYMPTOM' # Symptom deterioration folder = 'SYMPTOM'
reco_folder = 'TRREKO' # Recommender folder = 'TRREKO' (TReatment RECOmmender)

# Dates
split_date = '2017-06-30' # date to temporally split data into developement and test cohort
min_chemo_date = '2014-07-01' # beginning date of chemo cohort
max_chemo_date = '2020-06-30' # final date of chemo cohort

# Main Blood Types and Low Blood Count Thresholds
blood_types = {
    'neutrophil': {
        'cytopenia_name': 'Neutropenia',
        'unit': '10^9/L'
    },
    'hemoglobin': {
        'cytopenia_name': 'Anemia',
        'unit': 'g/L'
    }, 
    'platelet': {
        'cytopenia_name': 'Thrombocytopenia',
        'unit': '10^9/L'
    }
}
cytopenia_grades = {
    'Grade 2': {
        'Neutropenia': 1.5, 
        'Anemia': 100, 
        'Thrombocytopenia': 75, 
        'color': (236, 112, 20)
    },
    'Grade 3': {
        'Neutropenia': 1.0, 
        'Anemia': 80, 
        'Thrombocytopenia': 50, 
        'color': (204, 76, 2)
    },
    'Grade 4': {
        'Neutropenia': 0.5, 
        'Thrombocytopenia': 25, 
        'color': (102, 37, 6)
    }
}

# All Lab Tests and Blood Work
all_observations = { 
    '1742-6': 'alanine_aminotransferase',
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
    '2951-2': 'sodium'
}
observation_cols = [f'baseline_{observation}_value' for observation in set(all_observations.values())]
observation_change_cols = [col.replace('value', 'change') for col in observation_cols]

# Cancer Location/Cancer Type
cancer_code_mapping = {
    'C00': 'Lip',
    'C01': 'Base of tongue',
    'C02': 'Other and unspecified parts of tongue',
    'C03': 'Gum',
    'C04': 'Floor of mouth',
    'C05': 'Palate',
    'C06': 'Other and unspecified parts of mouth',
    'C07': 'Parotid gland',
    'C08': 'Other and unspecified major salivary glands',
    'C09': 'Tonsil',
    'C10': 'Oropharynx',
    'C11': 'Nasopharynx',
    'C12': 'Pyriform sinus',
    'C13': 'Hypopharynx',
    'C14': 'Other and ill-defined sites in lip, oral cavity, and pharynx',
    'C15': 'Esophagus',
    'C16': 'Stomach',
    'C17': 'Small intestine',
    'C18': 'Colon',
    'C19': 'Rectosigmoid junction',
    'C20': 'Rectum',
    'C21': 'Anus and anal canal',
    'C22': 'Liver and intrahepatic bile ducts',
    'C23': 'Gallbladder',
    'C24': 'Other and unspecified parts of biliary tract',
    'C25': 'Pancreas',
    'C26': 'Other and ill-defined digestive organs',
    'C30': 'Nasal cavity and middle ear',
    'C31': 'Accessory sinuses',
    'C32': 'Larynx',
    'C33': 'Trachea',
    'C34': 'Bronchus and lung',
    'C37': 'Thymus',
    'C38': 'Heart, mediastinum, and pleura',
    'C40': 'Bones, joints, and articular cartilage of limbs',
    'C41': 'Bones, joints, and articular cartilage of other and unspecified sites',
    'C42': 'Hematopoietic and reticuloendothelial systems',
    'C44': 'Skin',
    'C47': 'Peripheral nerves and autonomic nervous system',
    'C48': 'Retroperitoneum and peritoneum',
    'C49': 'Connective, subcataneous, and other soft tissues',
    'C50': 'Breast', 
    'C51': 'Vulva',
    'C52': 'Vagina',
    'C53': 'Cervix uteri',
    'C54': 'Corpus uteri',
    'C55': 'Uterus, NOS',
    'C56': 'Ovary',
    'C57': 'Other and unspecified female genital organs',
    'C58': 'Placenta',
    'C60': 'Penis',
    'C61': 'Prostate gland',
    'C62': 'Testis',
    'C63': 'Other and unspecified male genital organs',
    'C64': 'Kidney',
    'C65': 'Renal pelvis',
    'C66': 'Ureter',
    'C67': 'Bladder',
    'C68': 'Other and unspecified urinary organs',
    'C69': 'Eye and adnexa',
    'C71': 'Brain',
    'C72': 'Spinal cord, cranial nerves, and other parts of central nervous system',
    'C73': 'Thyroid and other endocrine glands',
    'C74': 'Adrenal Gland', 
    'C75': 'Other endocrine glands and related structures', 
    'C76': 'Other and ill-defined sites',
    'C77': 'Lymph Nodes',
    'C80': 'Unknown primary site',
    '800': 'Neoplasms, NOS',
    '801': 'Epithelial neoplasms, NOS',
    '802': 'Epithelial neoplasms, NOS',
    '803': 'Epithelial neoplasms, NOS',
    '804': 'Epithelial neoplasms, NOS',
    '805': 'Squamous cell neoplasms',
    '807': 'Squamous cell neoplasms',
    '808': 'Squamous cell neoplasms',
    '809': 'Basal cell neoplasms',
    '812': 'Transitional cell papillomas and carcinomas',
    '813': 'Transitional cell papillomas and carcinomas',
    '814': 'Adenomas and adenocarcinomas',
    '816': 'Adenomas and adenocarcinomas',
    '817': 'Adenomas and adenocarcinomas',
    '818': 'Adenomas and adenocarcinomas',
    '820': 'Adenomas and adenocarcinomas',
    '821': 'Adenomas and adenocarcinomas',
    '823': 'Adenomas and adenocarcinomas',
    '824': 'Adenomas and adenocarcinomas',
    '825': 'Adenomas and adenocarcinomas',
    '826': 'Adenomas and adenocarcinomas',
    '829': 'Adenomas and adenocarcinomas',
    '831': 'Adenomas and adenocarcinomas',
    '832': 'Adenomas and adenocarcinomas',
    '833': 'Adenomas and adenocarcinomas',
    '834': 'Adenomas and adenocarcinomas',
    '837': 'Adenomas and adenocarcinomas',
    '838': 'Adenomas and adenocarcinomas',
    '840': 'Adnexal and skin appendage neoplasms',
    '843': 'Mucoepidermoid neoplasms',
    '844': 'Cystic, mucinous, and serous neoplasms',
    '845': 'Cystic, mucinous, and serous neoplasms',
    '846': 'Cystic, mucinous, and serous neoplasms',
    '847': 'Cystic, mucinous, and serous neoplasms',
    '848': 'Cystic, mucinous, and serous neoplasms',
    '849': 'Cystic, mucinous, and serous neoplasms',
    '850': 'Ductal and lobular neoplasms',
    '851': 'Ductal and lobular neoplasms',
    '852': 'Ductal and lobular neoplasms',
    '853': 'Ductal and lobular neoplasms',
    '854': 'Ductal and lobular neoplasms',
    '855': 'Acinar cell neoplasms',
    '856': 'Complex epithelial neoplasms',
    '857': 'Complex epithelial neoplasms',
    '858': 'Thymic epithelial neoplasms',
    '862': 'Specialized gonadal neoplasms',
    '872': 'Paragangliomas and glomus tumors',
    '873': 'Nevi and melanomas',
    '874': 'Nevi and melanomas',
    '877': 'Nevi and melanomas',
    '880': 'Soft tissue tumors and sarcomas, NOS',
    '881': 'Fibromatous neoplasms',
    '883': 'Fibromatous neoplasms',
    '885': 'Lipomatous neoplasms',
    '889': 'Myomatous neoplasms',
    '890': 'Myomatous neoplasms',
    '891': 'Myomatous neoplasms',
    '892': 'Myomatous neoplasms',
    '892': 'Myomatous neoplasms',
    '893': 'Complex mixed and stromal neoplasms',
    '894': 'Complex mixed and stromal neoplasms',
    '895': 'Complex mixed and stromal neoplasms',
    '898': 'Complex mixed and stromal neoplasms',
    '902': 'Fibroepithelial neoplasms',
    '904': 'Synovial-like neoplasms',
    '905': 'Mesothelial neoplasms',
    '906': 'Germ cell neoplasms',
    '907': 'Germ cell neoplasms',
    '908': 'Germ cell neoplasms',
    '910': 'Trophoblastic neoplasms',
    '911': 'Mesonephromas',
    '912': 'Blood vessel tumors',
    '914': 'Blood vessel tumors',
    '915': 'Blood vessel tumors',
    '918': 'Osseous and chondromatous neoplasms',
    '922': 'Osseous and chondromatous neoplasms',
    '924': 'Osseous and chondromatous neoplasms',
    '926': 'Miscellaneous bone tumors',
    '936': 'Miscellaneous tumors',
    '938': 'Gliomas',
    '940': 'Gliomas',
    '942': 'Gliomas',
    '944': 'Gliomas',
    '945': 'Gliomas',
    '947': 'Gliomas',
    '954': 'Nerve sheath tumors',
    '956': 'Nerve sheath tumors',
    'other': 'Other'
}
blood_cancer_code = ['C77', 'C42']
cancer_grouping = {
    'Head and Neck': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C09', 'C10', 'C12', 'C13', 'C14'],
    'Gastrointestinal': ['C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'], # aka gastroenterology
    'Chest': ['C30', 'C31', 'C32', 'C33', 'C34', 'C37', 'C38'], # aka thoracic oncology, includes everything in the thorax (chest cavity) like lung, heart
    'Female Organs': ['C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58'], # aka gynecology
    'Male Organs': ['C60', 'C61', 'C62', 'C63'], # aka andrology
    'Urinary Tract': ['C64', 'C65', 'C66', 'C68'], # aka urology
    'Central Nervous System': ['C69', 'C70', 'C71', 'C72'], # aka neuro-oncology, includes eye, brain
    'Endocrine': ['C73', 'C74', 'C75'], # aka endocrine oncology, includes thyroid
}
palliative_cancer_grouping = {'Colorectal': ['C18', 'C19', 'C20']} # can group together only when intent is palliative

# Drugs
neutrophil_dins = [
    '02441489', '02454548', '01968017', '02485575', '02485583', '02485656', '02485591',
    '02484153', '02474565', '02249790', '02506238', '02497395' # exclude these drugs for Neutropenia
]
cisplatin_dins = ['02403188', '02355183', '02126613', '02366711'] # aka Platinol, CDDP
cisplatin_cco_drug_codes = ['003902']

# Official Language Codes
# NOTE: refer to datadictionary.ices.on.ca, Library: CIC, Member: CIC_IRCC
eng_lang_codes = ['1', '15220', '15222', '3']

# World Regions
# Reference: en.wikipedia.org/wiki/United_Nations_geoscheme
# Excluded Countries: 
# - unknown countries: 'Not Stated', 'Europe NES', 'Asia NES', 'Africa NES'
# - dissolved countries: 'Netherlands Antilles'
# - all other countries not listed below with codes greater than 4 characters
eastern_europe_countries = [
    'Belarus', 'Bulgaria', 'Czech Republic', 'Czechoslovakia', 'Hungary', 'Kosovo', 'Moldova',  'Poland', 'Romania', 
    'Russia', 'Slovak Republic', 'Ukraine', 'Union of Soviet Socialist Republics', 'Yugoslavia'
]
northern_europe_countries = [
    'Denmark', 'England', 'Estonia', 'Finland', 'Iceland', 'Ireland', 'Latvia', 'Lithuania', 'Norway', 'Scotland', 
    'Sweden', 'United Kingdom and Colonies', 'United Kingdom and Overseas Territories',  'Wales'
]
southern_europe_countries = [
    'Albania', 'Andorra', 'Bosnia-Herzegovina', 'Croatia', 'Greece', 'Holy See', 'Italy', 'Malta', 'Montenegro', 
    'Macedonia', 'Portugal', 'San Marino', 'Serbia', 'Serbia and Montenegro', 'Slovenia', 'Spain'
]
western_europe_countries = [
    'Austria', 'Belgium', 'France', 'Germany', 'Liechtenstein','Luxembourg','Monaco', 'Netherlands', 'Switzerland',
]
north_africa_countries = ['Algeria', 'Egypt', 'Libya', 'Morocco', 'Sudan', 'Tunisia']
sub_saharan_africa_countries = [
    'Angola', 'Benin', 'Botswana', 'Burkina-Faso', 'Burundi', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros',
    'Congo', 'Djibouti', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon Republic', 'Gambia', 'Ghana', 'Guinea', 
    'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 
    'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 
    'South Africa', 'South Sudan', 'Swaziland', 'Tanzania', 'Togo', 'Uganda', 'Zambia', 'Zimbabwe'
]
east_asia_countries = [
    'China', 'Hong Kong', 'Hong Kong SAR', 'Japan', 'Korea', 'Macau Sar', 'Mongolia', 'Taiwan', 'Tibet'
]
south_east_asia_countries = [
    'Brunei', 'Cambodia', 'East Timor', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar (Burma)', 'Philippines', 'Singapore',
    'Thailand', 'Vietnam'
]
south_asia_countries = ['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka']
west_asia_countries = [
    'Armenia', 'Azerbaijan', 'Bahrain', 'Cyprus', 'Georgia', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon',
    'Oman', 'Palestinian Authority (Gaza/West Bank)', 'Qatar', 'Saudi Arabia', 'Saudi Arabia', 'Syria', 'Turkey', 
    'United Arab Emirates', 'Western Sahara', 'Yemen'
]
# NOTE: Cuba and Puerto Rico are pre-dominantly white/hispanic, not black
carribean_countries = [
    'Anguilla', 'Antigua and Barbuda', 'Bahama Islands', 'Barbados', 'Cuba', 'Dominica', 'Dominican Republic', 'Grenada', 
    'Guadeloupe', 'Haiti', 'Jamaica', 'Martinique', 'Montserrat', 'Nevis', 'Puerto Rico', 'St. Kitts-Nevis', 'St. Lucia', 
    'St. Vincent and the Grenadines', 'Trinidad and Tobago', 'Virgin Islands'
]
south_america_countries = [
    'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador',  'French Guiana', 'Guyana', 'Paraguay', 'Peru',
    'Surinam', 'Uruguay', 'Venezuela'
]
central_america_countries = ['Belize', 'Costa Rica', 'El Salvador', 'Guatemala',  'Honduras', 'Mexico', 'Nicaragua', 'Panama']
north_america_countries = ['Canada', 'United States of America']

world_region_country_map = {
    'EA': # East Asia
        east_asia_countries,
    'SEA': # South East Asia
        south_east_asia_countries,
    'SA': # South Asia
        south_asia_countries,
    'MENA': # Middle East (West Asia)/North Africa
        west_asia_countries + north_africa_countries, 
    'Carrib/SSA': # Carribean/Sub-Saharan Africa
        carribean_countries + sub_saharan_africa_countries,
    'LatAm': # Latin America
        central_america_countries + south_america_countries,
    'EU/NA/A+NZ': # Europe/North America/Australia + New Zealand
        eastern_europe_countries + 
        western_europe_countries + 
        southern_europe_countries + 
        northern_europe_countries + 
        north_america_countries + 
        ['Australia', 'New Zealand'],
    'Other': 
        ['Aruba', 'Azores', 'Bermuda', 'Cape Verde Islands', 'Canary Islands', 'Fiji', 'French Polynesia', 'Gibraltar',
         'Guam', 'Kazakhstan', 'Kyrgyzstan', 'Mauritius', 'Madeira', 'Papua New Guinea', 'Reunion', 'Samoa', 
         'Tajikistan', 'Turkmenistan', 'Uzbekistan']
}
# Excuded Languages:
# - lanugage of unknown source: 'Ada', 'Facilitator', 'Fouki', 'Harara', 'Hargar', 'Ilican', 'Macena', 'Osal', 
#     'Other Languages Nes', 'Scoula', 'Shansai', 'Uigrigma', 'Unknown', 'Yiboe'
# - unspoken lanugage: 'Sign Language (Lsq)', 'Deaf-Mute'
# - colonial language: 'English', 'Spanish', 'French', 'Portuguese'
# - religious language: 'Hebrew', 'Yiddish'
# - all other languages not listed below with codes greater than 4 characters
chinese_lang = [
    'Cantonese', 'Changle', 'Chaocho', 'Chinese', 'Chiuchow', 'Chowchau', 'Foochow', 'Fujian', 'Fukinese', 'Fuqing', 
    'Hainam', 'Hakka', 'Hokkin', 'Mandarin', 'Other Chinese Dialects', 'Sechuan', 'Shanghai', 'Shanghainese', 'Taichew',
    'Tatshanese', 'Teochew', 'Tibetan', 'Tichiew', 'Toishan'
]
philippino_lang = [
    'Aklanon', 'Bicol', 'Bikol', 'Bisaya', 'Bontok', 'Capizeno', 'Cebuano', 'Chavacano', 'Hiligaynon', 'Igorot', 
    'Iiongo', 'Ilocano', 'Pampango', 'Pangasinan', 'Tagalog', 'Visayan', 'Waray', 'Waray-Waray'
]
indian_lang = [
    'Bengali', 'Bengali Sylhetti', 'Concani', 'Gujarati', 'Hindi', 'Kanarese', 'Kankani', 'Kannada', 'Kashmiri', 
    'Konkani', 'Malayalam', 'Marathi', 'Oriya', 'Other Western Hemisphere Indian Languages', 'Punjabi', 'Tamil', 
    'Telugu', 'Urdu'
]
world_region_language_map = {
    'EA': # East Asia
        chinese_lang + 
        ['Busan', 'Japanese', 'Korean'],
    'SEA': # South East Asia
        philippino_lang + 
        ['Burmese', 'Cambodian', 'Indonesian', 'Javanese', 'Khmer', 'Laotian', 'Malay', 
         'Other South East Asian Languages', 'Phuockien', 'Shan', 'Thai', 'Vietnamese'],
    'SA': # South Asia
        indian_lang + 
        ['Afghan', 'Chakma', 'Dari', 'Dari/Farsi', 'Hindko', 'Kacchi', 'Mizo', 'Nepali', 'Other South Asian Languages',
         'Pahari', 'Pashto', 'Pushta', 'Sinhala', 'Sinhalese', 'Telugu'],
    'MENA': # Middle East (West Asia)/North Africa
        ['Arabic', 'Arabic Standard', 'Assyrian', 'Azerbaijani', 'Azeri', 'Berber', 'Bijaiya', 'Chaldean', 'Farsi', 
         'Georgian', 'Kandahari', 'Kurdish', 'Kurdish-Northern', 'Lebanese', 'Other Middle Eastern Languages', 'Persian',
         'Sindhi', 'Turkish'],
    'Carrib/SSA': # Carribean/Sub-Saharan Africa
        ['Affar', 'Afrikaans', 'Aka', 'Akan', 'Akra', 'Amharic', 'Ashanti', 'Bajuni', 'Bambara', 'Bamileke', 'Bantu', 
         'Baule', 'Belen', 'Bemba', 'Beni', 'Benin', 'Bini', 'Bissa', 'Busango', 'Chichewa', 'Chiyao', 'Creole', 
         'Dioula', 'Edo', 'Efik', 'Eritrean', 'Esan', 'Ewe', 'Fang', 'Fanti', 'Foullah', 'Fulani', 'Ga', 'Guerze', 
         'Gurage', 'Harary', 'Hassanya', 'Hausa', 'Ibibio', 'Ibo', 'Igbo', 'Ika', 'Ishan', 'Izi', 'Jamaican', 'Jolay', 
         'Kakwa', 'Kihavu', 'Kikongo', 'Kikuyu', 'Kinyanwanda', 'Kinyarwanda', 'Kirundi', 'Kiswahili', 'Krio', 'Lengie', 
         'Lingala', 'Lowma', 'Luganda', 'Lugishu', 'Lutoro', 'Macua', 'Mahou', 'Makonde', 'Malagasy', 'Maligo', 
         'Malinke', 'Mandingo', 'Mashi', 'Maymay', 'Mende', 'Mina', 'More', 'Ndebele', 'Nzima', 'Okpe', 'Oromo',
         'Other African Languages', 'Peul', 'Pidgin', 'Portuguese-Guinea-Bissau', 'Poular', 'Rukiga', 'Runyankole', 
         'Rutooro', 'Samoli', 'Sango', 'Sesotho', 'Seswi', 'Seychelles', 'Shona', 'Somali', 'Soninke', 'Sotho', 
         'Soussou', 'Suesue', 'Sukuma', 'Swahili', 'Swati', 'Swazai', 'Tigre', 'Tigrigna', 'Tigrinya', 'Timini', 'Tiv',
         'Tshiluba', 'Tsibula', 'Twi', 'Uhrobo', 'Umbundu', 'Unama', 'Wolof', 'Xhosa', 'Yoruba', 'Zshiluba', 'Zuganda',
         'Zulu'],
    'LatAm': # Latin America
        ['Guyanese', 'Portuguese-Brazil'],
    'EU/NA/A+NZ': # Europe/North America/Australia + New Zealand
        ['Albanian', 'Aran', 'Armenian', 'Belarusian', 'Breton', 'Bulgarian', 'Catalan', 'Croatian', 'Czech', 'Danish',
         'Dutch', 'Estonian', 'Finnish', 'Flemish', 'Gaelic', 'German', 'Greek', 'Hungarian', 'Italian', 'Latvian', 
         'Lithuanian', 'Macedonian', 'Maltese', 'Moldovan', 'Norwegian', 'Other European Languages', 'Polish', 
         'Romanian', 'Russian', 'Serbian', 'Serbo-Croat', 'Serbo-Croatian', 'Slovak', 'Slovene', 'Swedish', 'Ukrainian',
         'Welsh'],
    'Other': 
        ['Ouighour', 'Samoan', 'Tari', 'Turkmen', 'Uzbek']
}

# Intent of Systemic Treatment
# NOTE: refer to datadictionary.ices.on.ca, Library: ALR, Member: SYSTEMIC 
intent_mapping = {
    'A': 'adjuvant', # applied after initial treatment for cancer (suppress secondary tumor formation)
    'C': 'curative', # promote recovery and cure disease
    'N': 'neoadjuvant', # shrink a tumor before main treatment 
    'P': 'palliative' # afford relief, but not cure
}

# Columns
DATE = 'visit_date'
BSA = 'body_surface_area' # m^2
INTENT = 'intent_of_systemic_treatment'
systemic_cols = [
    'ikn', 
    'regimen', 
    DATE, 
    BSA,
    INTENT,
    'inpatient_flag',
    'visit_hospital_number',
    'init_date',
    'init_regimen_date',
    'prev_date'
]

y3_cols = [
    'ikn', 
    'sex', 
    'dthdate', 
    'bdate',
    'lhin_cd', # local health integration network
    'curr_morph_cd', # cancer type (morphology)
    'curr_topog_cd', # cancer location (topography)
    # 'pstlcode'
]

DOSE = 'dose_administered'
drug_cols = [
    'din', # DIN: Drug Identification Number
    'cco_drug_code', # CCO: Cancer Care Ontario
    DOSE, 
    'measurement_unit'
] 

OBS_CODE = 'ObservationCode'
OBS_VALUE = 'value_recommended_d'
OBS_DATE = 'ObservationDateTime'
OBS_RDATE = 'ObservationReleaseTS'
olis_cols = [
    'ikn', 
    OBS_CODE, 
    OBS_VALUE,
    OBS_DATE, 
    OBS_RDATE, 
    'ReferenceRange', 
    'Units', 
]

symptom_cols = [
    'ecog_grade', 
    'prfs_grade', 
    'wellbeing',
    'tiredness', 
    'pain', 
    'shortness_of_breath', 
    'drowsiness', 
    'lack_of_appetite', 
    'depression', 
    'anxiety', 
    'nausea'
]

event_main_cols = [
    'ikn', 
    'arrival_date', 
    'depart_date'
]
diag_cols = [f'dx10code{i}' for i in range(1, 11)]

next_scr_cols = ['next_SCr_value', 'next_SCr_obs_date'] # Scr = serum creatinine
next_scr_cols += [f'next_{col}' for col in next_scr_cols]

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

diag_code_mapping = {
    'INFX': 
        fever_codes + 
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
    'GI': 
        diarrhea_codes + 
        nausea_codes + 
        abdominal_pain_codes + 
        heartburn_codes + 
        constipation_codes + 
        obstruction_codes + 
        stomatitis_codes + 
        cachexia_codes + 
        anorexia_codes
}
# treatment related encompasses everything!
diag_code_mapping['TR'] = \
    diag_code_mapping['INFX'] + \
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
event_map = {
    'H': {
        'event_name': 'hospitalization',
        'arrival_col': 'admdate',
        'depart_col': 'ddate',
        'event_cause_cols': [f'{cause}_H' for cause in diag_code_mapping]
    },
    'ED': {
        'event_name': 'emergency department visit',
        'arrival_col': 'regdate',
        'depart_col': 'regdate',
        'event_cause_cols': [f'{cause}_ED' for cause in diag_code_mapping]
    }
}

# Clean Variable Names 
# WARNING: ORDER MATTERS!
clean_variable_mapping = {
    'baseline_': '', 
    '_value': '',
    'prev': 'previous', 
    'num': 'number_of', 
    'chemo': 'chemotherapy', 
    'lhin_cd': 'local health integration network', 
    'cancer_topog_cd': 'topography_ICD-0-3', 
    'cancer_morph_cd': 'morphology_ICD-0-3', 
    'prfs': 'patient_reported_functional_status',
    'eGFR': 'estimated_glomerular_filtration_rate',
    'GF': 'growth_factor',
    'MCV': 'mean_corpuscular_volume',
    'MCHC': 'mean_corpuscular_hemoglobin_concentration',
    'MCH': 'mean_corpuscular_hemoglobin',
    'INFX': 'due_to_fever_and_infection', 
    'TR': 'due_to_treatment_related', 
    'GI': 'due_to_gastrointestinal_toxicity',
    'H': 'hospitalization', 
    'ED': 'ED_visit'
}

# Variable Groupings
# {group: keywords} - Any variables whose name contains these keywords are assigned into that group
# WARNING: ORDER MATTERS!
variable_groupings_by_keyword = {
    'Acute care use': 'INFX|GI|TR|prev_H|prior_H|prev_ED|prior_ED',
    'Cancer': 'cancer', 
    'Demographic': 'age|body|income|immigra|rural|lhin|sex|birth|english|diabetes|hypertension',
    'Laboratory': 'baseline',
    'Treatment': 'visit_month|regimen|intent|chemo|therapy|cycle|dosage|given|transfusion',
    'Symptoms': '|'.join(symptom_cols)
}

# Acute Kidney Injury
# serum creatinine (SCr) levels
SCr_max_threshold = 132.63 # umol/L (1.5mg/dL)
SCr_rise_threshold = 26.53 # umol/L (0.3mg/dL)
SCr_rise_threshold2 = 353.68 # umol/L (4.0mg/dL)

# Chronic Kdiney Disease 
# glomerular filtration rate (eGFR) - https://www.kidney.org/professionals/kdoqi/gfr_calculator/formula
eGFR_params = {
    'F': {'K': 0.7, 'a': -0.241, 'multiplier': 1.012}, # Female
    'M': {'K': 0.9, 'a': -0.302, 'multiplier': 1.0} # Male
}

# Model Training
# model tuning params
model_tuning_param = {
    'LR': {
        'inv_reg_strength': (0.0001, 1)
    },
    'RF': {
        'n_estimators': (50, 200),
        'max_depth': (3, 7),
        'max_features': (0.01, 1),
        'min_samples_leaf': (0.001, 0.1),
        'min_impurity_decrease': (0, 0.1),
    },
    'XGB': {
        'n_estimators': (50, 200),
        'max_depth': (3, 7),
        'learning_rate': (0.01, 0.3),
        'min_split_loss': (0, 0.5),
        'min_child_weight': (6, 100),
        'reg_lambda': (0, 1),
        'reg_alpha': (0, 1000)
    },
    'NN': {
        'batch_size': (64, 4096),
        'hidden_size1': (16, 256),
        'hidden_size2': (16, 256),
        'dropout': (0, 0.5),
        'optimizer': (0, 1),
        'learning_rate': (0.0001, 0.1),
        'weight_decay': (0.0001, 1),
        'momentum': (0, 0.9),
    },
    'RNN': {
        'batch_size': (16, 1024),
        'hidden_size': (16, 256),
        'hidden_layers': (1, 5),
        'dropout': (0.0, 0.9),
        'learning_rate': (0.0001, 0.01),
        'weight_decay': (0.0001, 1),
        'model': (0, 1)
    },
    'ENS': {alg: (0, 1) for alg in ['LR', 'XGB', 'RF', 'NN', 'RNN']},
    # Baseline Models
    'LOESS': {
        'span': (0.01, 1)
    },
    'SPLINE': {
        'n_knots': (2,9), 
        'degree': (2,5),
        'inv_reg_strength': (0.0001, 1)
    },
    'POLY': {
        'degree': (2,5),
        'inv_reg_strength': (0.0001, 1)
    },
    # Experimentally Supported Models
    'TCN': {
        'batch_size': (16, 1024),
        'num_channel1': (16, 256),
        'num_channel2': (16, 256),
        'num_channel3': (16, 256),
        'hidden_size': (16, 256),
        'kernel_size': (2, 5),
        'dropout': (0.0, 0.9),
        'learning_rate': (0.0001, 0.01),
        'weight_decay': (0.0001, 1),
    },
    'LGBM': {
        'n_estimators': (50, 200),
        'max_depth': (3, 7),
        'learning_rate': (0.01, 0.3),
        'num_leaves': (20, 40),
        'min_data_in_leaf': (6, 100),
        'feature_fraction': (0.5, 1),
        'bagging_fraction': (0.5, 1),
        'bagging_freq': (0, 10),
        'reg_lambda': (0, 1),
        'reg_alpha': (0, 1)
    }
}
                                 
bayesopt_param = {
    'LR': {'init_points': 2, 'n_iter': 10}, 
    'RF': {'init_points': 10, 'n_iter': 50}, 
    'XGB': {'init_points': 14, 'n_iter': 70},
    'NN': {'init_points': 16, 'n_iter': 80},
    'RNN': {'init_points': 14, 'n_iter': 70},
    'ENS': {'init_points': 4, 'n_iter': 40},
    # Baseline Models
    'LOESS': {'init_points': 3, 'n_iter': 10},
    'SPLINE': {'init_points': 3, 'n_iter': 20},
    'POLY': {'init_points': 3, 'n_iter': 15},
    # Experimentally Supported Models
    'TCN': {'init_points': 16, 'n_iter': 60},
    'LGBM': {'init_points': 20, 'n_iter': 200}
}
