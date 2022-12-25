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
reco_folder = 'TRREKO' # Recommender folder = 'TRREKO' (TReatment RECOmmender)

# Dates
split_date = '2017-06-30' # date to temporally split data into developement and test cohort
min_chemo_date = '2014-07-01' # beginning date of chemo cohort
max_chemo_date = '2020-06-30' # final date of chemo cohort

# Main Blood Types and Low Blood Count Thresholds
blood_types = {
    'neutrophil': {
        'cytopenia_threshold': 1.5,  # cytopenia = low blood count
        'cytopenia_name': 'Neutropenia',
        'unit': '10^9/L'
    },
    'hemoglobin': {
        'cytopenia_threshold': 100,
        'cytopenia_name': 'Anemia',
        'unit': 'g/L'
    }, 
    'platelet': {
        'cytopenia_threshold': 75,
        'cytopenia_name': 'Thrombocytopenia',
        'unit': '10^9/L'
    }
}

cytopenia_gradings = {
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
observation_cols = [f'baseline_{observation}_count' for observation in set(all_observations.values())]
observation_change_cols = [col.replace('count', 'change') for col in observation_cols]

# Cancer Location/Cancer Type
# legacy code
cancer_code_specific_mapping = {
    'C421': 'Bone Marrow',
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
    'C250': 'Pancreas (Head)',
    '81403': 'Adenocarcinoma', # originate in mucous glands inside of organs (e.g. lungs, colon, breasts)
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
    '81203': 'Tansitional Cell Cancer' # Can occur in kidney, bladder, ureter, urethra, urachus
}

cancer_code_mapping = {
    'C00': 'Lip',
    'C01': 'Tongue - Base',
    'C02': 'Tongue - Other',
    'C03': 'Gum',
    'C04': 'Mouth - Floor',
    'C05': 'Palate',
    'C06': 'Mouth - Other',
    'C07': 'Parotid Gland',
    'C08': 'Salivary Gland - Other',
    'C09': 'Tonsil',
    'C10': 'Oropharynx',
    'C11': 'Nasopharynx',
    'C12': 'Pyriform Sinus',
    'C13': 'Hypopharynx',
    'C14': 'Pharynx - Other', # and other lip, oral cavity sites
    'C15': 'Esophagus',
    'C16': 'Stomach',
    'C17': 'Small Intestine',
    'C18': 'Colon',
    'C19': 'Rectosigmoid Junction',
    'C20': 'Rectum',
    'C21': 'Anus', # and anal canal
    'C22': 'Liver', # and intrahepatic bile ducts
    'C23': 'Gallbladder',
    'C24': 'Biliary Tract - Other',
    'C25': 'Pancreas',
    'C26': 'Digestive Organs - Other',
    'C30': 'Nasal Cavity', # and middle ear
    'C31': 'Accessory Sinuses',
    'C32': 'Larynx',
    'C33': 'Trachea',
    'C34': 'Lung', # and bronchus
    'C37': 'Thymus',
    'C38': 'Heart', # and mediastinum, pleura
    'C40': 'Bones - Limbs', # and joints, articular cartilage of limbs
    'C41': 'Bones - Other', # and other joints, articular cartilage sites
    'C44': 'Skin',
    'C47': 'Peripheral Nerves', # and autonomic nervous system
    'C48': 'Peritoneum', # and retroperitoneum
    'C49': 'Soft Tissue', # and connective, subcutaneous tissues
    'C50': 'Breast', 
    'C51': 'Vulva',
    'C52': 'Vagina',
    'C53': 'Cervix Uteri',
    'C54': 'Corpus Uteri', # body of uterus
    'C55': 'Uterus',
    'C56': 'Ovary',
    'C57': 'Female Genital Organ - Other',
    'C58': 'Placenta',
    'C60': 'Penis',
    'C61': 'Prostate Gland',
    'C62': 'Testis',
    'C63': 'Male Genital Organ - Other',
    'C64': 'Kidney',
    'C65': 'Renal Pelvis',
    'C66': 'Ureter',
    'C67': 'Bladder',
    'C68': 'Urinary Organs - Other',
    'C69': 'Eye', # and adnexa
    'C71': 'Brain',
    'C72': 'Spinal Cord', # and cranial nerves, central nervous system
    'C73': 'Thyroid Gland',
    'C74': 'Adrenal Gland', 
    'C75': 'Endocrine Gland - Other', 
    'C76': 'Other Sites', # any other sites that were left out
    'C80': 'Unknown Sites', # unknown primary sites
    '804': 'Epithelial',
    '807': 'Squamous Cell',
    '814': 'Adenoma',
    '844': 'Cystic',
    '850': 'Ductal',
    '852': 'Lobular'
}
cancer_location_exclude = ['C77', 'C42'] # exclude blood cancers
cancer_grouping = {
    'Head and Neck': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C09', 'C10', 'C12', 'C13', 'C14', 'C32'],
    'Liver': ['C23', 'C24', 'C22'],
    'Connective Tissue': ['C40', 'C41', 'C44', 'C49'],
    'Uterus': ['C54', 'C55']
}
palliative_cancer_grouping = {'Colorectal': ['C18', 'C19', 'C20']} # can group together only when intent is palliative

# Drugs
din_exclude = [
    '02441489', '02454548', '01968017', '02485575', '02485583', '02485656', '02485591',
    '02484153', '02474565', '02249790', '02506238', '02497395' # drug exclusion for neutrophil
]
cisplatin_dins = ['02403188', '02355183', '02126613', '02366711'] # aka Platinol, CDDP
cisplatin_cco_drug_code = ['003902']

# Official Language Codes
# NOTE: refer to datadictionary.ices.on.ca, Library: CIC, Member: CIC_IRCC
english_lang_codes = ['1', '15220', '15222', '3']

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
INTENT = 'intent_of_systemic_treatment'
BSA = 'body_surface_area' # m^2
systemic_cols = [
    'ikn', 
    'regimen', 
    'visit_date', 
    BSA,
    INTENT,
]

y3_cols = [
    'ikn', 
    'sex', 
    'bdate',       
    'lhin_cd', # local health integration network
    'curr_morph_cd', # cancer type
    'curr_topog_cd', # cancer location
    # 'pstlcode'
]

drug_cols = [
    'din', # DIN: Drug Identification Number
    'cco_drug_code', # CCO: Cancer Care Ontario
    'dose_administered', 
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
    'Wellbeing',
    'Tiredness', 
    'Pain', 
    'Shortness of Breath', 
    'Drowsiness', 
    'Lack of Appetite', 
    'Depression', 
    'Anxiety', 
    'Nausea'
]

immigration_cols = [
    'ikn', 
    'is_immigrant', 
    'speaks_english',
    'landing_date'
]

event_main_cols = [
    'ikn', 
    'arrival_date', 
    'depart_date'
]
diag_cols = [f'dx10code{i}' for i in range(1, 11)]

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
        'date_col_name': ('admdate', 'ddate'),
        'database_name': 'dad',
        'event_cause_cols': [f'{cause}_H' for cause in diag_code_mapping]
    },
    'ED': {
        'event_name': 'emergency department visit',
        'date_col_name': ('regdate', 'regdate'),
        'database_name': 'nacrs',
        'event_cause_cols': [f'{cause}_ED' for cause in diag_code_mapping]
    }
}

# Subgroups
# {group column name: {category: subgroup name associated with the category}
subgroup_map = {
    'is_immigrant': {False: 'Non-Immigrant', True: 'Immigrant'},
    'speaks_english': {False: 'Non-English Speaker', True: 'English Speaker'},
    'sex': {'F': 'Female', 'M': 'Male'},
    'world_region_of_birth': {'do_not_include': ['Unknown', 'Other']},
    'neighborhood_income_quintile': {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5',},
    'rural': {False: 'Urban', True: 'Rural'},
    'years_since_immigration': {False: 'Arrival >= 10 years', True: 'Arrival < 10 years'}
}
group_title_map = {
    'is_immigrant': 'Immigration', 
    'speaks_english': 'Language', 
    'sex': 'Sex',
    'world_region_of_birth': 'World Region of Birth', 
    'neighborhood_income_quintile': 'Income',
    'rural': 'Area of Residence',
    'years_since_immigration': 'Immigration Arrival'
}

# Clean Variable Names (ORDER MATTERS!)
clean_variable_mapping = {
    'baseline_': '', 
    '_count': '',
    'prev': 'previous', 
    'num': 'number_of', 
    'chemo': 'chemotherapy', 
    'lhin_cd': 'local health integration network', 
    'curr_topog_cd': 'cancer_topography_ICD-0-3', 
    'curr_morph_cd': 'cancer_morphology_ICD-0-3', 
    'prfs': 'patient_reported_functional_status',
    'eGFR': 'estimated_glomerular_filtration_rate',
    'ODBGF': 'growth_factor',
    'MCV': 'mean_corpuscular_volume',
    'MCHC': 'mean_corpuscular_hemoglobin_concentration',
    'MCH': 'mean_corpuscular_hemoglobin',
    'INFX': 'due_to_fever_and_infection', 
    'TR': 'due_to_treatment_related', 
    'GI': 'due_to_gastrointestinal_toxicity',
    'OTH': 'other',
    'H': 'hospitalization', 
    'ED': 'ED_visit'
}

# Variable Groupings
# {group: keywords} - Any variables whose name contains these keywords are assigned into that group
# NOTE: ORDER MATTERS! cisplatin_dosage gets grouped with Demographic first (because of dosAGE), then Treatment
variable_groupings_by_keyword = {
    'Acute care use': 'INFX|GI|TR|prev_H|prior_H|prev_ED|prior_ED',
    'Cancer': 'curr_topog_cd|curr_morph_cd', 
    'Demographic': 'age|body|income|immigrant|lhin|sex|english|diabetes|hypertension',
    'Laboratory': 'baseline',
    'Treatment': 'visit_month|regimen|intent|chemo|therapy|cycle|cisplatin|given|transfusion',
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
# model param options
nn_solvers = ['adam', 'sgd']
nn_activations = ['tanh', 'relu', 'logistic']

# calibration params
calib_param = {'method': 'isotonic', 'cv': 3}
calib_param_logistic = {'method': 'sigmoid', 'cv': 3}

# model tuning params
model_tuning_param = {
    'LR': {
        'C': (0.0001, 1)
    },
    'XGB': {
        'learning_rate': (0.001, 0.1),
        'n_estimators': (50, 200),
        'max_depth': (3, 7),
        'gamma': (0, 1),
        'reg_lambda': (0, 1)
    },
    'RF': {
        'n_estimators': (50, 200),
        'max_depth': (3, 7),
        'max_features': (0.01, 1)
    },
    'NN': {
        'learning_rate_init': (0.0001, 0.1),
        'batch_size': (64, 512),
        'momentum': (0,1),
        'alpha': (0,1),
        'first_layer_size': (16, 256),
        'second_layer_size': (16, 256),
        'solver': (0, len(nn_solvers)-0.0001),
        'activation': (0, len(nn_activations)-0.0001)
    },
    'RNN': {
        'batch_size': (8, 512),
        'learning_rate': (0.0001, 0.01),
        'hidden_size': (10, 200),
        'hidden_layers': (1, 5),
        'dropout': (0.0, 0.9),
        'model': (0.0, 1.0)
    },
    'ENS': {alg: (0, 1) for alg in ['LR', 'XGB', 'RF', 'NN', 'RNN']},
    # Baseline Models
    'LOESS': {
        'span': (0.01, 1)
    },
    'SPLINE': {
        'n_knots': (2,10), 
        'degree': (2,6),
        'C': (0.0001, 1)
    },
    'POLY': {
        'degree': (2,6),
        'C': (0.0001, 1)
    }
}
                                 
bayesopt_param = {
    'LR': {'init_points': 3, 'n_iter': 10}, 
    'XGB': {'init_points': 5, 'n_iter': 25},
    'RF': {'init_points': 3, 'n_iter': 20}, 
    'NN': {'init_points': 5, 'n_iter': 50},
    'RNN': {'init_points': 3, 'n_iter': 50},
    'ENS': {'init_points': 4, 'n_iter': 30},
    # Baseline Models
    'LOESS': {'init_points': 3, 'n_iter': 10},
    'SPLINE': {'init_points': 3, 'n_iter': 20},
    'POLY': {'init_points': 3, 'n_iter': 15}
}
