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
library(lme4)
library(metafor)

data <- read.csv('models/tables/cancer_center.csv')
colnames(data)
nrow(data)

# do not include the other small centers when computing heterogenity
data <- data[data['Cancer.Center'] != 'Other', ]

calc <- function(data, target){
    start_time <- Sys.time()
    # take a subset of the data
    group <- data[data['Targets'] == target, ]
    # fit random-effect model
    res <- rma(AUROC, AUROC_variance, data=group, method='EE')
    end_time <- Sys.time()
    # print(end_time - start_time)
    res
}

# sigh...manually copy and paste the output to models/logs/hetero.txt
for(target in unique(data$Targets)){
    print('#####################################################')
    print(target)
    print('#####################################################')
    res <- calc(data, target)
    print(res)
}