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

library(geepack)

GEE <- function(data, corstr){
    start_time <- Sys.time()
    model <- geeglm(Label ~ Pred + Time, 
                    id = Subject, 
                    wave = Time, 
                    family = binomial, 
                    corstr = corstr, 
                    std.err = 'san.se', 
                    data = data)
    end_time <- Sys.time()
    print(end_time - start_time)
    print(summary(model))
    model
}

GLMM <- function(data, nAGQ){
    start_time <- Sys.time()
    model <- glmer(Label ~ Pred + Time + (1 | Subject), 
                   family = binomial, 
                   data = data, 
                   nAGQ = nAGQ)
    end_time <- Sys.time()
    print(end_time - start_time)
    print(summary(model))
    model
}

GLM <- function(data){
    start_time <- Sys.time()
    model <- glm(Label ~ Pred + Time, 
                 family = binomial, 
                 data = data)
    end_time <- Sys.time()
    print(end_time - start_time)
    print(summary(model))
    model
}

data <- read.csv('data/R_data.csv')
colnames(data)
max(data['Time'])
nrow(data)

glmm_model <- GLMM(data, nAGQ = 1)

glmm_model <- GLMM(data, nAGQ = 2)

glmm_model <- GLMM(data, nAGQ = 3)

result <- GLMM(data, nAGQ = 10)

result <- GLMM(data, nAGQ = 100)

# Testing to see if adding glmerControl(optimizer='bobyqa') would prevent hessian error
# Result: NOPE
GLMM <- function(data, nAGQ){
    start_time <- Sys.time()
    control = glmerControl(optimizer='bobyqa')
    model <- glmer(Label ~ Pred + Time + (1 | Subject), 
                   family = binomial,
                   control = control, 
                   data = data, 
                   nAGQ = nAGQ)
    end_time <- Sys.time()
    print(end_time - start_time)
    print(summary(model))
    model
}
result <- GLMM(data, nAGQ = 100)

# Try glmmTMB package to see what you get
# Same result as GLMM with nAGQ < 3, with slightly worse standard errors
library(glmmTMB) # TMB = Template Model Build
GLMM <- function(data){
    start_time <- Sys.time()
    model <- glmmTMB(Label ~ Pred + Time + (1 | Subject), 
                     family = binomial,
                     data = data)
    end_time <- Sys.time()
    print(end_time - start_time)
    print(summary(model))
    model
}
result <- GLMM(data)

# Get confidence interval
ci <- exp(confint(glmm_model, method='Wald', parm='beta_'))
fe <- exp(fixef(glmm_model))
cbind(est=fe, ci)

glm_model <- GLM(data)

# lower AIC/BIC is better
# AIC(glm_model, glmm_model)
# BIC(glm_model, glmm_model)
anova(glmm_model, glm_model)

ind <- GEE(data, corstr = 'independence')

ex <- GEE(data, corstr = 'exchangeable')

ar1 <- GEE(data, corstr = 'ar1')

# kernel crashes...
uns <- GEE(data, corstr = 'unstructured')

QIC(ind, ex, ar1)

data <- read.csv('data/R_data2.csv')
colnames(data)
max(data['Time'])
nrow(data)

result <- GLMM(data, nAGQ=1)

result <- GLMM(data, nAGQ = 10)

result <- GLMM(data, nAGQ = 100)

ind <- GEE(data, corstr = 'independence')

ex <- GEE(data, corstr = 'exchangeable')

ar1 <- GEE(data, corstr = 'ar1')

# need to restart kernel beforehand to clear memory
uns <- GEE(data, corstr = 'unstructured')

QIC(ind, ex, ar1, uns)
