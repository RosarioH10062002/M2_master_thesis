pacman::p_load(lme4, lmerTest, tidyverse, stringr,corrplot, performance,DHARMa,pbkrtest,nlme,naniar,mcclust,influence.ME)
dataset <- read.csv("behavior_eeg_true.csv")
#----------------TO CLEAN MY DATASET 
dataset <- dataset %>% select(-WEEK, -DATE, -DF, -IT, -BRIGHTNESS,-F_ORDER)

dataset <- dataset %>% select(-DPrime_Blocks)

dataset <- dataset %>% select(-Accuracy_Go, -Accuracy_NoGo)

dataset <- dataset %>% select(-Variability_Go,-Variability_NoGo,-Variability_Overall,-MeanGoRT)
#-------------------
dataset <- dataset %>%
  rename(
    #---------------- PRE OCC 
    PRE_OCC_Delta   = EEG_PRE_OCC_Delta_rel,
    PRE_OCC_Theta   = EEG_PRE_OCC_Theta_rel,
    PRE_OCC_Alpha   = EEG_PRE_OCC_Alpha_rel,
    PRE_OCC_Beta    = EEG_PRE_OCC_Beta_rel,
    PRE_OCC_Gamma   = EEG_PRE_OCC_Gamma_rel,
    PRE_OCC_TB      = EEG_PRE_OCC_Theta_Beta,
    
    #---------------- PRE FRONT 
    PRE_FRONT_Delta = EEG_PRE_FRONT_Delta_rel,
    PRE_FRONT_Theta = EEG_PRE_FRONT_Theta_rel,
    PRE_FRONT_Alpha = EEG_PRE_FRONT_Alpha_rel,
    PRE_FRONT_Beta  = EEG_PRE_FRONT_Beta_rel,
    PRE_FRONT_Gamma = EEG_PRE_FRONT_Gamma_rel,
    PRE_FRONT_TB    = EEG_PRE_FRONT_Theta_Beta,
    
    #---------------- POST OCC 
    POST_OCC_Delta   = EEG_POST_OCC_Delta_rel,
    POST_OCC_Theta   = EEG_POST_OCC_Theta_rel,
    POST_OCC_Alpha   = EEG_POST_OCC_Alpha_rel,
    POST_OCC_Beta    = EEG_POST_OCC_Beta_rel,
    POST_OCC_Gamma   = EEG_POST_OCC_Gamma_rel,
    POST_OCC_TB      = EEG_POST_OCC_Theta_Beta,
    
    #---------------- POST FRONT
    POST_FRONT_Delta = EEG_POST_FRONT_Delta_rel,
    POST_FRONT_Theta = EEG_POST_FRONT_Theta_rel,
    POST_FRONT_Alpha = EEG_POST_FRONT_Alpha_rel,
    POST_FRONT_Beta  = EEG_POST_FRONT_Beta_rel,
    POST_FRONT_Gamma = EEG_POST_FRONT_Gamma_rel,
    POST_FRONT_TB    = EEG_POST_FRONT_Theta_Beta,
    
    #---------------- DELTA OCC 
    DELTA_OCC_Delta   = EEG_DELTA_OCC_Delta_rel,
    DELTA_OCC_Theta   = EEG_DELTA_OCC_Theta_rel,
    DELTA_OCC_Alpha   = EEG_DELTA_OCC_Alpha_rel,
    DELTA_OCC_Beta    = EEG_DELTA_OCC_Beta_rel,
    DELTA_OCC_Gamma   = EEG_DELTA_OCC_Gamma_rel,
    DELTA_OCC_TB      = EEG_DELTA_OCC_Theta_Beta,
    
    #---------------- DELTA FRONT 
    DELTA_FRONT_Delta = EEG_DELTA_FRONT_Delta_rel,
    DELTA_FRONT_Theta = EEG_DELTA_FRONT_Theta_rel,
    DELTA_FRONT_Alpha = EEG_DELTA_FRONT_Alpha_rel,
    DELTA_FRONT_Beta  = EEG_DELTA_FRONT_Beta_rel,
    DELTA_FRONT_Gamma = EEG_DELTA_FRONT_Gamma_rel,
    DELTA_FRONT_TB    = EEG_DELTA_FRONT_Theta_Beta
  )
#-------------------I WOULD LIKE TO RECONSTRUCT THE NUMBER OF CORRECT AND NOT CORRECT
n_go   <- 90 * 0.80  # 72 Go trials
n_nogo <- 90 * 0.20  # 18 No-Go trials

dataset <- dataset %>%
  mutate(
    Correct_NoGo = as.integer(Mean_Accuracy_NoGo * 18),
    Incorrect_NoGo = 18 - Correct_NoGo
  )

dataset <- dataset %>%
  mutate(
    Correct_General = as.integer(Mean_Accuracy_General * 90),
    Incorrect_General = 90 - Correct_General
  )

dataset <- dataset %>%
  mutate(
    Correct_Go = as.integer(Mean_Accuracy_Go * 72),
    Incorrect_Go = 72 - Correct_Go
  )

#------------------------------
dataset$ID <- factor(dataset$ID)
dataset$Session <- factor(dataset$Session)
dataset$Session_Hour <- factor(dataset$Session_Hour)
dataset$PHASE <- factor(dataset$PHASE)
dataset$ASRS <- as.numeric(sub("/6", "", dataset$ASRS))
dataset$TIRED <- na_if(dataset$TIRED, "")
dataset$TIRED <- factor(dataset$TIRED)
dataset$GENDER <- factor(dataset$GENDER)
dataset$HRS_SLEEP <- as.numeric(dataset$HRS_SLEEP)
dataset$MOOD <- as.numeric(dataset$MOOD)
dataset$Session_Number <- as.numeric(dataset$Session_Number)
dataset$Session_c <- scale(
  dataset$Session_Number,
  center = TRUE,
  scale = FALSE
)
#-------------------------------
View(dataset)
names(dataset)
print(miss_var_summary(dataset), n = 5)
#-------------------------------
levels(dataset$ID)
levels(dataset$PHASE)
levels(dataset$ASRS)
levels(dataset$TIRED)
levels(dataset$Frequency)
levels(dataset$Session_Hour)
levels(dataset$GENDER)

#-------------------EXPLORATION
ggplot(dataset, aes(x = Session_Number, y = Mean_Accuracy_General, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")+
  facet_wrap(~ ID)

ggplot(dataset, aes(x = Session_Number, y = Mean_Accuracy_General, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")

ggplot(dataset,
       aes(Session_Number,
           Mean_Accuracy_General,
           color = factor(ID))) +
  geom_point() +
  geom_line() +
  geom_smooth(method = "lm", se = FALSE)


ggplot(dataset, aes(x = Session_Number, y = Mean_Accuracy_NoGo, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")+
  facet_wrap(~ ID)

ggplot(dataset, aes(x = Session_Number, y = Mean_Accuracy_NoGo, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")

ggplot(dataset,
       aes(Session_Number,
           Mean_Accuracy_NoGo,
           color = factor(ID))) +
  geom_point() +
  geom_line() +
  geom_smooth(method = "lm", se = FALSE)

ggplot(dataset, aes(x = Session_Number, y = DPrime_All, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")+
  facet_wrap(~ ID)

ggplot(dataset, aes(x = Session_Number, y = DPrime_All, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")

ggplot(dataset,
       aes(Session_Number,
           DPrime_All,
           color = factor(ID))) +
  geom_point() +
  geom_line() +
  geom_smooth(method = "lm", se = FALSE)


ggplot(dataset, aes(x = Session_Number, y = Mean_Variability_Go, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")+
  facet_wrap(~ ID)

ggplot(dataset, aes(x = Session_Number, y = Mean_Variability_Go, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")

ggplot(dataset,
       aes(Session_Number,
           Mean_Variability_Go,
           color = factor(ID))) +
  geom_point() +
  geom_line() +
  geom_smooth(method = "lm", se = FALSE)

ggplot(dataset, aes(x = Session_Number, y = MeanGoRT_All, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")+
  facet_wrap(~ ID)

ggplot(dataset, aes(x = Session_Number, y = MeanGoRT_All, color = PHASE)) + 
  geom_point() +
  geom_smooth(method = "lm")


ggplot(dataset,
       aes(Session_Number,
           MeanGoRT_All,
           color = factor(ID))) +
  geom_point() +
  geom_line() +
  geom_smooth(method = "lm", se = FALSE)
#-------------------------------------------------LOOK FOR CORRELATION
behavior_vars <- dataset %>%
  select(
    HRS_SLEEP,
    MOOD,
    MeanGoRT_All,
    Mean_Variability_Overall,
    DPrime_All,
    Mean_Accuracy_General,
    Session_Number
  )
pairs(behavior_vars)

cor(dataset$MeanGoRT_All,
    dataset$Mean_Accuracy_General, 
    method = "spearman")
cor(dataset$MeanGoRT_All,
    dataset$DPrime_All, 
    method = "spearman")
cor(dataset$DPrime_All,
    dataset$Mean_Accuracy_General, 
    method = "spearman")

cor_behavior <- cor(
  behavior_vars,
  method = "spearman",
  use = "pairwise.complete.obs"
)
corrplot(
  cor_behavior,
  method = "color",
  type = "upper",
  order = "hclust",
  addCoef.col = "black",
  tl.col = "black",
  tl.cex = 0.9,
  tl.srt = 45,
  number.cex = 0.7,
  col = colorRampPalette(
    c("blue","white","red")
  )(200)
)

#-------------------------------------------------DISTRIBUTION OUTCOME
par(mfrow = c(2, 2))

hist(dataset$Mean_Accuracy_General,
     main = "Mean Accuracy General",
     xlab = "Accuracy",
     col = "lightblue")

hist(dataset$DPrime_All,
     main = "DPrime",
     xlab = "DPrime",
     col = "lightgreen")

hist(dataset$MeanGoRT_All,
     main = "Mean Go RT",
     xlab = "Reaction Time (ms)",
     col = "lightpink")

hist(dataset$Mean_Variability_Go,
     main = "Mean Go RT Variability",
     xlab = "Variability",
     col = "lightgray")

par(mfrow = c(1,1))

#To look for normality 
ggplot(dataset, aes(sample = Mean_Accuracy_General)) +
  stat_qq() +
  stat_qq_line()
#-------------------------------------------------------------LMM ACCURACY 
m_simple0 <- lmer(
  Mean_Accuracy_General ~
    Session_Number + 
    PHASE +
    PHASE:Session_Number + 
    (1 | ID),
  data = dataset,
  REML = FALSE
)

m_simple1 <- lmer(
  Mean_Accuracy_General ~
    Session_Number +
    PHASE +
    PHASE:Session_Number +
    (1 + Session_Number | ID),
  data = dataset,
  REML = FALSE
)
summary(m_simple0)
summary(m_simple1)
anova(m_simple0, m_simple1)
VarCorr(m_simple1)

#------------------
# FINAL MODEL BEHAVIORAL ****
m1 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + ASRS + MOOD + HRS_SLEEP + TIRED + Session_Hour + VOLUME + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

summary(m1)
VarCorr(m1) 

#plot(resid(m1) ~ fitted(m1))
#qqnorm(resid(m1))
#qqline(resid(m1))
#acf(resid(m1))

#model_data <- model.frame(m1)
#model_data$resid <- resid(m1)
#model_data$fitted <- fitted(m1)

# ggplot(model_data,
#        aes(Session_Number,
#            resid,
#            group = ID,
#            color = ID)) +
#   geom_line() +
#   geom_hline(yintercept = 0,
#              linetype = 2)

drop1(m1, test = "Chisq")
# I AM GONNA DELETE THE SESSION_HOUR
m1_1 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + ASRS + MOOD + HRS_SLEEP + TIRED + VOLUME + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

summary(m1_1)
AIC(m1,m1_1)
BIC(m1,m1_1)
anova(m1, m1_1) # Its better without session hour 

# I AM GONNA DELETE HRS_SLEEP 
m1_2 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + ASRS + MOOD + TIRED + VOLUME + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

summary(m1_2)
AIC(m1_1,m1_2)
BIC(m1_1,m1_2)
anova(m1_1, m1_2) # Its better without hrs_sleep 
drop1(m1_2, test = "Chisq")

# I AM GONNA DELETE VOLUME 

m1_3 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + ASRS + MOOD + TIRED + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

summary(m1_3)
AIC(m1_2,m1_3)
BIC(m1_2,m1_3)
anova(m1_2, m1_3) # Its better without volume 
drop1(m1_3, test = "Chisq")

# I AM GONNA DELETE MOOD 
dataset_nomood_na <- dataset[!is.na(dataset$MOOD), ] 
m1_3_1 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + ASRS + MOOD + TIRED + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset_nomood_na,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

m1_4 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + ASRS + TIRED + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

summary(m1_4)

m1_4_1 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + ASRS + TIRED + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset_nomood_na,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m1_4_1)

AIC(m1_3_1,m1_4_1)
BIC(m1_3_1,m1_4_1)
anova(m1_3_1, m1_4_1) # Its better without volume 
drop1(m1_4, test = "Chisq")

# I AM GONNA DELETE ASRS (MY FINAL MODEL)

m1_5 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + TIRED + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m1_5)

m1_5_1 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + TIRED + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset_nomood_na,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m1_5_1)

AIC(m1_5_1,m1_4_1)
BIC(m1_5_1,m1_4_1)
anova(m1_5_1,m1_4_1)


drop1(m1_5, test = "Chisq")

VarCorr(m1) 

plot(resid(m1) ~ fitted(m1))
qqnorm(resid(m1))
qqline(resid(m1))
acf(resid(m1))

model_data <- model.frame(m1)

model_data$resid <- resid(m1)
model_data$fitted <- fitted(m1)

ggplot(model_data,
       aes(Session_Number,
           resid,
           group = ID,
           color = ID)) +
  geom_line() +
  geom_hline(yintercept = 0,
             linetype = 2)




#Parte de la variabilidad de los sujetos ahora es explicada por los factores y no por el slope 
# EEG ANALYSIS 
#----------------------------- EEG TENDENCIES CORRELATION 
pre_vars <- dataset %>%
  select(starts_with("PRE_"))

pairs(pre_vars)

cor_pre <- cor(
  pre_vars,
  method = "spearman",
  use = "pairwise.complete.obs"
)

corrplot(
  cor_pre,
  method = "color",
  type = "upper",
  order = "hclust",
  addCoef.col = "black",
  tl.col = "black",
  tl.cex = 0.9,
  tl.srt = 45,
  number.cex = 0.6,
  col = colorRampPalette(c("blue","white","red"))(2000)
)



post_vars <- dataset %>%
  select(starts_with("POST_"))

pairs(post_vars)

cor_post <- cor(
  post_vars,
  method = "spearman",
  use = "pairwise.complete.obs"
)

corrplot(
  cor_post,
  method = "color",
  type = "upper",
  order = "hclust",
  addCoef.col = "black",
  tl.col = "black",
  tl.cex = 0.9,
  tl.srt = 45,
  number.cex = 0.6,
  col = colorRampPalette(c("blue","white","red"))(2000)
)


delta_vars <- dataset %>%
  select(starts_with("DELTA"))
pairs(delta_vars)

cor_delta <- cor(
  delta_vars,
  method = "spearman",
  use = "pairwise.complete.obs"
)

corrplot(
  cor_delta,
  method = "color",
  type = "upper",
  order = "hclust",
  addCoef.col = "black",
  tl.col = "black",
  tl.cex = 0.9,
  tl.srt = 45,
  number.cex = 0.6,
  col = colorRampPalette(c("blue","white","red"))(2000)
)

names(pre_vars)

#-------------------------------------------I AM GONNA COMPUTE PER EACH CLUSTER THE MODEL THAT I OBTAINED BEFORE
eeg_dataset <- dataset %>%
  select(starts_with("DELTA"))

corr <- cor(
  eeg_dataset,
  use = "pairwise.complete.obs",
  method = "spearman"
)

hc <- hclust(
  as.dist(1 - abs(corr))
)

plot(hc)
clusters <- cutree(hc, h = 0.5)
data.frame(
  Variable = names(clusters),
  Cluster = clusters
)

cluster_list <- split(
  names(clusters),
  clusters
)

cluster_list
#-----------------------------------------------------------------------------
evaluate_cluster <- function(cluster_vars){
  
  results <- data.frame()
  
  for(var in cluster_vars){
    
    dataset_tmp <- dataset %>%
      dplyr::select(
        Mean_Accuracy_General,
        Session_Number,
        PHASE,
        TIRED,
        Mean_Variability_Overall,
        MeanGoRT_All,
        ID,
        all_of(var)
      ) %>%
      tidyr::drop_na()

    
    base_model <- lmer(
      
      Mean_Accuracy_General ~
        
        Session_Number +
        PHASE +
        PHASE:Session_Number +
        TIRED +
        Mean_Variability_Overall +
        MeanGoRT_All +
        
        (1 + Session_Number | ID),
      
      data = dataset_tmp,
      
      REML = FALSE,
      
      control = lmerControl(
        optimizer = "bobyqa"
      )
      
    )
    
    formula_txt <- paste(
      
      "Mean_Accuracy_General ~",
      
      "Session_Number +",
      "PHASE +",
      "PHASE:Session_Number +",
      "TIRED +",
      "Mean_Variability_Overall +",
      "MeanGoRT_All +",
      
      var,
      
      "+ (1 + Session_Number | ID)"
      
    )
    
    model <- lmer(
      
      as.formula(formula_txt),
      
      data = dataset_tmp,
      
      REML = FALSE,
      
      control = lmerControl(
        optimizer = "bobyqa"
      )
      
    )

    comparison <- anova(base_model, model)
    
    coef_table <- summary(model)$coefficients
    
    results <- rbind(
      
      results,
      
      data.frame(
        
        Variable = var,
        
        N = nrow(dataset_tmp),
        
        Estimate = coef_table[var,"Estimate"],
        
        Std_Error = coef_table[var,"Std. Error"],
        
        t = coef_table[var,"t value"],
        
        p = coef_table[var,"Pr(>|t|)"],
        
        AIC = AIC(model),
        
        BIC = BIC(model),
        
        DeltaAIC = AIC(base_model) - AIC(model),
        
        DeltaBIC = BIC(base_model) - BIC(model),
        
        LRT = comparison$Chisq[2],
        
        LRT_p = comparison$`Pr(>Chisq)`[2]
        
      )
      
    )
    
  }
  
  results <- results %>%
    
    mutate(
      
      FDR = p.adjust(LRT_p, method = "BH")
      
    ) %>%
    
    arrange(
      
      LRT_p,
      
      desc(DeltaAIC),
      
      p
      
    )
  
  return(results)
  
}

cluster_results <- lapply(
  cluster_list,
  evaluate_cluster
)

cluster_results
#---------------------------------------------------TRY WITH THE TWO IMPORTANT VARIABLES OF EEG 
dataset_new_eeg <- dataset %>%
  drop_na(
    DELTA_FRONT_TB,
    DELTA_FRONT_Alpha
  )
m1_5_11 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + TIRED + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset_new_eeg,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m1_5_11)


m2 <- lmer(Mean_Accuracy_General ~
  Session_Number +
  PHASE +
  PHASE:Session_Number +
  TIRED +
  Mean_Variability_Overall +
  MeanGoRT_All +
  DELTA_FRONT_TB +
  DELTA_FRONT_Alpha +
  (1 + Session_Number | ID),
  data = dataset_new_eeg,
  REML = FALSE,
  control = lmerControl(
    optimizer = "bobyqa"
  ))

summary(m2)
check_collinearity(m2)

AIC(m1_5_11, m2) 
BIC(m1_5_11, m2)
anova(m1_5_11, m2) # EEG improves the model 
drop1(m2, test = "Chisq")

#---------------------------------------------------TRY WITH ONLY DELTA_FRONT_ALPHA
dataset_new_eeg <- dataset %>%
  drop_na(
    DELTA_FRONT_Alpha
  )
m1_5_11 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + TIRED + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset_new_eeg,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m1_5_11)


m2 <- lmer(Mean_Accuracy_General ~
             Session_Number +
             PHASE +
             PHASE:Session_Number +
             TIRED +
             Mean_Variability_Overall +
             MeanGoRT_All +
             DELTA_FRONT_Alpha +
             (1 + Session_Number | ID),
           data = dataset_new_eeg,
           REML = FALSE,
           control = lmerControl(
             optimizer = "bobyqa"
           ))

summary(m2)
check_collinearity(m2)

AIC(m1_5_11, m2) 
BIC(m1_5_11, m2)
anova(m1_5_11, m2) # EEG improves the model 
drop1(m2, test = "Chisq")

#---------------------------------------------------TRY WITH ONLY DELTA_FRONT_TB (MY FINAL MODEL)
dataset_new_eeg <- dataset %>%
  drop_na(
    DELTA_FRONT_TB
  )
m1_5_11 <- lmer(
  Mean_Accuracy_General ~ Session_Number + PHASE + PHASE:Session_Number + TIRED + Mean_Variability_Overall + MeanGoRT_All + (1+ Session_Number | ID),
  data = dataset_new_eeg,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)
summary(m1_5_11)


m2 <- lmer(Mean_Accuracy_General ~
             Session_Number +
             PHASE +
             PHASE:Session_Number +
             TIRED +
             Mean_Variability_Overall +
             MeanGoRT_All +
             DELTA_FRONT_TB +
             (1 + Session_Number | ID),
           data = dataset_new_eeg,
           REML = FALSE,
           control = lmerControl(
             optimizer = "bobyqa"
           ))

summary(m2)
check_collinearity(m2)

AIC(m1_5_11, m2) 
BIC(m1_5_11, m2)
anova(m1_5_11, m2) # EEG improves the model 
drop1(m2, test = "Chisq")
final_model <-m2
#-------------------------------------------------------------------------RESIDUAL ANALYSIS 
model_data <- model.frame(m2)
model_data$residuals <- resid(m2)
model_data$fitted <- fitted(m2)

ggplot(
  model_data,
  aes(
    x = Session_Number,
    y = residuals,
    color = ID,
    group = ID
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    color = "black"
  ) +
  theme_classic(base_size = 14) +
  labs(
    x = "Session Number",
    y = "Residuals",
    color = "Participant"
  )
ggplot(
  model_data,
  aes(
    x = fitted,
    y = residuals
  )
) +
  geom_point(size = 2) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  theme_classic()



par(mfrow = c(2,3))

for(id in levels(model_data$ID)){
  
  acf(
    model_data$residuals[
      model_data$ID == id
    ],
    main = paste("ID", id)
  )
  
}

par(mfrow = c(1,1))

plot(
  fitted(m2),
  resid(m2)
)

abline(h = 0, lty = 2)


#-------------------------------------------------------------------------------TRYING WITH GLMM TO SEE IF I GET THE SAME RESULT
dataset_new_eeg <- dataset %>%
  drop_na(
    DELTA_FRONT_TB
  )
glmm1 <- glmer(
  cbind(Correct_General, Incorrect_General) ~
    Session_Number +
    PHASE +
    PHASE:Session_Number +
    TIRED +
    Mean_Variability_Overall +
    MeanGoRT_All +
    (1 + Session_Number | ID),
  
  family = binomial,
  
  data = dataset_new_eeg,
  
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 500000)
  )
)
summary(glmm1)


glmm2 <- glmer(
  cbind(Correct_General, Incorrect_General) ~
             Session_Number +
             PHASE +
             PHASE:Session_Number +
             TIRED +
             Mean_Variability_Overall +
             MeanGoRT_All +
             DELTA_FRONT_TB +
             (1 + Session_Number | ID),
  family = binomial(link = "logit"),
  data = dataset_new_eeg,
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 500000)
  )
  )

summary(glmm2)
check_collinearity(glmm2)

AIC(glmm1, glmm2) 
BIC(glmm1, glmm2)
anova(glmm1, glmm2) # EEG does NOT improve the model 

#-------------------------------------------------------------------------------------------
