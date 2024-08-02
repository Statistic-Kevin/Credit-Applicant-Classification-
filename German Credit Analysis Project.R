# load packages----
library(tidyverse)
library(tidymodels)
library(kableExtra)
library(reshape2)
library(ggpubr)
library(glmnet)
library(randomForest)
library(earth)
library(baguette)
library(dbarts)
library(viridis)
library(GGally)
library(vip)

### 2. Data Description & Preparation



# load data, clean up column names----
credit_raw <- read.csv2("GermanCredit.csv", #adjust filepath
                        header = T) %>% select(-OBS.)

names(credit_raw) <- tolower(names(credit_raw))
names(credit_raw) <- gsub(x = colnames(credit_raw), pattern = ".", replacement = "_", fixed = T)

# check the data for strange values----
# summary(credit_raw)
# inspect suspicious values
# table(credit_raw$education)
# table(credit_raw$guarantor)
# hist(credit_raw$age)

# remove obs. w/ impossible values, rm variable w/ very rare category----
credit <- credit_raw %>% filter(age < 100, education %in% c(0, 1), guarantor %in% c(0, 1)) %>%
  select(-foreign)

# check if reference category of credit purpose is already left out----
#credit %>% select(new_car, used_car, furniture, radio_tv, education, retraining) %>% 
#   mutate(s = rowSums(.), .keep = "none") %>% table()
# yes

# convert all categorical variables to factors----
num_vars <- c("duration", "amount", "install_rate", "age", "num_credits", "num_dependents")
credit_fac <- credit %>% mutate(across(-c(all_of(num_vars)), ~as.factor(.)))
# summary(credit)


# create dataset ready for modeling----

credit_modeling <- model.matrix(response ~ . , data = credit_fac) %>%
  as.data.frame() %>% select(-1) %>% bind_cols(., response=credit_fac$response)




# compute correlation matrix, check high values----
cor_mtrx <- round(cor(credit_modeling %>% mutate(response=as.numeric(response)-1)), 2) 
cor_mtrx[lower.tri(cor_mtrx)]<- NA
diag(cor_mtrx) <- NA
cor_mtrx <- cor_mtrx %>% melt(na.rm = T)
cor_mtrx %>% filter(abs(value) > 0.5)
# nothing dramatic

# histogram of continuous predictors----
contvars <- c("age", "amount", "duration")
pld <- credit_fac %>% select(all_of(contvars), response) %>%
  pivot_longer(-c(response), values_to = "value", names_to = "var")

ggplot(pld, aes(x=value, color=response, fill=response)) +
  geom_histogram(position="dodge", alpha=0.4, bins = 20) +
  scale_color_manual(name="credit", 
                     values = c("red4", "seagreen4"), labels=c("bad", "good")) +
  scale_fill_manual(name="credit", 
                    values = c("red4", "seagreen4"), labels=c("bad", "good")) +
  theme_classic() +
  facet_wrap(vars(var), scales="free_x", nrow = 1)


### 3 Modelling

mods <- c("BaggedMARS", "", "BART", "", "ElasticNetLogit", "", "RandomForest", "")
engs <- c("earth", "", "dbarts", "", "glmnet", "", "randomForest", "")
hyppars <- c("num_terms", "prod_degree",
             "prior_terminal_node_coef", "prior_outcome_range",
             "mixture", "log_penalty",
             "mtry", "min_n"
)
explan <- c("max. number of terms before pruning", "max. degree of interaction", 
            "prior on probability that tree node is terminal", 
            "prior on range of predicted outcomes",
            "mixture of Ridge (0) and LASSO (1) penalty",
            "Natural Log of regularization parameter",
            "# of predictors considered for tree splits",
            "min. # of observations in terminal nodes"
)

grids <- c("10-60", "1-3", "0.09-0.99", "0.05-4.05", "0-1", "(-25)-0", "1-30", "1-30")


# create tables for the final document
htbl <- data.frame(
  model = mods, hyperparameter = hyppars, explanation=explan, range=grids, package = engs) 

htbl %>% kbl(booktabs = T, caption = "Decription of hyperparameters and tuning grids",
             format = "latex", linesep = "") %>%
  kable_styling(font_size=9.5, latex_options = "hold_position") %>% 
  column_spec(column = 2, italic = T) %>%
  column_spec(column = 5, italic = T) %>%
  kable_paper(full_width = F)

# test/train split & CV----
set.seed(457)
credit_split <- initial_split(credit_modeling, prop = 0.8)
credit_train <- training(credit_split)
credit_test <- testing(credit_split)
credit_trainCV <- vfold_cv(credit_train, v=5)

# set up different models----

simple_logit_mod <- logistic_reg(
  engine = "glm"
)

logitnet_mod <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  penalty = tune(),
  mixture = tune()
)

rf_mod <- rand_forest(
  mode = "classification",
  engine = "randomForest",
  mtry = tune(),
  min_n = tune()
)

mars_mod <- bag_mars(
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth", times=20) %>%
  set_mode("classification")

bart_mod <- parsnip::bart(
  mode = "classification",
  engine = "dbarts",
  prior_terminal_node_coef = tune(),
  prior_outcome_range = tune()
)

# set up hyperparameter search grids and options----

grid_logitnet <- grid_regular(
  mixture(range = c(0, 1)),
  penalty(),
  levels = 11
)

grid_rf <- grid_regular(
  mtry(range = c(1, 30)),
  min_n(range = c(1, 30)),
  levels = 10
)

grid_mars <- grid_regular(
  num_terms(range = c(11, 60)),
  prod_degree(range = c(1, 3)),
  levels = 10
)

grid_bart <- grid_regular(
  prior_terminal_node_coef(range = c(0.09, 0.99)),
  prior_outcome_range(range = c(0.05, 4.05)),
  levels = 10
)

# options
ctrl <- control_grid(verbose = F, save_pred = T)

# create workflow object----

all_workflows <- 
  workflow_set(
    preproc = list("formula" = response ~ .),
    models = list(simplelogit = simple_logit_mod, logitnet = logitnet_mod,
                  rf = rf_mod, mars = mars_mod, bart = bart_mod
    )
  ) %>%
  option_add(id = "formula_simplelogit") %>%
  option_add(id = "formula_mars", control = ctrl, grid = grid_mars) %>%
  option_add(id = "formula_bart", control = ctrl, grid = grid_bart) %>%
  option_add(id = "formula_logitnet", control = ctrl, grid = grid_logitnet) %>%
  option_add(id = "formula_rf", control = ctrl, grid = grid_rf)


#***************************** tuning - run this outside the r script !----
# **************took about 4 hours

#res_wf_raw <- all_workflows %>%
#  workflow_map(resamples = credit_trainCV, verbose = T, metrics = metric_set(roc_auc)) 
# save results to work with later
#save(res_wf_raw, 
#  file = "res_wf_raw.RData")  # adjust filepath



#####4 Model Diagnostics

# load results----
load("res_wf_raw.RData") # adjust filepath

res_wf <- res_wf_raw %>% 
  select(wflow_id, result) %>% unnest(result) %>%
  select(-c(splits, .notes, .predictions)) %>% unnest(.metrics) %>% 
  rename(auroc = .estimate) %>% select(-c(.estimator, .metric)) %>%
  group_by(across(-c(id, auroc))) %>% summarise(auc = mean(auroc)) %>% 
  ungroup() %>% mutate(log_penalty = log(penalty)) # for plotting

# contourplots----
wflows <- c("formula_mars", "formula_bart", "formula_logitnet", "formula_rf")
tuning_params <- list(
  BaggedMARS = c("num_terms", "prod_degree"),
  BART = c("prior_terminal_node_coef", "prior_outcome_range"),
  ElasticNetLogit = c("log_penalty", "mixture"),
  RandomForest = c("mtry", "min_n")
)

auc_lims <- c(0.7, 1) # don't display really bad models
auc_breaks <- seq(auc_lims[1], auc_lims[2], 0.005)

for(i in 1:length(wflows)){
  
  res <- res_wf %>% filter(wflow_id == wflows[i]) %>%
    select(all_of(tuning_params[[i]]), auc)
  
  besthyp <- res %>% arrange(auc) %>% slice_tail(n=1) %>% select(-auc) %>% as.numeric()
  
  hp1 <- tuning_params[[i]][1]
  hp2 <- tuning_params[[i]][2]
  
  tune_auc_pl <- ggplot(res, aes(.data[[hp1]], .data[[hp2]], z=auc)) +
    geom_contour_filled(breaks = auc_breaks) +
    geom_contour(breaks = auc_breaks, color = "darkblue", alpha = 0.3,
                 lineend = "round", linejoin = "round") +
    scale_fill_viridis_d(drop = F, option = "turbo", direction = 1) +
    geom_point(aes(.data[[hp1]], .data[[hp2]]),
               shape = 3, size = rel(0.8), alpha = 0.3) +
    theme_classic() + ggtitle(names(tuning_params[i])) +
    theme(legend.position = "none", 
          plot.title = element_text(hjust = 0.5, size=rel(1), face = "bold"),
          axis.title = element_text(size=rel(0.9))
    ) +
    annotate(geom="text", x=besthyp[1], y=besthyp[2], label="X", color="red", size = 8, face="bold") +
    annotate(geom="text", x=besthyp[1], y=besthyp[2], label="O", color="red", size = 5, face="bold")
  
  
  assign(paste0("pl_", i), tune_auc_pl)
}

# create plot with same colors and color limits, extract legend
pl_leg <- ggplot((res_wf %>% select(mixture, penalty, auc) %>% na.omit()), aes(mixture, penalty, color = auc)) +
  geom_point() +
  scale_color_viridis(option = "turbo", direction = 1, limits = auc_lims, name = "AuROC") +
  theme(legend.title = element_text(size=rel(0.9)), legend.text = element_text(size=rel(0.8)))

leg <- get_legend(pl_leg)

# put together plots and legend
hyppl <- ggarrange(pl_1, pl_2, pl_3, pl_4)
hyppl <- ggarrange(hyppl, leg, ncol = 2, widths = c(6,1))
annotate_figure(hyppl, bottom = text_grob(
  "For better visibility, surface areas with an AuROC < 0.7 are not displayed.",hjust = 1, x = 0.99,
  face = "italic", size = 12))

# fit best models on the whole training set, get predicitions & evaluate on test set----

# intialize df of predictions on test set
preds_test <- as.data.frame(matrix(ncol = 8))
# intialize dataframe of ROC curves
roc_c <- data.frame(.threshold=NA, specificity=NA, sensitivity=NA, model=NA)
# initialize vector of AuROC's
auc <- numeric(length(wflows))

for(i in 1:length(wflows)){
  
  # extract best hyperparam specification for each model
  best_results <- res_wf_raw %>% 
    extract_workflow_set_result(id = wflows[i]) %>% 
    select_best()
  
  # fit these best models on whole train set, predict on test set
  test_results <- res_wf_raw %>% 
    extract_workflow(wflows[i]) %>% 
    finalize_workflow(best_results) %>% 
    last_fit(split = credit_split)
  
  # save predictions on test set for further analysis
  preds <- test_results %>% collect_predictions() %>% mutate(model = names(tuning_params)[i])
  names(preds_test) <- names(preds)
  preds_test <- bind_rows(preds_test, preds)
  
  # compute the ROC
  roc <- roc_curve(preds, truth = response, .pred_1, event_level = "second") %>% 
    mutate(model = names(tuning_params)[i])
  roc_c <- bind_rows(roc_c, roc)
  
  # calculate AuROC
  auc[i] <- round(roc_auc_vec(preds$response, preds$.pred_1, event_level = "second"), 3)
}

preds_test <- na.omit(preds_test) %>% select(model, .row, .pred_1, response) 
roc_c <- na.omit(roc_c)

# plot ROC curves together----

t <- " (AuROC: "
nam <- names(tuning_params)

ggplot(roc_c, aes(x = 1 - specificity, y = sensitivity, color = model)) + 
  geom_path(linewidth = 0.6) +
  geom_abline(lty = 2) + 
  coord_equal() +
  scale_color_manual(labels = c(paste0(nam[1], t, auc[1], ")"), paste0(nam[2], t, auc[2], ")"),
                                paste0(nam[3], t, auc[3], ")"), paste0(nam[4], t, auc[4], ")")),
                     values = c("firebrick3", "steelblue4", "darkgoldenrod2", "forestgreen")) +
  xlab("False positive rate") +
  ylab("True positive rate") +
  theme_classic() +
  theme(legend.position = "right", legend.title = element_blank(), 
        legend.text=element_text(size=rel(0.8)), axis.title = element_text(size = rel(0.9))) +
  guides(color=guide_legend(nrow=length(wflows)))

# histograms of predictions on test set----
ggplot(preds_test, aes(x=.pred_1, color=response, fill=response)) +
  geom_histogram(position="dodge", alpha=0.4, binwidth = 0.04) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_color_manual(name="actual credit", 
                     values = c("red4", "seagreen4"), labels=c("bad", "good")) +
  scale_fill_manual(name="actual credit", 
                    values = c("red4", "seagreen4"), labels=c("bad", "good")) +
  xlab("predicted probability of good credit") +
  theme_classic() +
  facet_wrap(vars(model))

# scatterplot matrix of predictions on test set----
preds_test2 <- preds_test %>% pivot_wider(names_from = model, values_from = .pred_1)

# function to extract lower triangle from scatterplot matrix
# from: https://stackoverflow.com/questions/42654928/how-to-show-only-the-lower-triangle-in-ggpairs
gpairs_lower <- function(g){
  g$plots <- g$plots[-(1:g$nrow)]
  g$yAxisLabels <- g$yAxisLabels[-1]
  g$nrow <- g$nrow -1
  
  g$plots <- g$plots[-(seq(g$ncol, length(g$plots), by = g$ncol))]
  g$xAxisLabels <- g$xAxisLabels[-g$ncol]
  g$ncol <- g$ncol - 1
  
  g
}

pl <- ggpairs(preds_test2, aes(colour=response, alpha=0.2), columns = 3:6, 
              diag = "blankDiag", upper = "blank", legend = c(2,1)) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0,1)) +
  scale_color_manual(name="actual credit", 
                     values = c("red4", "seagreen4"), labels=c("bad", "good")) +
  scale_fill_manual(name="actual credit", 
                    values = c("red4", "seagreen4"), labels=c("bad", "good")) +
  theme_classic2() +
  scale_alpha(guide = "none") +
  ylab("predicted probability of good credit") +
  xlab("predicted probability of good credit")

gpairs_lower(pl)

# ROC curves for averaged predictions
pred_avg <- preds_test %>% pivot_wider(names_from = "model", values_from = ".pred_1") %>% mutate(
  AveragedPredictions = (BaggedMARS+BART+ElasticNetLogit+RandomForest)/4)
pred_avg_roc <- roc_curve(pred_avg, truth = response, AveragedPredictions, event_level = "second") %>% 
  mutate(model = "AveragedPredictions")
auc_avged <- round(roc_auc_vec(pred_avg$response, pred_avg$AveragedPredictions, event_level = "second"), 3)

# for a standard logistic regression
simplelogit_preds <- predict(glm(response~., data = credit_train, family = binomial(link="logit")), newdata=credit_test, type="response")
simplelogit_preds <- cbind(simplelogit_preds, credit_test$response) %>% as.data.frame() %>% rename(.pred_1=1, response=2) %>%
  mutate(response=as.factor(response))
simplelogit_roc <-  roc_curve(simplelogit_preds, truth = response, .pred_1, event_level = "second") %>% 
  mutate(model = "SimpleLogit")
auc_simplelogit <- round(roc_auc_vec(simplelogit_preds$response, simplelogit_preds$.pred_1, event_level = "second"), 3)


# stacked model
res_wf_stack <- res_wf_raw %>% 
  select(wflow_id, result) %>% unnest(result) %>%
  select(-c(splits, .notes, .metrics)) %>% unnest(.predictions) %>% select(-c(.pred_0, id))


stack_dat <- data.frame(model=NA, .pred_1=NA, .row=NA)

for(i in 1:length(wflows)){
  best_res <- res_wf_raw %>% 
    extract_workflow_set_result(id = wflows[i]) %>% 
    select_best()
  
  oof_preds <- res_wf_stack %>% filter(wflow_id==wflows[i], .config==best_res$.config[1]) %>%
    mutate(model=names(tuning_params)[i]) %>%
    select(model, .pred_1, response, .row)
  
  stack_dat <- bind_rows(stack_dat, oof_preds %>% select(-response))
}

oof_resp <- oof_preds %>% arrange(.row) %>% select(response)
stack_dat <- na.omit(stack_dat) %>% pivot_wider(names_from = model, values_from = .pred_1) %>%
  arrange(.row) %>% select(-.row)
stack_dat <- bind_cols(stack_dat, oof_resp)

# build meta-model----
set.seed(174)
stackmod <- cv.glmnet(as.matrix(stack_dat %>% select(-response)), stack_dat$response, family = binomial,
                      alpha=0, nlambda = 20, lower.limits = 0, standardize=F, type.measure = "deviance", nfolds = 10)

# check stacking weights
#coef(stackmod)

# get predictions of base models on test set
preds_test_wide <- preds_test %>% select(-response) %>%
  pivot_wider(names_from = model, values_from = .pred_1) %>% select(-.row)

# stacked predictions on test set
stack_preds_test <- cbind(predict(stackmod, newx=as.matrix(preds_test_wide), s = "lambda.min", type="response"),
                          credit_test$response) %>% as.data.frame() %>% rename(.pred_1=1, response=2)
stack_roc <- roc_curve(stack_preds_test, truth=as.factor(response), .pred_1, event_level = "second") %>%
  mutate(model="Stacked")
auc_stack <- round(roc_auc_vec(as.factor(stack_preds_test$response), stack_preds_test$.pred_1, event_level = "second"), 3)


roc_c2 <- bind_rows(pred_avg_roc, simplelogit_roc, stack_roc)
auc_2 <- c(auc_avged, auc_simplelogit, auc_stack)

# plot ROC curves together----

t <- " (AuROC: "

ggplot(roc_c2, aes(x = 1 - specificity, y = sensitivity, color = model)) + 
  geom_path(linewidth = 0.6) +
  geom_abline(lty = 2) + 
  coord_equal() +
  scale_color_manual(labels = c(paste0("Averaged Predictions", t, auc_2[1], ")"), 
                                paste0("Simple Logit", t, auc_2[2], ")"), 
                                paste0("Stacked Model", t, auc_2[3], ")")),
                     values = c("limegreen", "gray60", "darkviolet")) +
  xlab("False positive rate") +
  ylab("True positive rate") +
  theme_classic() +
  theme(legend.position = "right", legend.title = element_blank(), 
        legend.text=element_text(size=rel(0.8)), axis.title = element_text(size = rel(0.9))) +
  guides(color=guide_legend(nrow=3))

#Elastic Net Logit VIPs

br <- res_wf_raw %>% 
  extract_workflow_set_result(id = "formula_logitnet") %>% 
  select_best()

logitfit <- res_wf_raw %>% 
  extract_workflow("formula_logitnet") %>% 
  finalize_workflow(br) %>% 
  last_fit(split = credit_split) %>% 
  extract_fit_parsnip() 

vimp <- logitfit %>% vip::vi(method="model")  

ggplot(vimp, aes(x=reorder(Variable, Importance), weight=Importance, fill=as.factor(Sign))) + 
  geom_bar() +
  scale_fill_manual(name="effect", values=c("red4", "seagreen4")) +
  xlab("") +
  ylab("Importance Score") +
  theme_classic() +
  coord_flip()
