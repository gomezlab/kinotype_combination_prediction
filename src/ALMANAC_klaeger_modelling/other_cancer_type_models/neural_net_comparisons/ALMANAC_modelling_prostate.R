
library(tidyverse)
library(here)
library(tidymodels)
library(keras)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)
library(reticulate)
library(vip)
library(recipeselectors)
library(conflicted)
library(Metrics)
library(brulee)

conflict_prefer("slice", "dplyr")
conflict_prefer("filter", "dplyr")
conflict_prefer("rmse", "Metrics")
conflict_prefer("vi", "vip")

#read in data 
ALMANAC_klaeger_CCLE_data = read_rds(here('results/ALMANAC_klaeger_models/prostate_cancer_models/ALMANAC_klaeger_data_for_ml_prostate.rds.gz'))

id_vars = ALMANAC_klaeger_CCLE_data %>% 
  select(-starts_with("act_"),
         -starts_with("exp_"),
         -starts_with("viability"))

#tuning lasso feature selection within xgboost

set.seed(2222)
folds = vfold_cv(ALMANAC_klaeger_CCLE_data, v = 5)

tic()	

this_recipe = recipe(ALMANAC_klaeger_CCLE_data) %>%
  update_role(-starts_with("act_"),
              -starts_with("exp_"),
              -starts_with("viability"),
              new_role = "id variable") %>%
  update_role(starts_with(c("act_", "exp_")), new_role = "predictor") %>% 
  update_role(viability, new_role = "outcome") %>%
  step_normalize(all_predictors()) %>%  
  step_zv(all_predictors())

mlp_spec <- mlp(
  hidden_units = tune(),
  dropout = tune(),
  epochs = tune(),
  activation = "relu"
) %>%
  set_engine("brulee") %>%
  set_mode("regression")

this_wflow <-
  workflow() %>%
  add_model(mlp_spec) %>%
  add_recipe(this_recipe)

race_ctrl = control_grid(
  save_pred = TRUE, 
  parallel_over = "everything",
  verbose = TRUE
)


set.seed(2222)
mlp_grid = grid_max_entropy(
  extract_parameter_set_dials(mlp_spec) %>% 
    update(epochs = epochs(c(10,200)),
           dropout = dropout(c(0.1, 0.9))),
  size = 10
)

results <- tune_grid(
  this_wflow,
  resamples = folds,
  grid = mlp_grid,
  control = race_ctrl
) %>% 
  collect_metrics()

toc()


write_csv(results, here('results/ALMANAC_klaeger_models/prostate_cancer_models/ALMANAC_klaeger_lasso_NN_all_tuning_results.csv'))

