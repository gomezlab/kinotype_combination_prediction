library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)
library(argparse)
library(xgboost)

full_dataset = read_rds(here('results/ALMANAC_klaeger_L1000_data_for_ml.rds.gz'))
feats =  vroom(here('results/ALMANAC_klaeger_L1000_models/feature_selection/lasso_selected_features.csv'))
set.seed(2222)
folds = vfold_cv(full_dataset, v = 5)

tic()	

dir.create(here('results/ALMANAC_klaeger_L1000_models/activation_expression/regression/', 
                sprintf('xgboost/results')), 
           showWarnings = F, recursive = T)

full_output_file = here('results/ALMANAC_klaeger_L1000_models/activation_expression/regression/xgboost/results/lasso_feature_results_inital.rds.gz')

id_vars = full_dataset %>% 
  select(-starts_with(c("act_", "pert_")), -viability)

this_recipe = recipe(viability ~ ., full_dataset) %>%
  update_role(any_of(names(id_vars)),
              new_role = "id variable") %>% 
  step_select(viability, any_of(c(names(id_vars), feats$feature))) %>% 
  step_zv(all_predictors())

xgb_spec <- boost_tree(
  trees = tune(), 
  tree_depth = tune()
) %>% 
  set_engine("xgboost", tree_method = "gpu_hist") %>% 
  set_mode("regression")

set.seed(2222)
xgb_grid = read_rds(here('results/hyperparameter_grids/xgb_grid.rds'))

this_wflow <-
  workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(this_recipe)

race_ctrl = control_grid(
  save_pred = TRUE, 
  parallel_over = "everything",
  verbose = TRUE
)

results <- tune_grid(
  this_wflow,
  resamples = folds,
  grid = xgb_grid,
  control = race_ctrl
) %>% 
  #collect_metrics() %>% 
  write_rds(full_output_file, compress = "gz")

toc()

