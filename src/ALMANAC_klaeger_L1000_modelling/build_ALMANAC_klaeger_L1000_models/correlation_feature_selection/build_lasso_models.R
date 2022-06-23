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
library(conflicted)
conflict_prefer("slice", "dplyr")

tic()
parser <- ArgumentParser(description='Process input paramters')
parser$add_argument('--feature_num', default = 100, type="integer")

args = parser$parse_args()
print(sprintf('Features: %02d',args$feature_num))

dir.create(here('results/ALMANAC_klaeger_L1000_models/activation_expression/regression/', 
                sprintf('lasso/results')), 
           showWarnings = F, recursive = T)

full_output_file = here('results/ALMANAC_klaeger_L1000_models/activation_expression/regression/lasso/results', 
                        sprintf('%dfeat.rds.gz',args$feature_num))

full_dataset = read_rds(here('results/ALMANAC_klaeger_L1000_data_for_ml_5000feat.rds.gz'))
feats =  vroom(here('results/ALMANAC_klaeger_L1000_models/feature_selection/correlation_selected_features.csv'))
folds = read_rds(here('results/cv_folds/ALMANAC_klaeger_L1000_folds.rds.gz'))
lr_grid = read_rds(here('results/hyperparameter_grids/lr_grid.rds'))

id_vars = full_dataset %>% 
  select(-starts_with(c("act_", "pert_")), -viability)

this_recipe = recipe(viability ~ ., full_dataset) %>%
  update_role(any_of(names(id_vars)),
              new_role = "id variable") %>% 
  step_select(viability, any_of(c(names(id_vars), feats$feature[1:args$feature_num]))) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

lr_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>% 
  set_mode("regression")

this_wflow <-
  workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(this_recipe) 

race_ctrl = control_grid(
  save_pred = TRUE, 
  parallel_over = "everything",
  verbose = TRUE
)

results <- tune_grid(
  this_wflow,
  resamples = folds,
  grid = lr_grid,
  control = race_ctrl
) %>% 
  collect_metrics() %>% 
  write_rds(full_output_file, compress = "gz")

toc()

