library(torch)
library(tidyverse)
library(here)
library(tabnet)
library(finetune)
library(tidymodels)
library(tictoc)
library(recipeselectors)

cancer_types = c("breast",
                 "lung",
                 "ovarian",
                 "melanoma",
                 "colon",
                 "cns",
                 "renal",
                 "prostate") %>% 
  as.data.frame() %>% 
  rename(cancer_type = 1) %>% 
  mutate(data_path = here('results/ALMANAC_klaeger_models',
                          paste0(cancer_type, '_cancer_models'),
                          paste0('ALMANAC_klaeger_data_for_ml_', cancer_type, '.rds.gz'))) %>% 
  mutate(results_path = here('results/ALMANAC_klaeger_models',
                          paste0(cancer_type, '_cancer_models'),
                          'ALMANAC_klaeger_lasso_xgboost_all_tuning_results_tabnet.csv'))

for (i in 1:dim(cancer_types)[1]) {
  
  tic()
  this_cancer_type = cancer_types$cancer_type[i]
  this_data_path = cancer_types$data_path[i]
  this_results_path = cancer_types$results_path[i]
  
  #read in data 
  ALMANAC_klaeger_CCLE_data = read_rds(this_data_path)
  
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
    step_zv(all_predictors()) %>% 
    step_select_linear(all_predictors(), outcome = "viability", engine = "glmnet", penalty = tune(), mixture = tune(), top_p = tune())
  
  tabnet_spec <- tabnet() %>%
    set_engine("torch", verbose = TRUE) %>%
    set_mode("regression")
  
  this_wflow <-
    workflow() %>%
    add_model(tabnet_spec) %>%
    add_recipe(this_recipe)
  
  race_ctrl = control_race(
    save_pred = TRUE,
    parallel_over = "everything",
    verbose = TRUE
  )
  
  set.seed(2222)
  recipe_grid = grid_max_entropy(
    extract_parameter_set_dials(this_recipe) %>% 
      update(top_p = top_p(c(5,2000))), 
    size = 10
  )
  
  results <- tune_grid(
    this_wflow,
    resamples = folds,
    grid = recipe_grid,
    control = race_ctrl
  ) %>% 
    collect_metrics()
  
  write_csv(results, this_results_path)
  
  toc()
  
}