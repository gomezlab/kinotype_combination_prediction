library(tidyverse)
library(here)
library(tabnet)
library(torch)
library(finetune)
library(tidymodels)
library(tictoc)
torch_manual_seed(2222) 

full_dataset = read_rds(here('results/ALMANAC_klaeger_L1000_data_for_ml.rds.gz'))

set.seed(2222)
folds = vfold_cv(full_dataset, v = 5)

this_recipe = recipe(viability ~ ., full_dataset) %>%
  update_role(-starts_with("act_"),
              -starts_with("pert_"),
              -starts_with("viability"),
              new_role = "id variable") %>% 
  step_zv(all_predictors())

tabnet_spec <- tabnet(epochs = tune(),
                      decision_width = tune(),
                      attention_width = tune(),
                      num_steps = tune(),
                      penalty = tune(),
                      batch_size = 91545,
                      virtual_batch_size = 91545,
                      momentum = 0.02,
                      feature_reusage = tune(),
                      learn_rate = 0.003,
                      num_independent = 3,
                      num_shared = 3) %>% 
  set_engine("torch", verbose = TRUE) %>%
  set_mode("regression")

this_wflow <-
  workflow() %>%
  add_model(tabnet_spec) %>%
  add_recipe(this_recipe)

set.seed(2222)
grid <-
  this_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    epochs =epochs(range = c(50,200)),
    decision_width = decision_width(range = c(16, 64)),
    attention_width = attention_width(range = c(16, 64)),
    num_steps = num_steps(range = c(3, 10))
  ) %>%
  grid_max_entropy(size = 10) %>% 
  mutate(attention_width = decision_width)

ctrl <- control_race(
  parallel_over = "everything",
  save_pred = TRUE,
  verbose = TRUE)

fit <- this_wflow %>% 
  tune_race_anova(
    resamples = folds, 
    grid = grid,
    control = ctrl,
    metrics = metric_set(rsq)
  )

cv_metrics_regression = collect_metrics(fit)

write_csv(cv_metrics_regression, here('results/ALMANAC_klaeger_L1000_models/tabnet_metrics.csv'))
