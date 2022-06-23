library(tidyverse)
library(here)
library(tabnet)
library(torch)
library(finetune)
library(tidymodels)
library(tictoc)
library(vroom)
torch_manual_seed(2222) 

full_dataset = read_rds(here('results/ALMANAC_klaeger_L1000_data_for_ml.rds.gz'))
feats =  vroom(here('results/ALMANAC_klaeger_L1000_models/feature_selection/lasso_selected_features.csv'))
id_vars = full_dataset %>% 
  select(-starts_with(c("act_", "pert_")), -viability)

this_dataset = full_dataset %>% 
  select(any_of(c(names(id_vars), feats$feature)), viability)

set.seed(2222)
folds = vfold_cv(this_dataset, v = 5)

this_recipe = recipe(viability ~ ., this_dataset) %>%
  update_role(any_of(names(id_vars)),
              new_role = "id variable") %>% 
  step_zv(all_predictors())

tabnet_spec <- tabnet(epochs = 20,
                      decision_width = tune(),
                      attention_width = tune(),
                      num_steps = tune(),
                      penalty = tune(),
                      batch_size = 2048,
                      virtual_batch_size = 512,
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
    decision_width = decision_width(range = c(16, 64)),
    attention_width = attention_width(range = c(16, 64)),
    num_steps = num_steps(range = c(3, 10))
  ) %>%
  grid_max_entropy(size = 10) %>% 
  mutate(attention_width = decision_width)

ctrl <- control_grid(
  parallel_over = "everything",
  save_pred = TRUE,
  verbose = TRUE)

fit <- this_wflow %>% 
  tune_grid(
    resamples = folds, 
    grid = grid,
    control = ctrl,
    metrics = metric_set(rsq)
  )

cv_metrics_regression = collect_metrics(fit)
preds = collect_predictions(fit)

preds %>% 
  ggplot(aes(.pred, viability)) +
  geom_point()

write_csv(cv_metrics_regression, here('results/ALMANAC_klaeger_L1000_models/tabnet_metrics.csv'))
