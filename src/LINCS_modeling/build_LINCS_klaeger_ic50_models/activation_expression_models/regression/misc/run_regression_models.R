library(tidyverse)
library(here)
library(tidymodels)

#load data 
binary_data_filtered = read_csv(here('results/all_binary_model_data.csv'))
binary_feat_cors =  read_csv(here('results/all_binary_data_feature_correlations.csv'))
all_data_filtered = read_csv(here('results/all_model_data.csv'))
all_data_feat_cors =  read_csv(here('results/all_data_feature_correlations.csv'))

#modelling functions
build_binary_regression_viability_set = function(num_features, all_data, feature_correlations) {
  this_data_filtered = all_data %>%
    select(starts_with("act_"),
           any_of(feature_correlations$feature[1:num_features]),
           depmap_id,
           ccle_name,
           ic50,
           broad_id)
}

get_binary_regression_cv_metrics = function(features, data) {
  this_dataset = build_binary_regression_viability_set(feature_cor =  binary_feat_cors,
                                                       num_features = features,
                                                       all_data = data)
  
  folds = vfold_cv(this_dataset, v = 10)
  
  this_recipe = recipe(ic50 ~ ., this_dataset) %>%
    update_role(-starts_with("act_"),
                -starts_with("exp_"),
                -starts_with("ic50"),
                new_role = "id variable")
  
  rand_forest_spec <- rand_forest(
    trees = 500
  ) %>% set_engine("ranger") %>%
    set_mode("regression")
  
  this_wflow <-
    workflow() %>%
    add_model(rand_forest_spec) %>%
    add_recipe(this_recipe)
  
  ctrl <- control_resamples(save_pred = TRUE)
  
  fit <-
    this_wflow %>%
    fit_resamples(folds, control = ctrl)
  
  cv_metrics_regression = collect_metrics(fit)
  
  return(cv_metrics_regression)
  
}

build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
  this_data_filtered = all_data %>%
    select(any_of(feature_correlations$feature[1:num_features]),
           depmap_id,
           ccle_name,
           ic50,
           broad_id,
           ic50)
}

get_all_data_regression_cv_metrics = function(features, data) {
  this_dataset = build_all_data_regression_viability_set(feature_cor =  all_data_feat_cors,
                                                         num_features = features,
                                                         all_data = data)
  
  folds = vfold_cv(this_dataset, v = 10)
  
  this_recipe = recipe(ic50 ~ ., this_dataset) %>%
    update_role(-starts_with("act_"),
                -starts_with("exp_"),
                -starts_with("ic50"),
                new_role = "id variable")
  
  rand_forest_spec <- rand_forest(
    trees = 500
  ) %>% set_engine("ranger") %>%
    set_mode("regression")
  
  this_wflow <-
    workflow() %>%
    add_model(rand_forest_spec) %>%
    add_recipe(this_recipe)
  
  ctrl <- control_resamples(save_pred = TRUE)
  
  fit <-
    this_wflow %>%
    fit_resamples(folds, control = ctrl)
  
  cv_metrics_regression = collect_metrics(fit)
  
  return(cv_metrics_regression)
  
}

#run models
feature_list = c(500, 1000 , 1500, 2000, 2500, 3000, 3500, 4000)


all_binary_regression_metrics = data.frame()
for (i in 1:length(feature_list)) {
  this_metrics = get_binary_regression_cv_metrics(features = feature_list[i], data = binary_data_filtered) %>%
    mutate(feature_number = feature_list[i])
  all_binary_regression_metrics = bind_rows(all_binary_regression_metrics, this_metrics)
}

write_csv(all_binary_regression_metrics, here('results/klaeger_LINCS_binary_hit_regression_results.csv'))

all_data_regression_metrics = data.frame()
for (i in 1:length(feature_list)) {
  this_metrics = get_all_data_regression_cv_metrics(features = feature_list[i], data = all_data_filtered) %>%
    mutate(feature_number = feature_list[i])
  all_data_regression_metrics = bind_rows(all_data_regression_metrics, this_metrics)
}

write_csv(all_data_regression_metrics, here('results/klaeger_LINCS_regression_results.csv'))