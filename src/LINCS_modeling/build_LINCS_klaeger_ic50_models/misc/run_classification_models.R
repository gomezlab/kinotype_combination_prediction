library(tidyverse)
library(here)
library(tidymodels)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)

build_all_data_classification_viability_set = function(num_features, all_data, feature_correlations) {
  this_data_filtered = all_data %>%
    select(any_of(feature_correlations$feature[1:num_features]),
           depmap_id,
           ccle_name,
           ic50,
           broad_id,
           ic50_binary)
}

get_all_data_classification_cv_metrics = function(features, data) {
  this_dataset = build_all_data_classification_viability_set(feature_cor =  all_data_feat_cors,
                                                             num_features = features,
                                                             all_data = data)
  
  folds = vfold_cv(this_dataset, v = 10)
  
  this_recipe = recipe(ic50_binary ~ ., this_dataset) %>%
    update_role(-starts_with("act_"),
                -starts_with("exp_"),
                -starts_with("ic50_binary"),
                new_role = "id variable")
  
  rand_forest_spec <- rand_forest(
    trees = 500
  ) %>% set_engine("ranger") %>%
    set_mode("classification")
  
  this_wflow <-
    workflow() %>%
    add_model(rand_forest_spec) %>%
    add_recipe(this_recipe)
  
  ctrl <- control_resamples(save_pred = TRUE)
  
  fit <-
    this_wflow %>%
    fit_resamples(folds, control = ctrl)
  
  cv_metrics_classification = collect_metrics(fit)
  
  return(cv_metrics_classification)
  
}

feature_list = c(500, 1000 , 1500, 2000, 2500, 3000, 3500, 4000)

all_data_classification_metrics = data.frame()
for (i in 1:length(feature_list)) {
  this_metrics = get_all_data_classification_cv_metrics(features = feature_list[i], data = all_data_filtered) %>%
    mutate(feature_number = feature_list[i])
  all_data_classification_metrics = bind_rows(all_data_classification_metrics, this_metrics)
}

write_csv(all_data_classification_metrics, here('results/klaeger_LINCS_classification_results.csv'))