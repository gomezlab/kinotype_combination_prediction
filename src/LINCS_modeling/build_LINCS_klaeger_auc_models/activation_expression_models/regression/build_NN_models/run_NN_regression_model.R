library(tidyverse)
library(here)
library(tidymodels)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)

doParallel::registerDoParallel()

all_data_filtered = read_csv(here('results/all_model_data_filtered.csv'))
all_data_feat_cors =  read_csv(here('results/all_filtered_data_feature_correlations.csv'))

build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 ic50,
					 broad_id,
					 ic50)
}

data = all_data_filtered
features = 1500

this_dataset = build_all_data_regression_viability_set(feature_cor =  all_data_feat_cors,
																											 num_features = features,
																											 all_data = data)

folds = vfold_cv(this_dataset, v = 10)

this_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							new_role = "id variable") %>% 
	step_zv(all_predictors()) %>% 
	step_normalize(all_predictors())

keras_spec <- mlp(
	hidden_units = tune(), 
	penalty = tune(), 
	activation = "linear"                  
) %>% 
	set_engine("keras") %>% 
	set_mode("regression")

keras_grid <- keras_spec %>%
	parameters %>% 
	grid_max_entropy(size = 10)

this_wflow <-
	workflow() %>%
	add_model(keras_spec) %>%
	add_recipe(this_recipe)

ctrl <- control_resamples(save_pred = TRUE)

fit <- tune_grid(
	this_wflow,
	resamples = folds,
	grid = keras_grid,
	control = ctrl
)

cv_metrics_regression = collect_metrics(fit)

write_csv(cv_metrics_regression, here('results/klaeger_LINCS_NN_regression_results.csv'))
write_rds(fit, here('results/klaeger_LINCS_NN_regression__model.rds'))