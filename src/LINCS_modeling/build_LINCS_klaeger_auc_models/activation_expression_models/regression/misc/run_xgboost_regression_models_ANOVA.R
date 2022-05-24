library(tidyverse)
library(here)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)
library(vroom)

doParallel::registerDoParallel()

all_data_filtered = vroom(here('results/all_model_data_filtered.csv'))
all_data_feat_cors =  vroom(here('results/all_filtered_data_feature_correlations.csv'))

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
	
	xgb_spec <- boost_tree(
		trees = tune(), 
		tree_depth = tune(),         ## randomness
		learn_rate = tune(),                         ## step size
	) %>% 
		set_engine("xgboost") %>% 
		set_mode("regression")
	
	this_wflow <-
		workflow() %>%
		add_model(xgb_spec) %>%
		add_recipe(this_recipe)
	
	xgb_grid = parameters(this_wflow) %>% 
		update(trees = trees(c(100, 1000)),
					 tree_depth = tree_depth(c(4, 30))) %>% 
		grid_latin_hypercube(size = 30)
	
	race_ctrl = control_race(
		save_pred = TRUE, 
		parallel_over = "everything",
		save_workflow = TRUE, 
		verbose = TRUE
	)
	
	fit <- tune_race_anova(
		this_wflow,
		resamples = folds,
		grid = xgb_grid,
		metrics = metric_set(rsq, rmse),
		control = race_ctrl
	)
	
	cv_metrics_regression = collect_metrics(fit)
	
	return(cv_metrics_regression)
	
}

feature_list = seq(500, 5000, by = 500)

all_data_regression_metrics = data.frame()
for (i in 1:length(feature_list)) {
	this_metrics = get_all_data_regression_cv_metrics(features = feature_list[i], data = all_data_filtered) %>%
		mutate(feature_number = feature_list[i])
	all_data_regression_metrics = bind_rows(all_data_regression_metrics, this_metrics)
}

vroom_write(all_data_regression_metrics, here('results/klaeger_LINCS_xgboost_regression_results_ANOVA.csv'))