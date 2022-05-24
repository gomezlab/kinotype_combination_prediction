library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)
library(recipeselectors)

doParallel::registerDoParallel()

dataset = vroom(here('results/PDX_klaeger_LINCS_data_for_ml.csv'))
cors =  vroom(here('results/PXD_LINCS_klaeger_data_feat_cors.csv'))


build_classification_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(Model,
					 Treatment,
					 ResponseCategory,
					 BestAvgResponse,
					 binary_response,
					 below_median_response,
					 any_of(feature_correlations$feature[1:num_features]),
		) %>% 
		mutate(binary_response = as.factor(binary_response)) %>% 
		mutate(below_median_response = as.factor(below_median_response))
}

get_all_data_classification_cv_metrics = function(features, data) {
	this_dataset = build_classification_viability_set(feature_cor =  cors,
																												 num_features = features,
																												 all_data = dataset)
	
	folds = vfold_cv(this_dataset, v = 10, strata = binary_response)
	
	normal_recipe = recipe(binary_response ~ ., this_dataset) %>%
		update_role(-starts_with("act_"),
								-starts_with("exp_"),
								-starts_with("cnv_"),
								-starts_with("binary_response"),
								new_role = "id variable") %>% 
		step_normalize(all_predictors())
	
	xgb_spec <- boost_tree(
		trees = tune(), 
		tree_depth = tune(),       
		learn_rate = tune()                   
	) %>% 
		set_engine("xgboost") %>% 
		set_mode("classification")
	
	this_wflow <-
		workflow() %>%
		add_model(xgb_spec) %>%
		add_recipe(normal_recipe)
	
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
		metrics = metric_set(roc_auc),
		control = race_ctrl
	)
	
	cv_metrics_classification = collect_metrics(fit)
	
	return(cv_metrics_classification)
	
}

feature_list = seq(500, 5000, by = 500)

all_data_classification_metrics = data.frame()
for (i in 1:length(feature_list)) {
	this_metrics = get_all_data_classification_cv_metrics(features = feature_list[i], data = dataset) %>%
		mutate(feature_number = feature_list[i])
	all_data_classification_metrics = bind_rows(all_data_classification_metrics, this_metrics)
}

write_csv(all_data_classification_metrics, here('results/PDX_klaeger_LINCS_xgboost_classification_results_ANOVA.csv'))