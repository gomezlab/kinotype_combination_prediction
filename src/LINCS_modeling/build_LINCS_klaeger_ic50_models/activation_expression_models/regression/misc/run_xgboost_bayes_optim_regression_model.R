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
								new_role = "id variable")
	
	xgb_spec <- boost_tree(
		trees = tune(), 
		tree_depth = tune(), min_n = tune(), 
		loss_reduction = tune(),                     ## first three: model complexity
		sample_size = tune(), mtry = tune(),         ## randomness
		learn_rate = tune(),                         ## step size
	) %>% 
		set_engine("xgboost") %>% 
		set_mode("regression")
	
	this_wflow <-
		workflow() %>%
		add_model(xgb_spec) %>%
		add_recipe(this_recipe)
	
	xgb_set <- parameters(this_wflow) %>%
		update(trees = trees(c(100, 1000)),
					 tree_depth = tree_depth(c(4, 30))) %>% 
		update(mtry = finalize(mtry(), this_dataset))
		
	
	fit <- tune_bayes(
		this_wflow,
		resamples = folds,
		param_info = xgb_set,
		initial = 8,
		iter = 30,
		metrics = metric_set(rsq),
		control = control_bayes(no_improve = 10, save_pred = TRUE, verbose = TRUE)
	)
	
	cv_metrics_regression = collect_metrics(fit)

write_csv(cv_metrics_regression, here('results/klaeger_LINCS_xgboost_regression_bayes_optimization_results.csv'))
write_rds(fit, here('results/klaeger_LINCS_xgboost_regression_bayes_optimization_model.rds'))