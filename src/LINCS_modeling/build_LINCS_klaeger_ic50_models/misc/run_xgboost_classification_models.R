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

build_all_data_classification_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		mutate(ic50_binary = as.factor(ic50_binary)) %>% 
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
	
	xgb_spec <- boost_tree(
		trees = 500, 
		tree_depth = tune(), min_n = tune(), 
		loss_reduction = tune(),                     ## first three: model complexity
		sample_size = tune(), mtry = tune(),         ## randomness
		learn_rate = tune(),                         ## step size
	) %>% 
		set_engine("xgboost") %>% 
		set_mode("classification")
	
	xgb_grid <- grid_latin_hypercube(
		tree_depth(),
		min_n(),
		loss_reduction(),
		sample_size = sample_prop(),
		finalize(mtry(), this_dataset),
		learn_rate(),
		size = 15
	)
	
	this_wflow <-
		workflow() %>%
		add_model(xgb_spec) %>%
		add_recipe(this_recipe)
	
	ctrl <- control_resamples(save_pred = TRUE)
	
	fit <- tune_grid(
		this_wflow,
		resamples = folds,
		grid = xgb_grid,
		control = control_grid(save_pred = TRUE)
	)
	
	cv_metrics_classification = collect_metrics(fit)
	
	return(cv_metrics_classification)
	
}

feature_list = c(250, 500, 1000 , 1500, 2000, 2500, 3000, 3500)

all_data_classification_metrics = data.frame()
for (i in 1:length(feature_list)) {
	this_metrics = get_all_data_classification_cv_metrics(features = feature_list[i], data = all_data_filtered) %>%
		mutate(feature_number = feature_list[i])
	all_data_classification_metrics = bind_rows(all_data_classification_metrics, this_metrics)
}

write_csv(all_data_classification_metrics, here('results/klaeger_LINCS_xgboost_classification_results.csv'))