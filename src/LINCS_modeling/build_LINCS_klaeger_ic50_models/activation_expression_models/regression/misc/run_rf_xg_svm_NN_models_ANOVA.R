library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)

all_cores <- parallel::detectCores(logical = FALSE)
cl <- makeCluster(all_cores)
registerDoParallel(cl)

data = vroom(here('results/PRISM_LINCS_klaeger_data_for_ml_5000feat.csv'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations.csv'))

build_all_data_classification_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		mutate(ic50_binary = as.factor(ic50_binary)) %>% 
		select(any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 ic50,
					 broad_id)
}

this_dataset = build_all_data_classification_viability_set(feature_correlations =  cors,
																							num_features = 5000,
																							all_data = data)

folds = vfold_cv(this_dataset, v = 10)

get_recipe = function(data, feature_number, feature_correlations) {
	normal_recipe = recipe(ic50 ~ ., this_dataset) %>%
		update_role(-starts_with("act_"),
								-starts_with("exp_"),
								-starts_with("ic50_binary"),
								new_role = "id variable") %>%
		step_select(depmap_id,
								ccle_name,
								ic50,
								broad_id,
								any_of(feature_correlations$feature[1:feature_number])) %>% 
		step_normalize(all_predictors()) 
	
	return(normal_recipe)
}

recipe_500 = get_recipe(data = this_dataset, feature_number = 500, feature_correlations = cors)
# recipe_1000 = get_recipe(data = this_dataset, feature_number = 1000, feature_correlations = cors)
# recipe_1500 = get_recipe(data = this_dataset, feature_number = 1500, feature_correlations = cors)
# recipe_2000 = get_recipe(data = this_dataset, feature_number = 2000, feature_correlations = cors)
# recipe_2500 = get_recipe(data = this_dataset, feature_number = 2500, feature_correlations = cors)
# recipe_3000 = get_recipe(data = this_dataset, feature_number = 3000, feature_correlations = cors)
# recipe_3500 = get_recipe(data = this_dataset, feature_number = 3500, feature_correlations = cors)
# recipe_4000 = get_recipe(data = this_dataset, feature_number = 4000, feature_correlations = cors)
# recipe_4500 = get_recipe(data = this_dataset, feature_number = 4500, feature_correlations = cors)
# recipe_5000 = get_recipe(data = this_dataset, feature_number = 5000, feature_correlations = cors)


xgb_spec <- boost_tree(
	trees = tune(), 
	tree_depth = tune(),       
	learn_rate = tune()                   
) %>% 
	set_engine("xgboost", tree_method = "gpu_hist", nthreads = 16) %>% 
	set_mode("regression")

xgb_param = xgb_spec %>% 
	parameters() %>% 
	update(trees = trees(c(100, 1000)),
				 tree_depth = tree_depth(c(4, 30)))

rf_spec <- rand_forest(
	trees = tune()
) %>% set_engine("ranger") %>%
	set_mode("regression")

rf_param = rf_spec %>% 
	parameters() %>% 
	update(trees = trees(c(100, 1000)))

lr_spec <- logistic_reg() %>%
	set_engine("glmnet")

keras_spec <- mlp(
	hidden_units = tune(), 
	penalty = tune(),
	epochs = tune()                  
) %>% 
	set_engine("keras") %>% 
	set_mode("regression")

keras_param = keras_spec %>% 
	parameters() %>% 
	update(hidden_units = hidden_units(c(1, 27)))

complete_workflowset = workflow_set(
	preproc = list(feat500 = recipe_500),
								 # feat1000 = recipe_1000,
								 # feat1500 = recipe_1500,
								 # feat2000 = recipe_2000,
								 # feat2500 = recipe_2500,
								 # feat3000 = recipe_3000,
								 # feat3500 = recipe_3500,
								 # feat4000 = recipe_4000,
								 # feat4500 = recipe_4500,
								 # feat5000 = recipe_5000),
	models = list(rf = rf_spec,
								xgb = xgb_spec,
								lr = lr_spec, 
								keras = keras_spec),
	cross = TRUE
)

complete_workflowset = complete_workflowset %>% 
	option_add(param_info = rf_param, id = "feat500_rf") %>%
	# option_add(param_info = rf_param, id = "feat1000_rf") %>% 
	# option_add(param_info = rf_param, id = "feat1500_rf") %>% 
	# option_add(param_info = rf_param, id = "feat2000_rf") %>% 
	# option_add(param_info = rf_param, id = "feat2500_rf") %>% 
	# option_add(param_info = rf_param, id = "feat3000_rf") %>% 
	# option_add(param_info = rf_param, id = "feat3500_rf") %>% 
	# option_add(param_info = rf_param, id = "feat4000_rf") %>% 
	# option_add(param_info = rf_param, id = "feat4500_rf") %>% 
	# option_add(param_info = rf_param, id = "feat5000_rf") %>% 
	option_add(param_info = xgb_param, id = "feat500_xgb") %>%
	# option_add(param_info = xgb_param, id = "feat1000_xgb") %>% 
	# option_add(param_info = xgb_param, id = "feat1500_xgb") %>% 
	# option_add(param_info = xgb_param, id = "feat2000_xgb") %>% 
	# option_add(param_info = xgb_param, id = "feat2500_xgb") %>% 
	# option_add(param_info = xgb_param, id = "feat3000_xgb") %>% 
	# option_add(param_info = xgb_param, id = "feat3500_xgb") %>% 
	# option_add(param_info = xgb_param, id = "feat4000_xgb") %>% 
	# option_add(param_info = xgb_param, id = "feat4500_xgb") %>% 
	# option_add(param_info = xgb_param, id = "feat5000_xgb") %>% 
	option_add(param_info = keras_param, id = "feat500_keras")
	# option_add(param_info = keras_param, id = "feat1000_keras") %>% 
	# option_add(param_info = keras_param, id = "feat1500_keras") %>% 
	# option_add(param_info = keras_param, id = "feat2000_keras") %>% 
	# option_add(param_info = keras_param, id = "feat2500_keras") %>% 
	# option_add(param_info = keras_param, id = "feat3000_keras") %>% 
	# option_add(param_info = keras_param, id = "feat3500_keras") %>% 
	# option_add(param_info = keras_param, id = "feat4000_keras") %>% 
	# option_add(param_info = keras_param, id = "feat4500_keras") %>% 
	# option_add(param_info = keras_param, id = "feat5000_keras")

race_ctrl = control_race(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

all_results = complete_workflowset %>% 
	workflow_map(
		"tune_race_anova",
		seed = 2222,
		resamples = folds,
		grid = 2,
		control = race_ctrl
	)

write_rds(all_results, here('results/LINCS_klaeger_PRISM_xgb_rf_lr_NN_models_classification_results_ANOVA.rds'))

cv_metrics_regression = collect_metrics(all_results)

write_csv(cv_metrics_regression, here('results/LINCS_klaeger_PRISM_xgb_rf_lr_NN_models_classification_metrics_ANOVA.csv'))
