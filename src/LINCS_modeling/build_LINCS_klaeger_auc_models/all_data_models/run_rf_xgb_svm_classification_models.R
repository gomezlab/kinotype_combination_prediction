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
library(argparse)

doParallel::registerDoParallel()

data = vroom(here('results/PRISM_klaeger_LINCS_data_all_datasets.csv'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_all_datasets_feat_cors.csv'))

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

this_dataset = build_all_data_classification_viability_set(feature_correlations =  cors,
																													 num_features = 5000,
																													 all_data = data)

folds = vfold_cv(this_dataset, v = 10)

this_recipe = recipe(ic50_binary ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50_binary"),
							new_role = "id variable") %>%
	step_select(ic50_binary,
							depmap_id,
							ccle_name,
							ic50,
							broad_id,
							any_of(feature_correlations$feature[1:feature_number])) %>% 
	step_normalize(all_predictors()) 

xgb_spec <- boost_tree(
	trees = tune(), 
	tree_depth = tune(),       
	learn_rate = tune()                   
) %>% 
	set_engine("xgboost") %>% 
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

keras_spec <- mlp(
	hidden_units = tune(), 
	penalty = tune(),
	epochs = tune()                  
) %>% 
	set_engine("keras") %>% 
	set_mode("regression")

keras_param = keras_spec %>% 
	parameters() %>% 
	update(hidden_units = hidden_units(c(1, 500)))

complete_workflowset = workflow_set(
	preproc = list(simple = simple_recipe, 
								 normal = normal_recipe, 
								 boruta = boruta_recipe, 
								 infgain = infgain_recipe),
	models = list(rf = rf_spec,
								xgb = xgb_spec,
								keras = keras_spec),
	cross = TRUE
)

complete_workflowset = complete_workflowset %>% 
	option_add(param_info = rf_param, id = "simple_rf") %>% 
	option_add(param_info = rf_param, id = "normal_rf") %>% 
	option_add(param_info = rf_param, id = "boruta_rf") %>% 
	option_add(param_info = rf_param, id = "infgain_rf") %>% 
	option_add(param_info = xgb_param, id = "simple_xgb") %>% 
	option_add(param_info = xgb_param, id = "normal_xgb") %>% 
	option_add(param_info = xgb_param, id = "boruta_xgb") %>% 
	option_add(param_info = xgb_param, id = "infgain_xgb") %>% 
	option_add(param_info = keras_param, id = "simple_keras") %>% 
	option_add(param_info = keras_param, id = "normal_keras") %>% 
	option_add(param_info = keras_param, id = "boruta_keras") %>% 
	option_add(param_info = keras_param, id = "infgain_keras")

race_ctrl = control_race(
	save_pred = TRUE, 
	parallel_over = "everything",
	save_workflow = TRUE, 
	verbose = TRUE
)

all_results = complete_workflowset %>% 
	workflow_map(
		"tune_race_anova",
		seed = 2222,
		resamples = folds,
		grid = 25,
		control = race_ctrl
	)

cv_metrics_regression = collect_metrics(all_results)


vroom_write(cv_metrics_regression, here('results/PRISM_LINCS_klaeger_all_models_all_datasets_regression_metrics_ANOVA.csv'))
write_rds(here('results/PRISM_LINCS_klaeger_all_models_all_datasets_regression_results_ANOVA.rds'))