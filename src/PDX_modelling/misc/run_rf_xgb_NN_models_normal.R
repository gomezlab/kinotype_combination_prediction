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

data = vroom(here('results/PDX_klaeger_LINCS_data_for_ml.csv'))
cors =  vroom(here('results/PXD_LINCS_klaeger_data_feat_cors.csv'))

build_regression_viability_set = function(num_features, all_data, feature_correlations) {
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


this_dataset = build_regression_viability_set(feature_cor =  cors,
																							num_features = 3000,
																							all_data = data)

folds = vfold_cv(this_dataset, v = 10, strata = binary_response)

normal_recipe = recipe(binary_response ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
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

xgb_param = xgb_spec %>% 
	parameters() %>% 
	update(trees = trees(c(100, 1000)),
				 tree_depth = tree_depth(c(4, 30)))

rf_spec <- rand_forest(
	trees = tune()
) %>% set_engine("ranger") %>%
	set_mode("classification")

rf_param = rf_spec %>% 
	parameters() %>% 
	update(trees = trees(c(100, 1000)))

keras_spec <- mlp(
	hidden_units = tune(), 
	penalty = tune(),
	epochs = tune()                  
) %>% 
	set_engine("keras") %>% 
	set_mode("classification")

keras_param = keras_spec %>% 
	parameters() %>% 
	update(hidden_units = hidden_units(c(1, 500)))

# , 
# 
# boruta = boruta_recipe, 
# infgain = infgain_recipe

complete_workflowset = workflow_set(
	preproc = list(normal = normal_recipe),
	models = list(rf = rf_spec,
								xgb = xgb_spec,
								keras = keras_spec),
	cross = TRUE
)

complete_workflowset = complete_workflowset %>% 
	#option_add(param_info = rf_param, id = "simple_rf") %>% 
	option_add(param_info = rf_param, id = "normal_rf") %>% 
	# option_add(param_info = rf_param, id = "boruta_rf") %>% 
	# option_add(param_info = rf_param, id = "infgain_rf") %>% 
	#option_add(param_info = xgb_param, id = "simple_xgb") %>% 
	option_add(param_info = xgb_param, id = "normal_xgb") %>% 
	# option_add(param_info = xgb_param, id = "boruta_xgb") %>% 
	# option_add(param_info = xgb_param, id = "infgain_xgb") %>% 
	#option_add(param_info = keras_param, id = "simple_keras") %>% 
	option_add(param_info = keras_param, id = "normal_keras") 
# option_add(param_info = keras_param, id = "boruta_keras") %>% 
# option_add(param_info = keras_param, id = "infgain_keras")

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


vroom_write(cv_metrics_regression, here('results/PDX_normal_models_classification_metrics_ANOVA.csv'))
write_rds(here('results/PDX_normal_models_classification_results_ANOVA.rds'))