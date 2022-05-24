library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)
library(Boruta)

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

build_boruta_viability_set = function(all_data, boruta_decisions) {
	this_data_filtered = all_data %>%
		select(Model,
					 Treatment,
					 ResponseCategory,
					 BestAvgResponse,
					 binary_response,
					 below_median_response,
					 any_of(this_dataset_boruta_decision),
		) %>% 
		mutate(binary_response = as.factor(binary_response)) %>% 
		mutate(below_median_response = as.factor(below_median_response))
}
query_data = data
feature_list = seq(500, 5000, by = 500)
all_data_regression_metrics = data.frame()
for (i in 1:length(feature_list)) {
this_dataset_preprocessed = build_regression_viability_set(feature_cor =  cors,
																													 num_features = feature_list[i],
																													 all_data = query_data) %>% 
	select(starts_with("binary_response"),
				 starts_with("act_"),
				 starts_with("exp_"),
				 starts_with("cnv_"))

this_dataset_boruta = Boruta(binary_response~., data = this_dataset_preprocessed, maxRuns = 200, doTrace = 2)

this_dataset_boruta_decision = getSelectedAttributes(this_dataset_boruta, withTentative = T)

this_dataset = build_boruta_viability_set(all_data = query_data, boruta_decisions = this_dataset_boruta_decision)

folds = vfold_cv(this_dataset, v = 10, strata = binary_response)

boruta_recipe = recipe(binary_response ~ ., this_dataset) %>%
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

# keras_spec <- mlp(
# 	hidden_units = tune(), 
# 	penalty = tune(),
# 	epochs = tune()                  
# ) %>% 
# 	set_engine("keras") %>% 
# 	set_mode("classification")
# 
# keras_param = keras_spec %>% 
# 	parameters() %>% 
# 	update(hidden_units = hidden_units(c(1, 500)))



# boruta = boruta_recipe,
# infgain = infgain_recipe
# ,
# keras = keras_spec

complete_workflowset = workflow_set(
	preproc = list(boruta = boruta_recipe),
	models = list(rf = rf_spec,
								xgb = xgb_spec),
	cross = TRUE
)

complete_workflowset = complete_workflowset %>% 
	# option_add(param_info = rf_param, id = "simple_rf") %>% 
	# option_add(param_info = rf_param, id = "normal_rf") %>% 
	option_add(param_info = rf_param, id = "boruta_rf") %>% 
	# option_add(param_info = rf_param, id = "infgain_rf") %>% 
	# option_add(param_info = xgb_param, id = "simple_xgb") %>% 
	# option_add(param_info = xgb_param, id = "normal_xgb") %>% 
	option_add(param_info = xgb_param, id = "boruta_xgb") 
# option_add(param_info = xgb_param, id = "infgain_xgb") %>% 
# option_add(param_info = keras_param, id = "simple_keras") %>% 
# option_add(param_info = keras_param, id = "normal_keras") 
#option_add(param_info = keras_param, id = "boruta_keras") 
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

this_metrics = collect_metrics(all_results) %>% 
	mutate(feature_number = feature_list[i])

all_data_regression_metrics = bind_rows(all_data_regression_metrics, this_metrics)

}

write_csv(all_data_regression_metrics, here('results/all_PDX_boruta_models_classification_metrics_ANOVA.csv'))
