library(tidyverse)
library(here)
library(tidymodels)
library(recipeselectors)
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
					 broad_id)
}


this_dataset = build_all_data_regression_viability_set(feature_cor =  all_data_feat_cors,
																											 num_features = 5000,
																											 all_data = all_data_filtered)

folds = vfold_cv(this_dataset, v = 10)

normal_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							
							new_role = "id variable")

boruta_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							
							new_role = "id variable") %>% 
	step_select_boruta(all_predictors(), outcome = "ic50")

infgain_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							
							new_role = "id variable") %>% 
	step_select_infgain(all_predictors(), outcome = "ic50", top_p = 10, threshold = 0.9)

mrmr_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							
							new_role = "id variable") %>% 
	step_select_mrmr(all_predictors(), outcome = "ic50", top_p = 10, threshold = 0.9)

carscore_recipe = recipe(ic50 ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("ic50"),
							
							new_role = "id variable") %>% 
	step_select_carscore(all_predictors(), outcome = "ic50", top_p = 10, threshold = 0.9)

xgb_spec <- boost_tree(
	trees = 500, 
	tree_depth = 13, min_n = 29, 
	loss_reduction = 0.02280748,                     ## first three: model complexity
	sample_size = 0.8467527, mtry = 148,         ## randomness
	learn_rate = 0.01328023,                         ## step size
) %>% 
	set_engine("xgboost") %>% 
	set_mode("regression")

this_workflow_set <-
	workflow_set(
		preproc = list(simple = normal_recipe, 
									 boruta = boruta_recipe, 
									 infgain = infgain_recipe, 
									 mrmr = mrmr_recipe, 
									 carscore = carscore_recipe),
		models = list(xgboost = xgb_spec), 
		cross = TRUE
	) 

ctrl <- control_resamples(save_pred = TRUE)

fit <-
	this_workflow_set %>%
	workflow_map("fit_resamples", 
							 resamples = folds, 
							 control = ctrl)
	fit_resamples(folds, control = ctrl)

cv_metrics_regression = collect_metrics(fit)

write_rds(fit, here('results/tuned_xgboost_feature_selection_comparison.rds'))
write_csv(cv_metrics_regression, here('results/tuned_xgboost_feature_selection_results.csv'))


