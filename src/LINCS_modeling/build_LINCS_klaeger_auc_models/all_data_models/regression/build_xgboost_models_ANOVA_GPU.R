library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)
library(argparse)
library(xgboost)

tic()
parser <- ArgumentParser(description='Process input paramters')
parser$add_argument('--feature_num', default = 100, type="integer")

args = parser$parse_args()
print(sprintf('Features: %02d',args$feature_num))

dir.create(here('results/PRISM_LINCS_klaeger_models_auc/all_datasets/regression/', 
								sprintf('xgboost/results')), 
					 showWarnings = F, recursive = T)
dir.create(here('results/PRISM_LINCS_klaeger_models_auc/all_datasets/regression/', 
								sprintf('xgboost/predictions')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models_auc/all_datasets/regression/xgboost/results', 
												sprintf('%dfeat.rds.gz',args$feature_num))

pred_output_file = here('results/PRISM_LINCS_klaeger_models_auc/all_datasets/regression/xgboost/predictions', 
												sprintf('%dfeat.rds.gz',args$feature_num))

data = vroom(here('results/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_5000feat_auc.csv'))
cors = vroom(here('results/PRISM_LINCS_klaeger_all_multiomic_data_feature_correlations_auc.csv'))

build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 auc,
					 broad_id)
}

this_dataset = build_all_data_regression_viability_set(feature_correlations =  cors,
																											 num_features = args$feature_num,
																											 all_data = data)

folds = vfold_cv(this_dataset, v = 10)

this_recipe = recipe(auc ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("cnv_"),
							-starts_with("prot_"),
							-starts_with("dep_"),
							-starts_with("auc"),
							new_role = "id variable") %>%
	step_normalize(all_predictors())

xgb_spec <- boost_tree(
	trees = tune(), 
	tree_depth = tune(),       
	learn_rate = tune()                   
) %>% 
	set_engine("xgboost", tree_method = "gpu_hist") %>% 
	set_mode("regression")

xgb_param = xgb_spec %>% 
	parameters() %>% 
	update(trees = trees(c(100, 1000)),
				 tree_depth = tree_depth(c(4, 30)))

xgb_grid = xgb_param %>% 
	grid_max_entropy(size = 20)

this_wflow <-
	workflow() %>%
	add_model(xgb_spec) %>%
	add_recipe(this_recipe) 

race_ctrl = control_race(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

results <- tune_race_anova(
	this_wflow,
	resamples = folds,
	grid = xgb_grid,
	metrics = metric_set(rsq),
	control = race_ctrl
) %>% 
	write_rds(full_output_file, compress = "gz")

write_rds(results$.predictions[[1]], pred_output_file)

toc()