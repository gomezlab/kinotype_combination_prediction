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

args = data.frame(feature_num = c(2000,3000,4000,5000))
data = vroom(here('results/PRISM_LINCS_klaeger_data_for_ml_5000feat_auc.csv'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))

build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 auc,
					 broad_id)
}

for(i in 1:length(args$feature_num)) {
	tic()	
print(sprintf('Features: %02d',args$feature_num[i]))

dir.create(here('results/PRISM_LINCS_klaeger_models/activation_expression/regression/', 
								sprintf('rand_forest/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models/activation_expression/regression/rand_forest/results', 
												sprintf('%dfeat.rds',args$feature_num[i]))

this_dataset = build_all_data_regression_viability_set(feature_correlations =  cors,
																													 num_features = args$feature_num[i],
																													 all_data = data)

folds = vfold_cv(this_dataset, v = 10)

this_recipe = recipe(auc ~ ., this_dataset) %>%
	update_role(-starts_with("act_"),
							-starts_with("exp_"),
							-starts_with("auc"),
							new_role = "id variable") %>%
	step_normalize(all_predictors())

rf_spec <- rand_forest(
	trees = tune()
) %>% set_engine("ranger", num.threads = 16) %>%
	set_mode("regression")

rf_param = rf_spec %>% 
	parameters() %>% 
	update(trees = trees(c(100, 2000)))

this_wflow <-
	workflow() %>%
	add_model(rf_spec) %>%
	add_recipe(this_recipe) 

rf_grid = rf_param %>% 
	grid_max_entropy(size = 30)

race_ctrl = control_race(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

results <- tune_race_anova(
	this_wflow,
	resamples = folds,
	grid = rf_grid,
	metrics = metric_set(rsq),
	control = race_ctrl
) %>% 
	write_rds(full_output_file, compress = "gz")

toc()
}