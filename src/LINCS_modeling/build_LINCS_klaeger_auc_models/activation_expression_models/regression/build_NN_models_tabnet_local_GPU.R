library(torch)
library(tidyverse)
library(here)
library(tabnet)
library(vroom)
library(finetune)
library(tidymodels)
library(tictoc)
library(doParallel)
library(vip)

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
args = data.frame(feature_num = c(500,1500,5000,10000))

for(i in 1:length(args$feature_num)) {
	tic()	
	print(sprintf('Features: %02d',args$feature_num[i]))
	
	dir.create(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/', 
									sprintf('tabnet/results')), 
						 showWarnings = F, recursive = T)
	dir.create(here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/', 
									sprintf('tabnet/predictions')), 
						 showWarnings = F, recursive = T)
	
	full_output_file = here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/tabnet/results', 
													sprintf('%dfeat.rds.gz',args$feature_num)[i])
	
	pred_output_file = here('results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/tabnet/predictions', 
													sprintf('%dfeat.rds.gz',args$feature_num)[i])
	
	this_dataset = build_all_data_regression_viability_set(feature_correlations =  cors,
																												 num_features = args$feature_num[i],
																												 all_data = data)
	
	folds = vfold_cv(this_dataset, v = 10)
	
	this_recipe = recipe(auc ~ ., this_dataset) %>%
		update_role(-starts_with("act_"),
								-starts_with("exp_"),
								-starts_with("auc"),
								new_role = "id variable") %>%
		step_BoxCox(all_predictors()) %>% 
		step_normalize(all_predictors())

tabnet_spec <- tabnet(epochs = 10, decision_width = tune(), attention_width = tune(),
											num_steps = tune(), penalty = 0.000001, virtual_batch_size = 512, momentum = 0.6,
											feature_reusage = 1.5, learn_rate = tune()) %>%
	set_engine("torch", verbose = TRUE) %>%
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(tabnet_spec) %>%
	add_recipe(this_recipe)

grid <-
	this_wflow %>%
	parameters() %>%
	update(
		decision_width = decision_width(range = c(8, 64)),
		attention_width = attention_width(range = c(8, 64)),
		num_steps = num_steps(range = c(3, 10)),
		learn_rate = learn_rate(range = c(-2.5, -1))
	) %>%
	grid_max_entropy(size = 16)

race_ctrl = control_race(
	save_pred = TRUE,
	parallel_over = "everything",
	verbose = TRUE
)

fit <- this_wflow %>% 
	tune_race_anova(
		resamples = folds, 
		grid = grid,
		control = race_ctrl,
		metrics = metric_set(rsq)
	) %>% 
	write_rds(full_output_file, compress = "gz")
toc()
}