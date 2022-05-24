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

args = data.frame(feature_num = c(100,200,300,400,500,1000,1500,2000,3000,4000,5000))
data = vroom(here('results/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_5000feat_auc.csv'))
cors = vroom(here('results/PRISM_LINCS_klaeger_all_multiomic_data_feature_correlations_auc.csv'))

build_all_data_regression_viability_set = function(num_features, all_data, feature_correlations) {
	this_data_filtered = all_data %>%
		select(auc,
					 any_of(feature_correlations$feature[1:num_features]),
					 depmap_id,
					 ccle_name,
					 broad_id)
}

for(i in 1:length(args$feature_num)) {
	tic()	
	print(sprintf('Features: %02d',args$feature_num[i]))

dir.create(here('results/PRISM_LINCS_klaeger_models_auc/all_datasets/regression/', 
								sprintf('lr/results')), 
					 showWarnings = F, recursive = T)

full_output_file = here('results/PRISM_LINCS_klaeger_models_auc/all_datasets/regression/lr/results', 
												sprintf('%dfeat.rds.gz',args$feature_num[i]))

this_dataset = build_all_data_regression_viability_set(feature_correlations =  cors,
																											 num_features = args$feature_num[i],
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

lr_spec <- linear_reg() %>%
	set_mode("regression")

this_wflow <-
	workflow() %>%
	add_model(lr_spec) %>%
	add_recipe(this_recipe) 

race_ctrl = control_resamples(
	save_pred = TRUE, 
	parallel_over = "everything",
	verbose = TRUE
)

results <- fit_resamples(
	this_wflow,
	resamples = folds,
	control = race_ctrl
) %>% 
	write_rds(full_output_file, compress = "gz")

toc()
}