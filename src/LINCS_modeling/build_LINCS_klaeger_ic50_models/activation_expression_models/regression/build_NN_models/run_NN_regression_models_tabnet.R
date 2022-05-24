library(tidyverse)
library(here)
library(tabnet)
library(finetune)
library(tidymodels)
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
					 broad_id,
					 ic50)
}

features = 3000
data = all_data_filtered

	this_dataset = build_all_data_regression_viability_set(feature_cor =  all_data_feat_cors,
																												 num_features = features,
																												 all_data = data)
	
	folds = vfold_cv(this_dataset, v = 10)
	
	this_recipe = recipe(ic50 ~ ., this_dataset) %>%
		update_role(-starts_with("act_"),
								-starts_with("exp_"),
								-starts_with("ic50"),
								new_role = "id variable") %>% 
		step_zv(all_predictors()) %>% 
		step_normalize(all_predictors())
	
	tabnet_spec <- tabnet(epochs = 100, batch_size = tune(), decision_width = tune(), attention_width = tune(),
												num_steps = tune(), penalty = 0.000001, virtual_batch_size = 512, momentum = 0.6,
												feature_reusage = tune(), learn_rate = tune()) %>%
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
			learn_rate = learn_rate(range = c(-2.5, -1)),
			batch_size = finalize(batch_size(), this_dataset)
		) %>%
		grid_max_entropy(size = 20)
	ctrl <- control_race(verbose_elim = TRUE)
	
	fit <- this_wflow %>% 
		tune_race_anova(
			resamples = folds, 
			grid = grid,
			control = ctrl,
			metrics = metric_set(rsq)
		)
	
	cv_metrics_regression = collect_metrics(fit)


write_csv(cv_metrics_regression, here('results/klaeger_LINCS_NN_tabnet_regression_results.csv'))
write_rds(fit, here('results/klaeger_LINCS_NN_tabnet_regression_race_result.rds'))
