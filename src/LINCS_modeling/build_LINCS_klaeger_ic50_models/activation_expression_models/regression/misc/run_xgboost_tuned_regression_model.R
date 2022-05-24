library(tidyverse)
library(here)
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

features = 1500
data = all_data_filtered

	this_dataset = build_all_data_regression_viability_set(feature_cor =  all_data_feat_cors,
																												 num_features = features,
																												 all_data = data)
	
	folds = vfold_cv(this_dataset, v = 10)
	
	this_recipe = recipe(ic50 ~ ., this_dataset) %>%
		update_role(-starts_with("act_"),
								-starts_with("exp_"),
								-starts_with("ic50"),
								new_role = "id variable")
	
	xgb_spec <- boost_tree(
		trees = 500, 
		tree_depth = 13, min_n = 29, 
		loss_reduction = 0.02280748,                     ## first three: model complexity
		sample_size = 0.8467527, mtry = 148,         ## randomness
		learn_rate = 0.01328023,                         ## step size
	) %>% 
		set_engine("xgboost") %>% 
		set_mode("regression")
	
	this_wflow <-
		workflow() %>%
		add_model(xgb_spec) %>%
		add_recipe(this_recipe)
	
	ctrl <- control_resamples(save_pred = TRUE)
	
	fit <-
		this_wflow %>%
		fit_resamples(folds, control = ctrl)
	
cv_metrics_regression = collect_metrics(fit)
 
predictions_regression_500 = collect_predictions(fit) %>% 
	rename('predicted_ic50' = .pred)

predictions_regression_500 %>% 
	ggplot(aes(x = ic50, y = predicted_ic50)) +
	geom_hex() +
	scale_fill_gradient(low="lightblue1",high="darkblue") +
	geom_smooth() +
	labs(title = paste0('Correlation = ', 
											round(
												cor(predictions_regression_500$ic50, 
														predictions_regression_500$predicted_ic50),
												4),
											', R-Squared = ', round(
												cv_metrics_regression$mean[2],
												4),
											', RMSE = ', round(cv_metrics_regression$mean[1],
																				 4)),
			 x = "log10_ic50",
			 y = "predicted log10ic50") +
	geom_abline(intercept = 0, slope = 1, size = 0.5, colour = 'red') 
# xlim(c(-1.5,1.5)) +
# ylim(c(-1.5,1.5))

ggsave(here('figures/1500_feat_tuned_regression_model_results.png'))

write_rds(fit, here('results/1500_feat_tuned_regression_model_results.rds'))