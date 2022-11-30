library(torch)
library(tidyverse)
library(here)
library(tabnet)
library(tidymodels)
library(finetune)

set.seed(2222)
#build RF grid
rf_spec <- rand_forest(
	trees = tune()
) %>% set_engine("ranger", num.threads = 16) %>%
	set_mode("regression")

rf_param = rf_spec %>% 
	parameters() %>% 
	update(trees = trees(c(100, 2000)))

rf_grid = rf_param %>% 
	grid_max_entropy(size = 15)

#build XGB grid 
xgb_spec <- boost_tree(
	trees = tune(), 
	tree_depth = tune()                 
) %>% 
	set_engine("xgboost", tree_method = "gpu_hist") %>% 
	set_mode("regression")

xgb_param = xgb_spec %>% 
	parameters() %>% 
	update(trees = trees(c(20, 600)),
				 tree_depth = tree_depth(c(3, 20)))

xgb_grid = xgb_param %>% 
	grid_max_entropy(size = 30)

#build NN grid
keras_spec <- mlp(
	hidden_units = tune(), 
	penalty = tune()                  
) %>% 
	set_engine("keras", verbose = 0) %>% 
	set_mode("regression")

keras_param = keras_spec %>% 
	parameters() %>% 
	update(hidden_units = hidden_units(c(1, 2000)),
				 penalty = penalty(range = c(-10, -0.0969)))

keras_grid = keras_param %>% 
	grid_max_entropy(size = 30)

#build tabnet grid 
tabnet_spec <- tabnet(epochs = 10, decision_width = tune(), attention_width = tune(),
											num_steps = tune(), penalty = 0.000001, virtual_batch_size = 512, momentum = 0.6,
											feature_reusage = 1.5, learn_rate = tune()) %>%
	set_engine("torch", verbose = TRUE) %>%
	set_mode("regression")

set.seed(2222)
tabnet_grid <-
	tabnet_spec %>%
	parameters() %>%
	update(
		decision_width = decision_width(range = c(8, 64)),
		attention_width = attention_width(range = c(8, 64)),
		num_steps = num_steps(range = c(3, 10)),
		learn_rate = learn_rate(range = c(-2.5, -1))
	) %>%
	grid_max_entropy(size = 30)

#write out grids 
write_rds(rf_grid, here('results/hyperparameter_grids/rf_grid.rds'))
write_rds(xgb_grid, here('results/hyperparameter_grids/xgb_grid.rds'))
write_rds(keras_grid, here('results/hyperparameter_grids/keras_grid.rds'))
write_rds(tabnet_grid, here('results/hyperparameter_grids/tabnet_grid.rds'))
