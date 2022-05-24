
library(tidyverse)
library(doParallel)

all_model_data_filtered_long = all_model_data_filtered %>% 
	pivot_longer(starts_with(c('act', 'exp')), names_to = 'feature', values_to = 'value')

variable_genes = all_model_data_filtered_long %>% 
	select(feature, value) %>% 
	group_by(feature) %>% 
	summarise(var = var(value)) %>% 
	filter(var > 0.01)