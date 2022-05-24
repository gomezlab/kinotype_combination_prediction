library(tidyverse)
library(here)
library(vroom)

data = vroom(here('results/PDX_klaeger_LINCS_data_for_ml.csv'))

find_all_data_feature_correlations <- function(row_indexes = NA, all_data) {
	if (is.na(row_indexes)) {
		row_indexes = 1:dim(all_data)[1]
	}
	
	all_cor = cor(
		all_data %>% 
			pull(BestAvgResponse),
		
		all_data %>% 
			select(starts_with(c('act','exp', 'cnv')))
	) %>%
		as.data.frame() %>%
		pivot_longer(everything(), names_to = "feature",values_to = "cor")
	
	
	all_correlations = all_cor %>% 
		mutate(abs_cor = abs(cor)) %>% 
		arrange(desc(abs_cor)) %>% 
		mutate(rank = 1:n()) %>%
		mutate(feature_type = case_when(
			str_detect(feature, "^act_") ~ "Activation",
			str_detect(feature, "^exp_") ~ "Expression",
			str_detect(feature, "^cnv_") ~ "CopyNumber",
			T ~ feature
		))
	
	return(all_correlations)
}

cors = find_all_data_feature_correlations(all_data = data)
vroom_write(cors, here('results/PXD_LINCS_klaeger_data_feat_cors.csv'))
