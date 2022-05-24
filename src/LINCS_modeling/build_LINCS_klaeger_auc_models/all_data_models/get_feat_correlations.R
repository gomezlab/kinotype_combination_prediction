library(tidyverse)
library(here)
library(vroom)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 10)
data = vroom(here('results/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_auc.csv'))
act_exp_cors = vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations_auc.csv'))


find_all_data_feature_correlations <- function(row_indexes = NA, all_data, query) {
	if (is.na(row_indexes)) {
		row_indexes = 1:dim(all_data)[1]
	}
	
	all_cor = cor(
		all_data %>% 
			
			
			pull(ic50),
		
		all_data %>% 
			select(starts_with(query))
	) %>%
		as.data.frame() %>%
		pivot_longer(everything(), names_to = "feature",values_to = "cor") %>% 
		mutate(cor = as.numeric(cor))
	
	
	all_correlations = all_cor %>% 
		mutate(abs_cor = abs(cor)) %>% 
		arrange(desc(abs_cor)) %>% 
		mutate(feature_type = case_when(
			str_detect(feature, "^act_") ~ "Activation",
			str_detect(feature, "^exp_") ~ "Expression",
			str_detect(feature, "^cnv_") ~ "CopyNumber",
			str_detect(feature, "^dep_") ~ "DepMap",
			str_detect(feature, "^prot_") ~ "Proteomics",
			T ~ feature
		))
	
	return(all_correlations)
}

cnv_cors = find_all_data_feature_correlations(all_data = data, query = "cnv_")
prot_cors = find_all_data_feature_correlations(all_data = data, query = "prot_")
dep_cors = find_all_data_feature_correlations(all_data = data, query = "dep_")

all_cors = bind_rows(act_exp_cors, cnv_cors, prot_cors, dep_cors) %>% 
	select(-rank) %>% 
	arrange(desc(abs_cor)) %>% 
	mutate(rank = 1:n())

write_csv(all_cors, here('results/PRISM_LINCS_klaeger_all_multiomic_data_feature_correlations_auc.csv'))
