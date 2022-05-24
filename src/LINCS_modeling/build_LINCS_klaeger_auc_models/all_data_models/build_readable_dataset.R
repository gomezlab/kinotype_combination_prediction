library(tidyverse)
library(here)
library(vroom)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 10)
data = vroom(here('results/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_auc.csv'))
cors = vroom(here('results/PRISM_LINCS_klaeger_all_multiomic_data_feature_correlations_auc.csv'))

feat5000_data = data %>% 
	select(any_of(cors$feature[1:5005]),
				 depmap_id,
				 ccle_name,
				 auc,
				 broad_id,
				 auc_binary)

write_csv(feat5000_data, here('results/PRISM_LINCS_klaeger_all_multiomic_data_for_ml_5000feat_auc.csv'))


	