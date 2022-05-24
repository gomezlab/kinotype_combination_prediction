library(tidyverse)
library(here)
library(vroom)

data = vroom(here('results/PDX_klaeger_LINCS_data_for_ml.csv'))
cors =  vroom(here('results/PXD_LINCS_klaeger_data_feat_cors.csv'))

data5000 = data %>% 
	select(-starts_with("cnv_")) %>% 
	select(Model,
				 Treatment,
				 ResponseCategory,
				 BestAvgResponse,
				 binary_response,
				 below_median_response,
				 any_of(feature_correlations$feature[1:5100]),
	)

all_data5000 = data %>% 
	select(-starts_with("cnv_")) %>% 
	select(Model,
				 Treatment,
				 ResponseCategory,
				 BestAvgResponse,
				 binary_response,
				 below_median_response,
				 any_of(feature_correlations$feature[1:5100]),
	)

write_csv(data5000, here('results/PDX_LINCS_klaeger_data_for_ml_5000feat.csv'))
write_csv(all_data5000, here('results/PDX_LINCS_klaeger_data_with_CNV_for_ml_5000feat.csv'))