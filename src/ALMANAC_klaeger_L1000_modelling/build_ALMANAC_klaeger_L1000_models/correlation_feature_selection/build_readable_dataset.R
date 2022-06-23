library(tidyverse)
library(here)
library(vroom)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 10)
data = read_rds(here('results/ALMANAC_klaeger_L1000_data_for_ml.rds.gz'))
feats =  vroom(here('results/ALMANAC_klaeger_L1000_models/feature_selection/correlation_selected_features.csv'))

id_vars = data %>% 
  select(-starts_with(c("act_", "pert_")))

feat5000_data = data %>% 
  select(any_of(c(names(id_vars), feats$feature[1:5005])))

write_rds(feat5000_data, here('results/ALMANAC_klaeger_L1000_models/ALMANAC_klaeger_L1000_data_for_ml_5000feat.rds.gz'),
                              compress = "gz")

feat10000_data = data %>% 
  select(any_of(c(names(id_vars), feats$feature[1:10005])))

write_rds(feat10000_data, here('results/ALMANAC_klaeger_L1000_models/ALMANAC_klaeger_L1000_data_for_ml_10000feat.rds.gz'),
          compress = "gz")
