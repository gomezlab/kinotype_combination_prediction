library(tidyverse)
library(here)
library(vroom)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 10)
data = vroom(here('results/PRISM_LINCS_klaeger_data_for_ml.csv'))
cors =  vroom(here('results/PRISM_LINCS_klaeger_data_feature_correlations.csv'))

feat5000_data = data %>% 
  select(any_of(cors$feature[1:5005]),
         depmap_id,
         ccle_name,
         ic50,
         broad_id,
         ic50_binary)

write_csv(feat5000_data, here('results/PRISM_LINCS_klaeger_data_for_ml_5000feat.csv'))


