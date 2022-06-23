library(tidyverse)
library(here)
library(vroom)

data = read_rds(here('results/ALMANAC_klaeger_L1000_data_for_ml.rds.gz'))

find_feature_correlations <- function(row_indexes = NA, all_data) {
  if (is.na(row_indexes)) {
    row_indexes = 1:dim(all_data)[1]
  }
  
  all_cor = cor(
    all_data %>% 
      pull(viability),
    
    all_data %>% 
      select(starts_with(c('act_','pert_')))
  ) %>%
    as.data.frame() %>%
    pivot_longer(everything(), names_to = "feature",values_to = "cor")
  
  
  all_correlations = all_cor %>% 
    mutate(abs_cor = abs(cor)) %>% 
    arrange(desc(abs_cor)) %>% 
    mutate(rank = 1:n()) %>%
    mutate(feature_type = case_when(
      str_detect(feature, "^act_") ~ "Activation",
      str_detect(feature, "^pert_") ~ "Perturbed Transcriptomics",
      T ~ feature
    ))
  
  return(all_correlations)	
}

feat_cors = find_feature_correlations(all_data = data)

write_csv(feat_cors, here('results/ALMANAC_klaeger_L1000_models/feature_selection/correlation_selected_features.csv'))