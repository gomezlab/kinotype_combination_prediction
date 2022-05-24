library(tidyverse)
library(here)
library(vroom)

data = vroom(here('results/PRISM_LINCS_klaeger_data_for_ml.csv'))

find_all_data_feature_correlations <- function(row_indexes = NA, all_data) {
  if (is.na(row_indexes)) {
    row_indexes = 1:dim(all_data)[1]
  }
  
  all_cor = cor(
    all_data %>% 
      pull(ic50),
    
    all_data %>% 
      select(starts_with(c('act','exp')))
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
      T ~ feature
    ))
  
  return(all_correlations)	
}

feat_cors = find_feature_correlations(all_data = data)

write_csv(feat_cors, here('results/PRISM_LINCS_klaeger_data_feature_correlations.csv'))