library(tidyverse)
library(here)
library(vroom)

binary_data = vroom(here('results/PRISM_LINCS_klaeger_binary_data_for_ml.csv'))

find_feature_correlations <- function(row_indexes = NA, all_data) {
  if (is.na(row_indexes)) {
    row_indexes = 1:dim(all_data)[1]
  }
  
  all_cor = cor(
    all_data %>% 
      pull(ic50),
    
    all_data %>% 
      select(starts_with(c('exp')))
  ) %>%
    as.data.frame() %>%
    pivot_longer(everything(), names_to = "feature",values_to = "cor")
  
  
  all_correlations = all_cor %>% 
    mutate(abs_cor = abs(cor)) %>% 
    arrange(desc(abs_cor)) %>% 
    mutate(rank = 1:n()) %>%
    mutate(feature_type = case_when(
      str_detect(feature, "^exp_") ~ "Expression",
      T ~ feature
    ))
  
  return(all_correlations)	
}

binary_feat_cors = find_feature_correlations(all_data = binary_data)

write_csv(binary_feat_cors, here('results/PRISM_LINCS_klaeger_binary_data_feature_correlations.csv'))