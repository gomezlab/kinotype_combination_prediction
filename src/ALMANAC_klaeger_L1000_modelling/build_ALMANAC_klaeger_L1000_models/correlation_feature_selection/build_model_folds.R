library(tidyverse)
library(here)
library(vroom)
library(tidymodels)
library(finetune)
library(tictoc)
library(doParallel)
library(patchwork)
library(ROCR)

data = read_rds(here('results/ALMANAC_klaeger_L1000_models/ALMANAC_klaeger_L1000_data_for_ml_5000feat.rds.gz'))
data_10000 = read_rds(here('results/ALMANAC_klaeger_L1000_models/ALMANAC_klaeger_L1000_data_for_ml_10000feat.rds.gz'))

set.seed(2222)
folds = vfold_cv(data, v = 5) %>% 
  write_rds(here('results/cv_folds/ALMANAC_klaeger_L1000_folds.rds.gz'), compress = "gz")

set.seed(2222)
folds = vfold_cv(data_10000, v = 5) %>% 
  write_rds(here('results/cv_folds/ALMANAC_klaeger_L1000_folds_10000.rds.gz'), compress = "gz")