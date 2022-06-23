library(here)
library(tidyverse)

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
  
  job_name = sprintf('cRF%d',feature_num)
  
  command = sprintf('sbatch --job-name=%s --mem=99G -c 16 --time=40:00:00 --wrap "Rscript src/ALMANAC_klaeger_L1000_modelling/build_ALMANAC_klaeger_L1000_models/correlation_feature_selection/build_rf_models.R --feature_num %d"', job_name, feature_num)
  
  # print(command)
  system(command)
  
}

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {

	job_name = sprintf('clr%d',feature_num)

	command = sprintf('sbatch --job-name=%s --mem=99G -c 16 --time=40:00:00 --wrap "Rscript src/ALMANAC_klaeger_L1000_modelling/build_ALMANAC_klaeger_L1000_models/correlation_feature_selection/build_lasso_models.R --feature_num %d"', job_name, feature_num)

	# print(command)
	system(command)

}

for (feature_num in c(400,500,1000,1500,2000,3000,4000,5000)) {
  
  job_name = sprintf('cxg%d',feature_num)
  
  #command = sprintf('sbatch -N 1 -n 1 -p gpu --job-name=%s --mem=90G --time=90:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_auc_models_matched_only/activation_expression_models/regression/build_xgboost_models_ANOVA_GPU.R --feature_num %d"', job_name, feature_num)
  command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --constraint=rhel8 --job-name=%s --mem=90G --time=48:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu --wrap=\"echo \' module load r/4.1.0; module load cuda/11.4; module load gcc/9.1.0; Rscript src/ALMANAC_klaeger_L1000_modelling/build_ALMANAC_klaeger_L1000_models/correlation_feature_selection/build_xgboost_models.R --feature_num %d \' | bash\"', job_name, feature_num)
  
  # print(command)
  system(command)
  
}