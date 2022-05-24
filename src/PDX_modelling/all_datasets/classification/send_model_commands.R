library(here)
library(tidyverse)

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {

		job_name = sprintf('a_RF_%d',feature_num)

		command = sprintf('sbatch --job-name=%s --mem=90G -c 8 --time=120:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/classification/build_rf_models_ANOVA.R --feature_num %d"', job_name, feature_num)

		# print(command)
		system(command)

}

for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {

	job_name = sprintf('a_xbg_%d',feature_num)

	command = sprintf('sbatch --job-name=%s --mem=90G -c 8 --time=120:00:00 --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/classification/build_xgboost_models_ANOVA.R --feature_num %d"', job_name, feature_num)

	# print(command)
	system(command)

}


for (feature_num in c(100,200,300,400,500,1000,1500,2000,3000,4000,5000)) {
	
	
	job_name = sprintf('a_NN_%d',feature_num)
	
	command = sprintf('sbatch -N 1 -n 1 -p volta-gpu --job-name=%s --mem=90G --time=120:00:00 --qos gpu_access --gres=gpu:1 --mail-user=cujoisa@live.unc.edu   --wrap "Rscript src/LINCS_modeling/build_LINCS_klaeger_ic50_models/all_data_models/classification/build_NN_models.R --feature_num %d"', job_name, feature_num)
	
	# print(command)
	system(command)
	
}