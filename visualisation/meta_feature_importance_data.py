'''
Transformation of the data for the average feature importance plot.
'''

# packages
import os
import pandas as pd
import numpy as np
from scikits import bootstrap

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

method_group = 'all_methods'
taxonomy = ['BD', 'PC']
methods = ['kmeans', 'ward', 'spectral', 'pca', 'ica', 'sparse_pca', 'ha_pc', 'md_pc', 'ts_pc']
k = [100, 100, 100, 100, 100, 100, 5000, 5000, 5000]
title = 'peak_coordinates'

for modality in ['vbm', 'rs', 'vbm_rs']:
    for prior_group in taxonomy:
        print prior_group

        # average rank position across pipelines
        rank_data_mean = data_mean.rank(ascending=False)
        rank_mean_data_mean = pd.DataFrame(zip(mean_priors, rank_data_mean.mean(axis=1)), columns=['priors', 'means'])
        rank_ci_data = pd.DataFrame(ci_data).rank(ascending=False, axis=1)

        rank_distr_values = []
        rank_distr_priors = []
        for i in range(len(rank_ci_data.columns)):
            for j in range(9000):
                rank_distr_values.append(rank_ci_data[i][j])
                rank_distr_priors.append(mean_file['priors'][i])
        rank_distr_data = pd.DataFrame(zip(rank_distr_priors, rank_distr_values), columns=['priors', 'data'])

        rank_ci_priors = []
        rank_ci_values = []

        i = 0
        def fi_function(input_data):
            global i
            i += 1
            fi_row = np.array(rank_ci_data)[i-1]
            return fi_row
        rank_ci = bootstrap.ci(np.array(rank_ci_data), fi_function, n_samples=9000, method='pi')

        for i, prior in enumerate(mean_file['priors']):
            rank_ci_values.append(rank_ci[0][i])
            rank_ci_values.append(rank_ci[1][i])
            rank_ci_priors.append(prior)
            rank_ci_priors.append(prior)

        rank_data_ci = pd.DataFrame(zip(rank_ci_priors, rank_ci_values), columns=['priors', 'values'])

        new_target_dir = project_dir + '/models/LogReg_RF/' + modality + '/_meta_ranking/' + method_group + os.sep
        if not os.path.exists(new_target_dir):
            os.makedirs(new_target_dir)
        rank_mean_data_mean.to_csv(new_target_dir + prior_group + '_' + modality + '_' + method_group + '_rank_BT_data_mean.csv')
        rank_data_ci.to_csv(new_target_dir + prior_group + '_' + modality + '_' + method_group + '_rank_BT_data_ci.csv')
        rank_distr_data.to_csv(new_target_dir + prior_group + '_' + modality + '_' + method_group + '_rank_BT_data_joy.csv')
