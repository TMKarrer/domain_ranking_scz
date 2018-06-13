'''
Combination of the neurobiological information from brain structure and function.
'''

# packages
import os
import numpy as np
import pandas as pd
import joblib

# reproducibility
np.random.seed(42)

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
taxonomy = ['BD', 'PC']

for method, k in zip(['ts_pc', 'ha_pc', 'md_pc', 'kmeans', 'ward', 'spectral', 'pca', 'sparse_pca', 'ica'], [5000, 5000, 5000, 100, 100, 100, 100, 100, 100]):
    print method, k

    for prior_group in taxonomy:
        print prior_group

        vbm_meta_categories, vbm_features = zip(*joblib.load(project_dir + '/features/vbm/' + method + os.sep + prior_group + '_' + method + '_feature_space_' + str(k) + '.npy'))
        rs_meta_categories, rs_features = zip(*joblib.load(project_dir + '/features/rs/' + method + os.sep + prior_group + '_' + method + '_feature_space_' + str(k) + '.npy'))

        assert (vbm_meta_categories == rs_meta_categories)

        combined_features = []
        for i in range(len(vbm_features)):
            combi = np.concatenate((vbm_features[i], rs_features[i]), axis=1)
            if k == 5000:
                assert combi.shape == (len(data), (k+(k/250*k/250-k/250)/2))
            elif k == 100:
                assert combi.shape == (len(data), k+(k*k-k)/2)

            combined_features.append(combi)
        assert len(combined_features) == len(rs_features)

        target_dir = project_dir + '/features/vbm_rs/%s/' % method
        if not os.path.exists(target_dir):
                os.makedirs(target_dir)

        combined_features_prior = zip(vbm_meta_categories, combined_features)
        joblib.dump(combined_features_prior, target_dir + prior_group + '_' + method + '_feature_space_' + str(k) + '.npy')
