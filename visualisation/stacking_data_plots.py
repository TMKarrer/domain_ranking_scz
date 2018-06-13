'''
Transformation of the data for the feature importance plot.
'''

# packages
import os
import glob
import joblib
import pandas as pd
import numpy as np
from scikits import bootstrap

# reproducibiltiy
np.random.seed(42)

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

# parameters
method = 'pca'
taxonomy = ['BD', 'PC']
k_list = 100
n_folds = 10
number_trees = 1000
number_bootstrapping = 100
stacked_model = 'RF'
BT = True

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
diags = np.array(data[99])

for modality in ['vbm', 'rs', 'vbm_rs']:
    print modality
    for method, k_list in zip(['ha_pc', 'md_pc', 'ts_pc', 'kmeans', 'ward', 'spectral', 'ica', 'sparse_pca', 'pca'], [5000, 5000, 5000, 100, 100, 100, 100, 100, 100]):
        print modality, method
        target_dir = project_dir + '/models/LogReg_RF/%s/%s/' % (modality, method)

        filelist = os.listdir(target_dir)
        for item in filelist:
            for ext in ['.png', '.csv', '.pdf']:
                if item.endswith(ext):
                    print 'deletes ' + os.path.join(target_dir, item)
                    os.remove(os.path.join(target_dir, item))

        for prior_group in taxonomy:
            print prior_group
            meta_file_pathes = joblib.load(project_dir + '/data/interim/%s_Archiv_order.npy' % prior_group)
            meta_files = [path.replace('~', project_dir) for path in meta_file_pathes]
            meta_categories = [nifti_path.split(os.sep)[-1].split('_')[-1].split('.nii')[0] for nifti_path in meta_files]

            for k in str(k_list).split(','):
                print str(k) + ' ' + method
                k = int(k)

                ranking_files = glob.glob(target_dir + prior_group + '_' + method + '_first_meta_summary_' + str(k) + '_RF*.npy')
                if len(ranking_files) > 1:
                    print 'too many files'
                    stop
                else:
                    ranking_file = ranking_files[0]

                data_ranking = np.load(ranking_file)

                # bootstrapped confidence intervals
                prior_labels = data_ranking.T[1]
                BT_prior_importance = np.load(target_dir + prior_group + '_' + method + '_' + str(k) + '_BT_prior_importance.npy')
                acc_list = np.load(target_dir + prior_group + '_' + method + '_' + str(k) + '_BT_acc_list.npy')
                acc = round(acc_list.mean() * 100, 1)
                data = pd.DataFrame(BT_prior_importance, columns=prior_labels)
                data_mean = pd.DataFrame(data_ranking.T[-1], columns=['means'])
                data_mean['priors'] = prior_labels
                if prior_group == 'BD':
                    xname = 'Behaviors'
                elif prior_group == 'PC':
                    xname = 'Tasks'

                priors = []
                values = []

                i = 0

                def fi_function(input_data):
                    global i
                    i += 1
                    fi_row = BT_prior_importance[i-1]
                    return fi_row
                ci = bootstrap.ci(BT_prior_importance, fi_function, n_samples=1000, method='pi')

                for i, prior in enumerate(data.columns):
                    values.append(ci[0][i])
                    values.append(ci[1][i])
                    priors.append(prior)
                    priors.append(prior)

                data_ci = pd.DataFrame(zip(values, priors))
                data_ci.columns = ['values', 'priors']

                data_list = []
                prior_list = []
                for p in data.columns:
                    cur_data = data[p]
                    for i in range(1000):
                        prior_list.append(p)
                        data_list.append(cur_data[i])

                data_joy = pd.DataFrame(zip(prior_list, data_list), columns=['priors', 'data'])

                data_mean.to_csv(target_dir + prior_group + '_' + method + '_' + str(k) + '_BT_data_mean.csv')
                data_ci.to_csv(target_dir + prior_group + '_' + method + '_' + str(k) + '_BT_data_ci.csv')
                data_joy.to_csv(target_dir + prior_group + '_' + method + '_' + str(k) + '_BT_data_joy.csv')
