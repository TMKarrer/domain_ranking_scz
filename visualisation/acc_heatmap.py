'''
Visualisation of a heatmap of out-of-sample prediction accuracies.
'''

# packages
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcdefaults()

project_dir = 'scz_ranking_project'
taxonomy = ['BD', 'PC']

# parameters
methods = ['ha_pc', 'md_pc', 'ts_pc', 'kmeans', 'ward', 'spectral', 'pca', 'ica', 'sparse_pca']
k_list = [5000, 5000, 5000, 100, 100, 100, 100, 100, 100]
title = 'RF prediction accuracy across all methods'
filename = '_acc_comparison.png'

for prior_group in taxonomy:
    print prior_group
    acc_data = []

    for method, k in zip(methods, k_list):
        print method

        if prior_group == 'BD':
            prior_group_name = 'Mental domains'
        else:
            prior_group_name = 'Experimental tasks'

        modality_data = [method]
        modality_labels = ['method']
        for modality in ['vbm', 'rs', 'vbm_rs']:
            modality_labels.append(modality + '_acc')

            target_dir = project_dir + '/models/LogReg_RF/%s/%s/' % (modality, method)
            ranking_files = glob.glob(target_dir + prior_group + '_' + method + '_first_meta_summary_' + str(k) + '_RF*.npy')
            if len(ranking_files) > 1:
                print 'too many files'
                stop
            elif len(ranking_files) == 0:
                acc = None
            else:
                ranking_file = ranking_files[0]
                acc = float(ranking_file.split('_RF')[-1].split('.npy')[0].split('_')[-2])

            modality_data.append(acc)

        acc_data.append(modality_data)

    data = pd.DataFrame(acc_data, columns=modality_labels)
    data = data.set_index(['method'])
    data = data * 100
    data = data.round(2)
    data.to_csv(project_dir + '/models/LogReg_RF/%s_accuracy_list.csv' % prior_group)

    # heatmap
    plt.close()
    data.columns = ['Structure', 'Function', 'Structure + Function']
    data.index = ['Highest act. peaks', 'Mean act. peaks', 'Standardized act. peaks', 'K-means clustering', 'Ward clustering', 'Spectral clustering', 'PCA', 'ICA', 'Sparse PCA']
    sns.heatmap(data, linewidth=1.5, annot=True, fmt='g', cmap='YlGnBu_r', center=69.5, annot_kws={'color': 'black'}, cbar=False)
    plt.yticks(rotation=0)
    plt.title('Prediction accuracy of composite model [%]')
    plt.tick_params(bottom='off', top='off', left='off', right='off')
    plt.tight_layout()
    plt.savefig(project_dir + '/models/LogReg_RF/%s_accuracy_heatmap.png' % prior_group, dpi=500)
