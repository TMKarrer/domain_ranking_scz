'''
Computation of the significance level of permutation tests.
'''

# packages
import pandas as pd
import scipy.stats

project_dir = 'scz_ranking_project'

taxonomy = ['BD', 'PC']
modalities = ['vbm', 'rs', 'vbm_rs']
methods = ['ha_pc', 'md_pc', 'ts_pc', 'kmeans', 'ward', 'spectral', 'pca', 'ica', 'sparse_pca']

for prior_group in taxonomy:
    print prior_group
    for modality in modalities:
        print modality
        res_modality = []
        res_method = []
        res_percentile = []
        res_signlevel = []

        pdata = pd.read_csv(project_dir + '/models/LogReg_RF/' + modality + '/_meta_ranking/' + prior_group + '_acc_distribution_pt.csv')
        perm_data = pdata[pdata.type == 'perm_test']
        mdata = pd.read_csv(project_dir + '/models/LogReg_RF/' + modality + '/_meta_ranking/' + prior_group + '_acc_ci_pt.csv')
        mean_data = mdata[mdata.type == 'result']
        for method in methods:
            res_modality.append(modality)
            res_method.append(method)
            print method
            mean = mean_data[mean_data.methods = =method].means.iloc[0]
            perm_list = perm_data[perm_data.methods == method].values
            res = scipy.stats.percentileofscore(perm_list, mean, kind='strict')
            res_percentile.append(res)
            res_signlevel.append((100-res)/100)
        results = pd.DataFrame(zip(res_modality, res_method, res_percentile, res_signlevel))
        results.columns = ['modality', 'method', 'percentile', 'signlevel']
        results.to_csv(project_dir + '/models/LogReg_RF/' + modality + '/_meta_ranking/' + prior_group + '_permutation_signficance_%f.csv' % (max(res_signlevel)))
