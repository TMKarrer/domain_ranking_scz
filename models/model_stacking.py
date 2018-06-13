'''
Stacked model to rank meta-priors for their relevance in schizophrenia.
'''

# packages
import os
import pandas as pd
import numpy as np
import joblib
import copy
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# reproducibility
np.random.seed(42)

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

# parameters
method = 'pca'
taxonomy = ['BD', 'PC']
modality = 'vbm_rs'
k_list = 100
n_folds = 10
number_trees = 1000
number_bootstrapping = 1000
number_permutations = 1000
BT = True
permutation = True
depth = 5
features = 1

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
diags = np.array(data[99])

target_dir = project_dir + '/models/LogReg_RF/%s/%s/' % (modality, method)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for prior_group in taxonomy:
    print modality, prior_group
    meta_file_pathes = joblib.load(project_dir + '/data/interim/%s_Archiv_order.npy' % prior_group)
    meta_files = [path.replace('~', project_dir) for path in meta_file_pathes]
    meta_categories = [nifti_path.split(os.sep)[-1].split('_')[-1].split('.nii')[0] for nifti_path in meta_files]

    for k in str(k_list).split(','):
        print str(k) + ' ' + method
        k = int(k)

        # load feature space for base models
        listed_meta_categories, FS_list = zip(*joblib.load(project_dir + '/features/' + modality + os.sep + method + os.sep + prior_group + '_' + method + '_feature_space_' + str(k) + '.npy'))
        assert (list(listed_meta_categories) == meta_categories)

        seed = np.random.RandomState(42)  # reproducible bootstrapping
        acc_RF = []
        acc_PT = []
        acc_LR = []
        prior_importance = []
        if BT:
            BT_prior_importances = []
            BT_acc_list = []

        # outer CV loop
        outer_folder = StratifiedKFold(diags, n_folds=n_folds, shuffle=True, random_state=42)
        final_predictions = np.ones((len(diags), len(meta_categories))) * (-1000)
        pt_seed = np.random.RandomState(42)  # reproducible permutation

        for outer_fold, (train_val_idx, test_idx) in enumerate(outer_folder):
            acc_list_LR = []
            print 'OUTER CV LOOP %s OF %s' % (outer_fold + 1, n_folds)
            assert (len(np.intersect1d(train_val_idx, test_idx)) == 0)

            val_predictions = np.ones((len(diags), len(meta_categories))) * (-1000)
            test_predictions = np.ones((len(diags), len(meta_categories))) * (-1000)

            # inner CV loop
            inner_folder = StratifiedKFold(diags[train_val_idx], n_folds=n_folds, shuffle=True, random_state=42)
            for inner_fold, (train_idx, val_idx) in enumerate(inner_folder):
                print 'Inner CV loop %s of %s' % (inner_fold+1, n_folds)
                assert (len(np.intersect1d(train_idx, val_idx)) == 0)
                assert (len(np.intersect1d(train_val_idx[train_idx], test_idx)) == 0)
                assert (len(np.intersect1d(train_val_idx[val_idx], test_idx)) == 0)

                for i_meta, meta_category in enumerate(meta_categories):
                    assert (meta_category == listed_meta_categories[i_meta])
                    print 'Current MetaPrior is %s' % meta_category
                    scaler = StandardScaler()
                    FS_prior = scaler.fit_transform(np.array(FS_list)[i_meta, :, :])

                    inner_clf = LogisticRegression(random_state=42, tol=0.0001, max_iter=1000, penalty='l2', dual=True)

                    if modality == 'vbm':
                        inner_clf.fit(FS_prior[train_val_idx[train_idx]], diags[train_val_idx[train_idx]])
                        val_predictions[train_val_idx[val_idx], i_meta] = inner_clf.predict_proba(FS_prior[train_val_idx[val_idx]])[:, 1]

                    else:
                        from sklearn.pipeline import Pipeline
                        from sklearn.feature_selection import SelectPercentile, f_classif
                        feat_sel = SelectPercentile(f_classif, percentile=25)
                        inner_clf_pipe = Pipeline([('fs', feat_sel), ('lr', inner_clf)])
                        inner_clf_pipe.fit(FS_prior[train_val_idx[train_idx]], diags[train_val_idx[train_idx]])
                        val_predictions[train_val_idx[val_idx], i_meta] = inner_clf_pipe.predict_proba(FS_prior[train_val_idx[val_idx]])[:, 1]

            # continue outer CV loop
            for i_meta, meta_category in enumerate(meta_categories):
                assert (meta_category == listed_meta_categories[i_meta])
                print 'Current MetaPrior is %s' % meta_category
                FS_prior = np.array(FS_list)[i_meta, :, :]
                outer_clf = LogisticRegression(random_state=42, tol=0.0001, max_iter=1000, penalty='l2', dual=True)

                if modality == 'vbm':
                    outer_clf.fit(FS_prior[train_val_idx], diags[train_val_idx])
                    test_predictions[test_idx, i_meta] = outer_clf.predict_proba(FS_prior[test_idx])[:, 1]
                    base_pred_labels = outer_clf.predict(FS_prior[test_idx])

                else:
                    feat_sel = SelectPercentile(f_classif, percentile=25)
                    outer_clf_pipe = Pipeline([('fs', feat_sel), ('lr', outer_clf)])
                    outer_clf_pipe.fit(FS_prior[train_val_idx], diags[train_val_idx])
                    test_predictions[test_idx, i_meta] = outer_clf_pipe.predict_proba(FS_prior[test_idx])[:, 1]
                    base_pred_labels = outer_clf_pipe.predict(FS_prior[test_idx])

                acc_list_LR.append(np.mean(base_pred_labels == diags[test_idx]))
            acc_LR.append(acc_list_LR)

            assert np.all(val_predictions[train_val_idx] != -1000)
            assert np.all(test_predictions[test_idx] != -1000)

            stacked_clf = RandomForestClassifier(n_estimators=number_trees, max_features=features, max_depth=depth, random_state=42)
            stacked_clf.fit(val_predictions[train_val_idx], diags[train_val_idx])
            pred_labels = stacked_clf.predict(test_predictions[test_idx])
            cur_acc = np.mean(pred_labels == diags[test_idx])
            acc_RF.append(cur_acc)

            # permutation test
            if permutation:
                for p in range(number_permutations/n_folds):
                    print 'permutation cycle: ' + str(p)
                    val_predictions_pt = np.ones((len(diags), len(meta_categories))) * (-1000)
                    test_predictions_pt = np.ones((len(diags), len(meta_categories))) * (-1000)
                    for s in range(len(meta_categories)):
                        train_val_idx_pt = copy.deepcopy(train_val_idx)
                        test_idx_pt = copy.deepcopy(test_idx)

                        train_val_idx_pt_SZ = train_val_idx_pt[diags[train_val_idx] == 1]
                        train_val_idx_pt_HC = train_val_idx_pt[diags[train_val_idx] == 0]

                        test_idx_pt_SZ = test_idx_pt[diags[test_idx] == 1]
                        test_idx_pt_HC = test_idx_pt[diags[test_idx] == 0]

                        pt_seed.shuffle(train_val_idx_pt_SZ)
                        pt_seed.shuffle(train_val_idx_pt_HC)

                        pt_seed.shuffle(test_idx_pt_SZ)
                        pt_seed.shuffle(test_idx_pt_HC)

                        assert len(set(train_val_idx_pt_SZ).symmetric_difference(set(train_val_idx[diags[train_val_idx] == 1]))) == 0
                        assert len(set(train_val_idx_pt_HC).symmetric_difference(set(train_val_idx[diags[train_val_idx] == 0]))) == 0
                        assert len(set(test_idx_pt_SZ).symmetric_difference(set(test_idx[diags[test_idx] == 1]))) == 0
                        assert len(set(test_idx_pt_HC).symmetric_difference(set(test_idx[diags[test_idx] == 0]))) == 0

                        train_val_idx_pt_s = np.concatenate((train_val_idx_pt_SZ, train_val_idx_pt_HC))
                        test_idx_pt_s = np.concatenate((test_idx_pt_SZ, test_idx_pt_HC))

                        assert len(train_val_idx) == len(train_val_idx_pt_s)
                        assert len(test_idx) == len(test_idx_pt_s)
                        assert len(set(train_val_idx).symmetric_difference(set(train_val_idx_pt_s))) == 0
                        assert len(set(test_idx).symmetric_difference(set(test_idx_pt_s))) == 0

                        val_predictions_pt[train_val_idx, s] = val_predictions[train_val_idx_pt_s, s]
                        test_predictions_pt[test_idx, s] = test_predictions[test_idx_pt_s, s]

                    assert np.all(val_predictions_pt[train_val_idx] != -1000)
                    assert np.all(test_predictions_pt[test_idx] != -1000)

                    stacked_clf_pt = RandomForestClassifier(n_estimators=number_trees, max_features=features, max_depth=depth, random_state=42)
                    stacked_clf_pt.fit(val_predictions_pt[train_val_idx], diags[train_val_idx])
                    pred_labels_pt = stacked_clf_pt.predict(test_predictions_pt[test_idx])
                    cur_acc_pt = np.mean(pred_labels_pt == diags[test_idx])
                    acc_PT.append(cur_acc_pt)

            final_predictions[test_idx] = test_predictions[test_idx]

            # bootstrapping prediction accuracy
            if BT:
                for i in range(number_bootstrapping/n_folds):
                    print 'BT %s of %s' % (str(i+1), str(number_bootstrapping/n_folds))
                    ind = seed.randint(0, number_trees, int(0.8*number_trees))
                    BT_clf = copy.deepcopy(stacked_clf)
                    BT_trees = []
                    for i in ind:
                        BT_trees.append(BT_clf.estimators_[i])
                    BT_clf.estimators_ = BT_trees
                    BT_pred_labels = BT_clf.predict(test_predictions[test_idx])
                    BT_cur_acc = np.mean(BT_pred_labels == diags[test_idx])
                    BT_acc_list.append(BT_cur_acc)

        print 'Random Forest Mean Accuracy: %f' % np.mean(acc_RF)
        if permutation:
            assert len(acc_PT) == number_permutations
            print 'Permutation Tests Mean Accuracy: %f' % np.mean(acc_PT)

        if BT:
            assert (len(BT_acc_list) == number_bootstrapping)

        final_clf = RandomForestClassifier(n_estimators=number_trees, max_features=features, max_depth=depth, random_state=42)
        final_clf.fit(final_predictions, diags)
        prior_importances = final_clf.feature_importances_

        # bootstrapping feature importances
        if BT:
            print 'bootstrapping CI for feature importances'
            for i in range(number_bootstrapping):
                print 'BT %s/%s' % (str(i+1), str(number_bootstrapping))
                ind = seed.randint(0, number_trees, int(0.8*number_trees))
                BT_clf = copy.deepcopy(final_clf)
                BT_trees = []
                for i in ind:
                    BT_trees.append(BT_clf.estimators_[i])
                BT_clf.estimators_ = BT_trees
                BT_prior_importances.append(BT_clf.feature_importances_)

        # meta_summary
        meta_summary_LR = []
        meta_summary_RF = []
        mean_acc_LR = np.array(acc_LR).mean(axis=0)
        for i, meta_category in enumerate(meta_categories):
            meta_summary_LR.append([prior_group, meta_category, k, mean_acc_LR[i]])
            meta_summary_RF.append([prior_group, meta_category, k, prior_importances[i]])

        np.save(target_dir + prior_group + '_' + method + '_first_meta_summary_' + str(k) + '_LR_%f_acc.npy' % mean_acc_LR.max(), meta_summary_LR)
        np.save(target_dir + prior_group + '_' + method + '_first_meta_summary_' + str(k) + '_RF_%f_acc.npy' % np.mean(acc_RF), meta_summary_RF)
        np.save(target_dir + prior_group + '_' + method + '_' + str(k) + '_acc_list', acc_RF)

        joblib.dump(final_clf, target_dir + prior_group + '_' + method + '_final_stacked_clf_' + str(k) + '.npy')
        joblib.dump(final_predictions, target_dir + prior_group + '_' + method + '_final_predictions_' + str(k) + '.npy')
        if permutation:
            np.save(target_dir + prior_group + '_' + method + '_' + str(k) + '_permutation_acc_list_%f.npy' % np.mean(acc_PT), acc_PT)

        if BT:
            np.save(target_dir + prior_group + '_' + method + '_' + str(k) + '_BT_prior_importance', BT_prior_importances)
            np.save(target_dir + prior_group + '_' + method + '_' + str(k) + '_BT_acc_list', BT_acc_list)
