'''
Meta-prior wise determination of the peak-coordinates.
'''

# packages
import os
import copy
import random
import numpy as np
import pandas as pd
import joblib
import nibabel as nib
import nilearn
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from skimage import measure
from scipy.ndimage import binary_dilation
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# reproducibility
np.random.seed(42)

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

# parameters
taxonomy = ['BD', 'PC']
k_list = 5000
method = 'ha_pc'

# peak_coordinate methods
if method == 'ha_pc':
    title = 'highest-activation_voxels'
    mapname = '_maps_ha_pc'
elif method == 'md_pc':
    title = 'max-distance-to-mean_voxels'
    mapname = '_maps_md_pc'
elif method == 'ts_pc':
    title = 'top-standardized_voxels'
    mapname = '_maps_ts_pc'

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
diags = np.array(data[99])
gm_mask = nib.load(project_dir + '/data/raw/icbm_fsl.nii')

for prior_group in taxonomy:
    print prior_group
    meta_files_pathes = joblib.load(project_dir + '/data/interim/%s_Archiv_order.npy' % prior_group)
    meta_files = [path.replace('~', project_dir) for path in meta_files_pathes]
    meta_categories = [nifti_path.split(os.sep)[-1].split('_')[-1].split('.nii')[0] for nifti_path in meta_files]
    print 'Found %i %s meta priors' % (len(meta_files), prior_group)

    # concatenate priors
    GM_masker = NiftiMasker(project_dir + '/data/raw/ref_vbm_bin.nii.gz')
    GM_masker.fit()
    prior_voxels = []
    meta_check1 = []
    for meta_file in meta_files:
        prior = nib.load(meta_file)
        prior_masked = GM_masker.transform(prior)
        prior_voxels.append(prior_masked[0])
        meta_check1.append(meta_file)
    prior_array = np.array(prior_voxels)

    # process prior_space
    if method == 'ha_pc':
        new_prior_array = copy.deepcopy(prior_array)
    elif method == 'md_pc':
        new_prior_array = prior_array - np.mean(prior_array, axis=0)
    elif method == 'ts_pc':
        new_prior_array = preprocessing.StandardScaler().fit_transform(prior_array)

    # extract voxel maps for each prior and k
    for k in str(k_list).split(','):
        k = int(k)
        print 'k = ' + str(k)
        directory = project_dir + '/data/processed/' + prior_group + mapname + os.sep + prior_group + '_' + str(k)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # the k/250 peak coordinates with surrounding spheres of 250 voxels
        less_than_twenty = 0
        result = []
        threshold_list = []
        meta_check2 = []
        for i_prior in range(len(new_prior_array)):
            prior_name = meta_files[i_prior].split(os.sep)[-1].split('.nii')[0]
            print 'Get rough threshold for ' + prior_name
            prior = GM_masker.inverse_transform(np.squeeze(new_prior_array[i_prior]))
            threshold = prior.get_data().max()
            num_peaks = 0
            iteration = 0
            meta_check2.append(meta_files[i_prior])

            # find rough threshold to substract in finer process
            while num_peaks < int(k)/250:
                iteration += 1
                threshold = threshold - 0.1
                prior_t = copy.deepcopy(prior.get_data())
                prior_t[prior_t < threshold] = 0
                prior_t[prior_t > threshold] = 1
                num_peaks = measure.label(prior_t, return_num=True, neighbors=4)[1]
                print threshold, num_peaks
                if iteration > 200:
                    less_than_twenty += 1
                    break
            print threshold
            threshold_list.append(prior.get_data().max() - threshold)
        substract_threshold = np.array(threshold_list) - 0.1

        # find exact threshold for 20 connected components
        meta_check3 = []
        for i_prior in range(len(new_prior_array)):
            meta_check3.append(meta_files[i_prior])
            prior_name = meta_files[i_prior].split(os.sep)[-1].split('.nii')[0]
            print 'Get exact threshold for ' + prior_name
            prior = GM_masker.inverse_transform(np.squeeze(new_prior_array[i_prior]))
            threshold = prior.get_data().max() - substract_threshold[i_prior]
            num_peaks = 0
            iteration = 0
            while num_peaks != int(k)/250:
                iteration += 1
                threshold = threshold - 0.001
                prior_t = copy.deepcopy(prior.get_data())
                prior_t[prior_t < threshold] = 0
                prior_t[prior_t > threshold] = 1
                num_peaks = measure.label(prior_t, return_num=True, neighbors=4)[1]
                print threshold, num_peaks
                if num_peaks > int(k)/250:
                    print 'number of peaks larger than ' + str(int(k)/250)
                    inner_iteration = 0
                    threshold = threshold + 0.001
                    prior_t = copy.deepcopy(prior.get_data())
                    prior_t[prior_t < threshold] = 0
                    prior_t[prior_t > threshold] = 1
                    num_peaks = measure.label(prior_t, return_num=True, neighbors=4)[1]
                    print threshold, num_peaks
                    while num_peaks < int(k)/250:
                        print 'entered inner iteration loop'
                        inner_iteration += 1
                        if method == 'md_pc':
                            threshold = threshold - 0.00001
                        else:
                            threshold = threshold - 0.0001
                        prior_t = copy.deepcopy(prior.get_data())
                        prior_t[prior_t < threshold] = 0
                        prior_t[prior_t > threshold] = 1
                        num_peaks = measure.label(prior_t, return_num=True, neighbors=4)[1]
                if iteration > 200:
                    less_than_twenty += 1
                    break
            print threshold
            result.append([iteration, prior.get_data().max(), threshold, (prior.get_data().max() - threshold), num_peaks, prior_t.sum()])
            assert less_than_twenty == 0

            mask = nib.Nifti1Image(measure.label(prior_t, neighbors=4), prior.get_affine())

            # target space
            tar_voxel_count = 1
            counter = 0
            tar_nii = nib.load(project_dir + '/data/raw/ref_vbm_bin.nii.gz')
            all_label_space = np.zeros(tar_nii.shape)
            all_label_space_test = np.zeros(tar_nii.shape)
            gm_weights = np.nan_to_num(gm_mask.get_data())

            prior_data = prior.get_data()
            mask_data = mask.get_data()

            # peak values
            peak_list = []
            random.seed(42)  # reproducible peaks
            for i in range(1, 21):
                label_data = copy.deepcopy(prior_data)
                label_data[mask_data != i] = -1000
                peak_coordinates = tuple(random.choice(np.argwhere(label_data == label_data.max())).tolist())
                print 'peak ' + str(i) + ' is at: ' + str(peak_coordinates)
                peak_list.append(peak_coordinates)

            # grow peak value
            seed = np.random.RandomState(42)  # reproducible peak growing
            for i in range(int(k)/(tar_voxel_count*20)):
                print i
                random_list = range(1, 21)
                seed.shuffle(random_list)
                for rand_i in random_list:
                    counter += 1
                    print rand_i
                    cur_seed_space = np.zeros(tar_nii.shape)
                    cur_seed_space[peak_list[rand_i-1]] = rand_i

                    while (True):
                        if (all_label_space == rand_i).sum() > 0:
                            rand_i_space = (all_label_space == rand_i)
                            larger_seed_space = binary_dilation(rand_i_space, mask=(all_label_space == 0))
                            cur_seed_space_new = np.logical_xor(larger_seed_space, (rand_i_space > 0))
                        else:
                            rand_i_space = cur_seed_space
                            larger_seed_space = binary_dilation(rand_i_space, mask=(all_label_space == 0))
                            cur_seed_space_new = larger_seed_space

                        # prune to highest GM probabilities
                        inds = np.where(cur_seed_space_new != 0)
                        cur_wei = gm_weights[inds]
                        isort = np.argsort(cur_wei)
                        for i_small in np.array(isort)[:len(isort) / 3]:
                            cur_x = inds[0][i_small]
                            cur_y = inds[1][i_small]
                            cur_z = inds[2][i_small]
                            cur_seed_space_new[cur_x, cur_y, cur_z] = 0
                        rand_i_space[cur_seed_space_new] = 1
                        print 'pruned to %s voxels' % cur_seed_space_new.sum()
                        if cur_seed_space_new.sum() >= tar_voxel_count:
                            break

                    # prune all voxels over tar_voxel_count
                    cur_wei = gm_weights[inds]
                    isort = np.argsort(cur_wei)
                    for i_small in np.array(isort[::-1])[tar_voxel_count:]:
                        cur_x = inds[0][i_small]
                        cur_y = inds[1][i_small]
                        cur_z = inds[2][i_small]
                        rand_i_space[cur_x, cur_y, cur_z] = 0
                    assert np.int(rand_i_space.sum()) == tar_voxel_count*(i+1)
                    print 'cut to %s voxels' % rand_i_space.sum()

                    # add peak region to voxel_space_mask
                    all_label_space[rand_i_space == 1] = rand_i
                    print 'counter: ' + str(counter)
                    print 'all_label_space: ' + str((all_label_space > 0).sum())
                    assert (all_label_space > 0).sum() == counter
                    if (all_label_space > 0).sum() == 5000:
                        break

            # add peak region to 4D label_mask
            all_voxel_space = copy.deepcopy(all_label_space)
            all_voxel_space[all_label_space > 0] = 1

            voxel_img = nib.Nifti1Image(np.array(all_voxel_space), tar_nii.get_affine(), header=tar_nii.get_header())
            label_img = nib.Nifti1Image(np.array(all_label_space), tar_nii.get_affine(), header=tar_nii.get_header())
            voxel_img.to_filename(directory + '/%s_%s_%s.nii.gz' % (prior_name, k, title))
            label_img.to_filename(directory + '/%s_%s_%s_labels.nii.gz' % (prior_name, k, title))
            plot_stat_map(label_img,
                          display_mode='ortho', colorbar=True,
                          black_bg=True, draw_cross=True,
                          bg_img=project_dir + '/data/raw/colin.nii',
                          output_file=directory + '/%s_%s_%s_labels.png' % (prior_name, k, title))
        assert (meta_check1 == meta_check2 == meta_check3 == meta_files)
