'''
Visualization of the decision boundary of the classifier.
'''

# packages
import numpy as np
import pandas as pd
import joblib
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold.t_sne import TSNE
from sklearn.neighbors import KNeighborsRegressor

# reproducibilty
np.random.seed(42)

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

# parameters
method = 'pca'
taxonomy = ['BD', 'PC']
method_name = 'PCA'
modality = 'vbm_rs'
k = '100'

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
y = np.array(data[99])
cm = plt.cm.viridis
cm_bright = ListedColormap(['darkviolet', 'gold'])

for prior_group in taxonomy:
    X = joblib.load(target_dir + '%s_%s_final_predictions_%s.npy' % (prior_group, method, k))
    X_Train_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
    print X_Train_embedded.shape
    model = joblib.load(target_dir + '%s_%s_final_stacked_clf_%s.npy' % (prior_group, method, k))
    y_predicted = model.predict_proba(X)[:, 1]

    ranking_files = glob.glob(target_dir + prior_group + '_' + method + '_first_meta_summary_' + str(k) + '_RF*.npy')
    if len(ranking_files) > 1:
        print 'too many files'
        stop
    elif len(ranking_files) == 0:
        acc = None
    else:
        ranking_file = ranking_files[0]
        acc = float(ranking_file.split('_RF')[-1].split('.npy')[0].split('_')[-2])

    resolution = 100
    X2d_xmin, X2d_xmax = np.min(X_Train_embedded[:, 0]) - 0.5, np.max(X_Train_embedded[:, 0] + 0.5)
    X2d_ymin, X2d_ymax = np.min(X_Train_embedded[:, 1]) - 0.5, np.max(X_Train_embedded[:, 1] + 0.5)
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

    background_model = KNeighborsRegressor(n_neighbors=1).fit(X_Train_embedded, y_predicted)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    plt.contourf(xx, yy, voronoiBackground, cmap=cm, alpha=0.7)
    plt.xticks(())
    plt.yticks(())
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(X_Train_embedded[:, 0], X_Train_embedded[:, 1], c=y, cmap=cm_bright, edgecolors='k', alpha=0.6)
    t = plt.text(xx.max() - 0.5, yy.min() + 0.7, 'Classification perfomance: ' + ('{:.2%}'.format(acc)).lstrip('0') + '\nDecision boundary of random forest (%s)' % method_name, size=15, horizontalalignment='right', fontsize=18)
    t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='k'))
    plt.xlabel('disease dimension 1', fontsize=18)
    plt.ylabel('disease dimension 2', fontsize=18)
    plt.tight_layout()
    plt.savefig(target_dir + '%s_%s_final_decision_boundary_%s.png' % (prior_group, method, k), dpi=500, transparent=True)
    plt.close()
