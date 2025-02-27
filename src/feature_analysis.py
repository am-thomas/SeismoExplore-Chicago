# Program to plot the correlation coefficient matrix of all desired feature pairs
# and return pairs with correlations greater than a given threshold

from constants import DATA_PATH
import argparse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import logging
import os

# from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
# from sklearn.feature_selection import SelectFromModel
# from sklearn.inspection import permutation_importance


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--clust_dir", default='clustering_test', type=str, help='name of folder with clustering results')
    parser.add_argument("-d", "--load_expdirs", required=True,  nargs='+', type=str,help='list of experiment folders (with feature values) to parse through')
    parser.add_argument("--features", nargs='+', default = 'all', 
                        help='List of feature names to use for clustering. Default is to use all available features')
    parser.add_argument("--customlabels", default=False, action=argparse.BooleanOptionalAction, 
                        help='Use custom labels to create publication figure. Note: you need to input the same 14 features with args.features: vg_Zcomp_ratio avg_H1comp_ratio avg_H2comp_ratio PSDmf_Zcomp PSDmf_H1comp PSDmf_H2comp kur_Zcomp_stw kur_H1comp_stw kur_H2comp_stw skw_Zcomp_stw skw_H1comp_stw skw_H2comp_stw hour day_of_week')
    parser.add_argument("--T", default=0.9, type=float, help="Correlation threshold. Features having abs(corr) > T are saved")
    args = parser.parse_args()

    CLUST_DIR = DATA_PATH / args.clust_dir
    os.makedirs(CLUST_DIR, exist_ok=True)
    num_feats = len(args.features)
    logging.basicConfig(level=logging.INFO, filename=str(CLUST_DIR / 'feats_analysis.log'), format='%(message)s', filemode='w')
    logger=logging.getLogger() 
    feature_list = np.array(args.features)

    # Plot params 
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.set_theme(style='whitegrid')
    plt.rcParams['savefig.dpi'] = 300

    # load and combine X and times arrays
    for i, load_dir in enumerate(args.load_expdirs):
        df = pd.read_csv(DATA_PATH / load_dir / 'PSDcuts_allfeatures.csv')
        if args.features != 'all':
            X_dir = df[args.features]
        else:
            X_dir = df.drop(columns=['start_time'])
        X_dir = X_dir.values

        if i==0: 
            X = X_dir.copy() 
        else: 
            X = np.concatenate((X, X_dir), axis=0)

    fig, ax = plt.subplots()

    # Calculate correlation matrix 
    df_features = pd.DataFrame(X, columns=args.features)
    df_corr = df_features.corr()
    if args.customlabels:
        custom_labels = ['STA/LTA [HHZ]', 'STA/LTA [HH1]', 'STA/LTA [HH2]', 
                         'Skewness [HHZ]', 'Skewness [HH1]','Skewness [HH2]',
                         'Kurtosis [HHZ]', 'Kurtosis [HH1]','Kurtosis [HH2]',
                         'PSD misfit [HHZ]', 'PSD misfit [HH1]', 'PSD misfit [HH2]',
                         'Hour','Day']
        sns.heatmap(df_corr, cmap=cmap, xticklabels=custom_labels, yticklabels=custom_labels, 
                    ax=ax, linewidth=0.3, cbar_kws={'label': 'Pearson Correlation Coefficient'})
    else:
        sns.heatmap(df_corr, cmap=cmap, xticklabels=True, yticklabels=True,
                     ax=ax, linewidth=0.3, cbar_kws={'label': 'Pearson Correlation Coefficient'})
    ax.tick_params(labelsize=6)
    plt.xticks(rotation=45, ha='right')
    #ax.set_title(f'Correlation between {num_feats} features')
    fig.tight_layout()
    fig.savefig(CLUST_DIR / f'{num_feats}_feats_corr.png')

    mask = df_corr.abs().gt(args.T) 
    df_high_corr = df_corr[mask] 
    df_high_corr = df_high_corr.stack(dropna=False)
    logger.info(f"Features having correlation > {args.T}")
    for feats, corr_val in df_high_corr.items():
        if not np.isnan(corr_val) and feats[0] != feats[1]:
            logger.info(f'{feats[0]} - {feats[1]}: {corr_val}')
    logger.info("*"*20)
  

    '''
    Y = np.load(CLUST_DIR / f'{args.nc}_clusters_{num_feats}_feats_labels.npy')
    clust_centers = np.loadtxt(CLUST_DIR / f'{args.nc}_clusters_{num_feats}_feats_clust_centers.txt')

    # For larg clusters, create subset of 5000 for faster calculation 
    subset = []
    max_points = 5000
    for label in np.unique(Y): 
        indices = np.where(Y == label)[0] 
        # Maximum of T points will be chosen from each cluster 
        if len(indices) > max_points:
            indices = np.random.permutation(indices) 
            subset.extend(list(indices[:max_points]))
        else: 
            subset.extend(list(indices))
    
    X_subset = X[subset] 
    Y_subset = Y[subset]

    # # Permutation importance 
    rf = RandomForestClassifier(max_depth=8, random_state=0)
    rf.fit(X, Y) 
    print("Random Forest Train score: ", rf.score(X, Y))
    print("Random Forest Subset score: ", rf.score(X_subset, Y_subset))
    model = SelectFromModel(rf, prefit=True)
    feature_mask = model.get_support()
    selected_feat_list = feature_list[feature_mask] 
    logger.info(f"Number of selected features by random forest: {len(selected_feat_list)}")
    logger.info(f"Selected features are: {selected_feat_list}")
    logger.info("*"*20)
    
    fig2, ax2 = plt.subplots()
    result = permutation_importance(rf, X_subset, Y_subset, n_repeats=5, n_jobs=2)
    sorted_importances_idx = result.importances_mean.argsort()
    logger.info("Features sorted by importance (decrease in accuracy) using permutation importance: ")
    logger.info(feature_list[sorted_importances_idx[::-1]])
    logger.info("*"*20)
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=feature_list[sorted_importances_idx],
    )
    importances.plot.box(vert=False, whis=10, ax=ax2)
    ax2.set_title("Permutation Importances")
    ax2.axvline(x=0, color="k", linestyle="--")
    ax2.set_xlabel("Decrease in accuracy score")
    ax2.tick_params(labelsize=5)
    fig2.savefig(CLUST_DIR / f'{args.nc}_clusters_{num_feats}_feats_permutation_featimp.png')

    # Extra Trees Classifier 
    et = ExtraTreesClassifier(max_depth=8)
    et = et.fit(X, Y)
    print("Extra Tree Classifier Train score: ", et.score(X, Y))
    print("Extra Tree Classifier Subset score: ", et.score(X_subset, Y_subset))
    model = SelectFromModel(et, prefit=True)
    feature_mask = model.get_support()
    selected_feat_list = np.array(args.features)[feature_mask] 
    logger.info(f"Number of selected features by extra trees classifier: {len(selected_feat_list)}")
    logger.info(f"Selected features are: {selected_feat_list}")
    logger.info("*"*20)
    

    # Naive feature ranking 
    feat_var = []
    for idx in range(num_feats):
        var = np.var(clust_centers[:,idx])
        feat_var.append(var)
    featrank_idx = np.argsort(feat_var)
    naivefeatrank = [args.features[i] for i in featrank_idx]
    logger.info(f'Ranked features from simple cluster centroid analysis: {naivefeatrank}')
    logger.info("*"*20)
    '''
