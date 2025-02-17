from constants import DATA_PATH
import argparse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import os 
import logging
from datetime import datetime

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--clust_dir", default='clustering_test', type=str, help='name of folder to save results to')
    parser.add_argument("-d", "--load_expdirs", required=True,  nargs='+', type=str,help='list of experiment folders (with feature values) to parse through')
    parser.add_argument("--features", nargs='+', default = 'all', 
                        help='List of feature names to use for clustering. Default is to use all available features')
    parser.add_argument("--kstart", default=2, type=int, help="Start value of k (no. of clusters)")
    parser.add_argument("--kend", default=20, type=int, help="End value of k (no. of clusters)") 
    args = parser.parse_args()

    # save and create clustering directory 
    CLUST_DIR = DATA_PATH / args.clust_dir
    os.makedirs(CLUST_DIR, exist_ok=True) 
    num_feats = len(args.features)

    # Plotting settings 
    sns.set_theme(style='whitegrid')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'
    
    # create log file to store results
    logging.basicConfig(level=logging.INFO, filename=str(CLUST_DIR / 'selectk_info.log'), format='%(message)s')
    logger = logging.getLogger()
    time_stamp = datetime.now().strftime("%H:%M:%S")
    logger.info("*" * 20)
    logger.info(f"Time Stamp: {time_stamp}")

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
    print(len(X))

    inertias = []
    distortions = []
    sil_scores = []

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    fig2, ax2 = plt.subplots() 
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    k_list = list(range(args.kstart, args.kend+1)) 
    test = []

    for k in k_list: 
        print(f"Running k-means for k: {k}")
        kmeans = KMeans(n_clusters=k,n_init='auto').fit(X)
        inertias.append(kmeans.inertia_)
        distortions.append(kmeans.inertia_/X.shape[0])

        """
        # Manual inertia and distortion calculation for reference: 
        labels = kmeans.labels_ 
        cluster_centers = kmeans.cluster_centers_
        inertia_sub = 0
        distortion_sub = 0
        for label in np.unique(labels): 
            indices = np.where(labels == label)[0]
            X_sub = X[indices]
            inertia_sub += np.sum(np.square(np.linalg.norm(X_sub - cluster_centers[label, :], axis=1)))  

        distortion_sub = inertia_sub/X.shape[0]

        # Alternate method to calculate distortion 
        # distortions.append(sum(np.square(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))) / X.shape[0])
        """

        # Silhouette score needs > 1 cluster 
        # print(k)
        if k > 1: 
            sil_scores.append(silhouette_score(X, kmeans.labels_, n_jobs=-1))


    ax[0].plot(k_list, inertias)
    ax[0].set_xlabel('K')
    ax[0].set_ylabel('Inertia')
    # ax[1].set_xticks(k_list)

    ax[1].plot(k_list, distortions)
    ax[1].set_xlabel('K')
    ax[1].set_ylabel('Distortion')
    fig.tight_layout()
    fig.savefig(CLUST_DIR / 'elbow_plot.png')

    ax2.plot(k_list, sil_scores) 
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    fig2.savefig(CLUST_DIR / 'sil_plot.png')

    logger.info(f"Inertias: {inertias}")
    logger.info(f"Distortions: {distortions}")
    logger.info(f"Silhouette Score: {sil_scores}")
    




