# Applies a k-means clustering algorithm (either pre-trained or newly trained) on a desired time period. 

from constants import DATA_PATH
import argparse
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans 
import os
import logging
from datetime import datetime
import utils


if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--nc", default=12, type=int, help="Number of clusters")
    parser.add_argument("-d", "--load_expdirs", required=True,  nargs='+', type=str,help='list of experiment folders to parse through')
    parser.add_argument("--clust_dir", default='clustering_test', type=str, help='name of folder to save clustering results to')
    parser.add_argument("--features", nargs='+', default = 'final_model', 
                        help='List of feature names to use for clustering. Default is to use the 10 final model feature used in the paper')
    parser.add_argument("--load_savedmodel", default=False, action=argparse.BooleanOptionalAction, 
                    help='Flag to apply pretrained K means model')
    parser.add_argument("--save_model", default=False, action=argparse.BooleanOptionalAction, 
                    help='Flag to save trained model')
    args = parser.parse_args()

    # make directory to save clustering results if it does not exist
    CLUST_DIR = DATA_PATH  / args.clust_dir
    os.makedirs(CLUST_DIR, exist_ok=True)

    # create log file to store results
    logging.basicConfig(level=logging.INFO, filename=str(CLUST_DIR / 'info.log'), format='%(message)s')
    logger = logging.getLogger()
    time_stamp = datetime.now().strftime("%H:%M:%S")
    logger.info("*" * 20)
    logger.info(f"Time Stamp: {time_stamp}")

    # load and combine feature (X) and times arrays
    for i, load_dir in enumerate(args.load_expdirs):
        df = pd.read_csv(DATA_PATH / load_dir / 'PSDcuts_allfeatures.csv')
        times_dir = df['start_time'].to_numpy()

        if args.features == 'final_model':
            final_features = ['avg_Zcomp_ratio', 'avg_H2comp_ratio', 'PSDmf_Zcomp', 'PSDmf_H2comp',
                        'kur_H1comp_stw', 'skw_Zcomp_stw', 'skw_H1comp_stw', 'skw_H2comp_stw',
                        'hour', 'day_of_week']
            X_dir = df[final_features]
        else:
            X_dir = df[args.features]
        X_dir = X_dir.values

        if i==0: 
            X = X_dir.copy()
            times = times_dir.copy()  
        else: 
            X = np.concatenate((X, X_dir), axis=0)
            times = np.concatenate((times, times_dir), axis=0)

    # store number of features
    num_feat = np.shape(X)[1]
    logger.info(f"Number features used: {num_feat}")

    # K-means cluster events using default parameters
    if args.load_savedmodel:
        kmeans = utils.load_file(DATA_PATH / 'Kmeans_finalmodel.pkl')
        labels = kmeans.predict(X)
    else:
        kmeans = KMeans(n_clusters=args.nc, verbose=1, n_init='auto').fit(X)
        labels = kmeans.labels_ 
        if args.save_model:
            utils.save_file(DATA_PATH / 'Kmeans_model.pkl', kmeans)
    unique, counts = np.unique(labels, return_counts=True)
    clust_dist = dict(zip(unique, counts))
    print(f"Cluster Distribution: {clust_dist}")
    logger.info(f"Distribution of points in cluster: {clust_dist}")

    # get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # save labels and cluster centers
    np.save(CLUST_DIR / f'{args.nc}_clusters_{num_feat}_feats_labels.npy', labels)
    utils.save_file(CLUST_DIR / f'{args.nc}_clusters_{num_feat}_feats_times.pkl', times)
    np.savetxt(CLUST_DIR / f'{args.nc}_clusters_{num_feat}_feats_clust_centers.txt', cluster_centers, delimiter='\t')

