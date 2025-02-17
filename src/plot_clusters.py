import argparse
from constants import DATA_PATH
import os
import utils_process
from obspy import UTCDateTime, clients
import matplotlib.pyplot as plt 
import seaborn as sns  
import pandas as pd

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # Station Parameters
    parser.add_argument("--net", default='NW', type=str, help='network code')
    parser.add_argument("--sta", default='HQIL', type=str, help='station code')
    parser.add_argument("--loc", default='00', type=str, help='location code')
    parser.add_argument('-l','--chan_list', nargs='+', default=['HHZ','HH1','HH2'], help='list of channel codes')
    parser.add_argument("--samp_rate", default=100, type=int, help='station sampling rate (Hz)')

    # Clustering related params 
    parser.add_argument("--nc", default=12, type=int, help="Number of clusters")
    parser.add_argument("--num_feat", default=10, type=int, help="Number of features used")
    parser.add_argument("--clust_dir", default='clustering_test', type=str, help='name of folder with clustering results')
    parser.add_argument('--saveplot', default=True, action=argparse.BooleanOptionalAction, help="Pass --no-saveplot to prevent saving")
    parser.add_argument('--label_list', nargs='+', default='all', 
                        help='list of clusters to plot. default is to plot all clusters')
    args = parser.parse_args()

    # save and create results/plot directory
    CLUST_DIR = DATA_PATH / args.clust_dir
    PLOT_DIR = CLUST_DIR / f'figs_{args.nc}clusters_{args.num_feat}feats'
    os.makedirs(PLOT_DIR, exist_ok=True)

    # load csv with clustering results
    df_signals = pd.read_csv(CLUST_DIR / f'{args.nc}_clusters_{args.num_feat}_feats_df_clust.csv')
    points_of_interest = df_signals[['Index', 'Time', 'Label']].to_numpy()

    # plotting settings 
    sns.set_theme(style='whitegrid')
    plt.rcParams['savefig.dpi'] = 300
    duration = 80 

    # if specific clusters are requested, convert argument labels to integers
    if args.label_list != 'all':
        label_list = [int(val) for val in args.label_list]
    else:
        label_list = args.label_list

    # plot 40 seconds of waveform data, before and after point of interest
    for index, utc_time, clust_label in points_of_interest:
        if args.label_list != 'all' and clust_label not in label_list:
            continue
        print(index, utc_time, clust_label)
        start = str(UTCDateTime(utc_time)-(duration/2))
        for i in range(3):
            try:
                st = utils_process.get_rawdata(args.net, args.sta, args.loc, args.chan_list[i], start, duration, args.samp_rate, plot_wave=False, save = False )
            except clients.fdsn.header.FDSNNoDataException as error:
                print('Could not get data. Skipping to next event...')
                break
            st.detrend('linear')

            if i == 0:
                st_3c = st
            else:
                st_3c = st_3c + st

        if args.saveplot:
            os.makedirs(PLOT_DIR / str(clust_label), exist_ok=True)
            filename =  PLOT_DIR / str(clust_label) / f'{args.nc}_clusters_Clust_No_{clust_label}_{args.num_feat}_feats_{index}_d{duration}.png'  
            # Plot the channels 
            st_3c.plot(outfile=filename)
        else:
            st_3c.plot()
    