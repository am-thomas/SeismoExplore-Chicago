# Visualizes 20-s windows (10-s detection with preceding 10-s 'noise' window) of events in a given
# event type. 
#
# Note that not all windows will share the event type characteristics. Since we detect events from overlapping
# (50%) 10-s windows, multiple windows may contain the same event. The onset/end of a window may not be 
# captured within the 10-s window. 

import argparse
from constants import DATA_PATH
import os
import utils_process 
import pandas as pd
import numpy as np
import utils
from obspy import UTCDateTime


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # Station Parameters
    parser.add_argument("--net", default='NW', type=str, help='network code')
    parser.add_argument("--sta", default='HQIL', type=str, help='station code')
    parser.add_argument("--loc", default='00', type=str, help='location code')
    parser.add_argument("--samp_rate", default=100, type=int, help='station sampling rate (Hz)')

    # Clustering related params 
    parser.add_argument("--nc", default=12, type=int, help="Number of clusters")
    parser.add_argument("--num_feat", default=10, type=int, help="Number of features used")
    parser.add_argument("--clust_dir", default='clusters_alldata', type=str, help='name of folder with clustering results')
    parser.add_argument('--eventtype', type=str, default='D', 
                        help='Event type (A-D) to analyze')
    parser.add_argument('--savedcsv', default=False, action=argparse.BooleanOptionalAction, 
                        help="Pass no-savedcsv is there is no csv of start times of a desired event type. A new csv will be created")
    args = parser.parse_args()

    # save and create results/plot directory
    CLUST_DIR = DATA_PATH / args.clust_dir
    PLOT_DIR = CLUST_DIR / f'figs_eventtype_{args.eventtype}'
    os.makedirs(PLOT_DIR, exist_ok=True)

    if not args.savedcsv:
        # load cluster times and labels
        df_times = utils.load_file(CLUST_DIR / f'{args.nc}_clusters_{args.num_feat}_feats_times.pkl')
        df_labels = np.load(CLUST_DIR / f'{args.nc}_clusters_{args.num_feat}_feats_labels.npy')

        # set label list for each 
        labels_dict = {'A':[0,2,5,7,10], 'B':[11], 'C':[1,4,6,8,9], 'D':[3]}
        label_list = labels_dict[args.eventtype]

        mask = np.isin(df_labels, label_list)
        cut_times = df_times[mask]
        # np.random.shuffle(cut_times)

        # save to csv
        df_dict = {'start_time': cut_times}
        df = pd.DataFrame(df_dict)
        df.to_csv(CLUST_DIR/ f'eventtype_{args.eventtype}.csv', index=False)


    df = pd.read_csv(CLUST_DIR/ f'eventtype_{args.eventtype}.csv')
    duration = 20
    for i, time in enumerate(df['start_time']):
        print(i, time)
        starttime_str = str(UTCDateTime(time) - 10)
        for chan in ['HHZ', 'HH1', 'HH2']:
            st = utils_process.get_rawdata(args.net, args.sta, args.loc, chan, starttime_str, duration,
                args.samp_rate, plot_wave=False, save=False)

            if chan == 'HHZ':
                st_3c = st
            else:
                st_3c = st_3c + st

        st_3c.detrend('linear')
        filename = PLOT_DIR / f'{i}.png'
        st_3c.plot(outfile=filename)

