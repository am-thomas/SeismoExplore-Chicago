# Creates a csv with "detected" windows (10-s windows with a PSD misfit
# that exceeds a given threshold) and all 10 model features

import numpy as np
import argparse
from constants import DATA_PATH, NUM_SECS_DAY
from obspy import UTCDateTime
import pandas as pd
import utils
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Station Parameters
    parser.add_argument("--net", default='NW', type=str, help='network code')
    parser.add_argument("--sta", default='HQIL', type=str, help='station code')
    parser.add_argument("--loc", default='00', type=str, help='location code')
    parser.add_argument('-l', '--chan_list', nargs='+', default=['HHZ', 'HH1', 'HH2'], help='list of channel codes')
    parser.add_argument("--samp_rate", default=100, type=int, help='station sampling rate (Hz)')
    parser.add_argument("--units", default='VEL', type=str,
                        help='output (displacement, velocity, or acceleration) to apply when correcting for instrument response')

    # Experiment parameters
    parser.add_argument("--threshold", type=float, default = 1.0,
                        help='Threshold misfit value, above which to store events. As long as one components PSD mf is above threshold, the event will be included')
    parser.add_argument("--exp_name", type=str, required=True, help = 'name of folder containing feature files')
    parser.add_argument("--days", default=7, type=int)
    parser.add_argument("--start_day", type=str, required=True,
                        help= 'first day (e.g. "2015-04-01") to start computing and storing PSD misfits')
    parser.add_argument("--skipdays", nargs='+', default = ['2019-05-17','2019-05-18','2019-05-19'],
                        help='List of days with no data to skip' )
    args = parser.parse_args()

    # store experiment path
    EXP_PATH = DATA_PATH / args.exp_name

    # loop through all PSD misfits and keep only data for windows with misfits greater than the desired threshold
    # retrieve time-based features (hour, day of week) in the same loop
    # note that only one chnnael's PSD misfit needs to exceed the threshold to be added to the subset
    cuts_dict = {'start_time':[],'PSDmf_Zcomp':[],'PSDmf_H1comp':[], 'PSDmf_H2comp':[], 'hour':[], 'day_of_week':[]}
    day_i = args.start_day
    for i in range(args.days):
        # skip days with no data
        if day_i in args.skipdays:
            day_i = UTCDateTime(day_i) + NUM_SECS_DAY
            day_i = str(day_i)[:10]
            continue

        PSDmf_Zcomp = np.loadtxt(EXP_PATH / f'PSDmf_HHZ_{day_i}_ACC.txt',dtype=str)[:,1]
        PSDmf_H1comp = np.loadtxt(EXP_PATH / f'PSDmf_HH1_{day_i}_ACC.txt',dtype=str)[:,1]
        PSDmf_H2comp = np.loadtxt(EXP_PATH / f'PSDmf_HH2_{day_i}_ACC.txt',dtype=str)[:,1]
        PSDmf_Zcomp = [float(val) for val in PSDmf_Zcomp]
        PSDmf_H1comp = [float(val) for val in PSDmf_H1comp]
        PSDmf_H2comp = [float(val) for val in PSDmf_H2comp]
        all_times = np.loadtxt(EXP_PATH / f'PSDmf_HHZ_{day_i}_ACC.txt',dtype=str)[:,0]

        if not (len(PSDmf_Zcomp) == len(PSDmf_H1comp) == len(PSDmf_H2comp)):
            raise ValueError('Lengths of the three PSD misfit files for day does not match', day_i)
        
        for j, PSDmf_Z in enumerate(PSDmf_Zcomp):
            if PSDmf_Z > args.threshold or PSDmf_H1comp[j] > args.threshold or PSDmf_H2comp[j] > args.threshold:
                time = str(all_times[j])
                cuts_dict['start_time'].append(time)
                cuts_dict['PSDmf_Zcomp'].append(PSDmf_Z)
                cuts_dict['PSDmf_H1comp'].append(PSDmf_H1comp[j])
                cuts_dict['PSDmf_H2comp'].append(PSDmf_H2comp[j])

                #convert utc time to local time
                ctime = utils.utc_to_local(time[:19], 'US/Central')
                cuts_dict['hour'].append(ctime.hour)
                cuts_dict['day_of_week'].append(ctime.isoweekday())

        # update day
        day_i = UTCDateTime(day_i) + NUM_SECS_DAY
        day_i = str(day_i)[:10]

    # create dataframe of "detected windows" with their PSD misfit values and time-based features
    df_PSDcuts = pd.DataFrame(cuts_dict)

    # create a dataframe of statistical features
    stat_features = ['avg_Zcomp_ratio', 'avg_H1comp_ratio', 'avg_H2comp_ratio', 
                'skw_Zcomp_stw', 'skw_H1comp_stw', 'skw_H2comp_stw',
                'kur_Zcomp_stw', 'kur_H1comp_stw', 'kur_H2comp_stw']
    for file in os.listdir(EXP_PATH):
        if file.startswith('X_9feats') and file.endswith('.npy'):
            statfeat_data = np.load(EXP_PATH/ file)
        elif file.startswith('times') and file.endswith('.pkl') and 'days' in file:
            statfeat_times = utils.load_file(EXP_PATH / file).flatten()
    df_statfeats = pd.DataFrame(statfeat_data, columns=stat_features)
    df_statfeats['start_time'] = statfeat_times

    # convert start_time columns to string and "inner" merge the two dataframes by shared times
    df_statfeats['start_time'] = df_statfeats['start_time'].astype(str)
    # df_PSDcuts['start_time'] = df_PSDcuts['start_time'].astype(str)
    merged_df = pd.merge(df_PSDcuts, df_statfeats, on='start_time', how='inner')

    # save as csv
    merged_df.to_csv(EXP_PATH / 'PSDcuts_allfeatures.csv', index=False)

