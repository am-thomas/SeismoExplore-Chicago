# Computes the following features in all three components
# 1. STA/LTA using a 10-s Short Term (ST) Window and a 40-s preceding Long Term (LT) window
# 2. Skewness of 10-s Short Term (ST) Window
# 3. Kurtosis of 10-s Short Term (ST) Window
# and stores all features into a numpy array 
# 
# By default (and for our other study), ratio-based features (e.g. STK/LTK or the ratio 
# of the short-term window's kurtosis to that of the long-term window) are saved for all 
# three statistical moments (average, skewness, kurtosis), even though we don't use them 
# in our final model. We also compute stat features for all continous windows, not just 
# windows detected by the PSD misfit detector
# If applying this code for another station, we suggest modifying stlt_feats.py so that 
# these extra computations are not made 

from subprocess import run
import argparse
from constants import DATA_PATH, NUM_SECS_DAY
import numpy as np
import utils
from obspy import UTCDateTime

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--start_time", required=True, type=str,
                        help='start time of desired time series (ex. 2015-04-01T00:00:00)')
  parser.add_argument("--days", required=True, type=int)
  parser.add_argument("--exp_name", required=True, type=str,
                      help = 'Name of data folder that you want to save features to')
  parser.add_argument("--chan_list", nargs='+', default=['HHZ', 'HH1', 'HH2'], 
                      help='list of channel codes')
  args = parser.parse_args()

  # set path to experiment folder
  EXP_PATH = DATA_PATH / args.exp_name

  # set list of statistical features
  feat_list = ['avg_Zcomp_ratio', 'avg_H1comp_ratio', 'avg_H2comp_ratio', 
                'skw_Zcomp_stw', 'skw_H1comp_stw', 'skw_H2comp_stw',
                'kur_Zcomp_stw', 'kur_H1comp_stw', 'kur_H2comp_stw']
  num_feats = len(feat_list)
  print('Number of features:', num_feats )

  # compute statistical features for all components and store to numpy files
  print('Running...')
  for idx, chan in enumerate(args.chan_list):
    command = ["python", "stlt_feats.py", "--start_time", args.start_time,
                "--param_list", "avg", "skw", "kur", "--chan", chan, "--exp_name", args.exp_name,
                "--filt", 'None', "--days", str(args.days)]
    if idx == 0:
        command.append("--save_timespkl")
    print(' '.join(command))
    run(command)   

  # create a dictionary of row indices that map to each ST/LT feature type
  row_idx = {'ratio': 2,              # index to get ST/LT ratios 
            'stw': 1,                 # index to get short-term window (STW) values
            'ltw': 0}                 # index to get long-term window (LTW) values (not-relevant)

  # create a dictionary to specify the cut off indices for special days, which may have more data in one component
  # We cut all feature and times arrays so that all arrays are the same size
  special_days = {'2017-05-12': 23040,   
                  '2017-05-13': 31680}

  # combine all features into a simple file
  start_day = args.start_time[:10]
  date = start_day
  for i in range(args.days):
      # print('Retrieving features and times for', date)

      # check if date is a special day and store final index value
      if date in special_days.keys():
          special = True
          cutoff = special_days[date]
      else:
          special = False

      # load times for give day
      times_1day = np.array(utils.load_file(EXP_PATH / f'times_{date}.pkl'))

      # cut times array size as needed for special days
      if special:
          times_1day = times_1day[:cutoff]

      # load and stack feature values for given day
      for j, feat in enumerate(feat_list):

          # store type of STLT feature ('ratio', 'ltw', or 'stw')
          stlt_type= feat.split('_')[-1]

          # load array of appropriate feature
          feat_pfx = feat.replace(f'_{stlt_type}', '')
          feat_array = np.load(EXP_PATH / f'{feat_pfx}_{date}.npy')
          feat_array = feat_array[ row_idx[stlt_type] ]

          # cut feature array size as needed for special days
          if special:
              feat_array = feat_array[:cutoff]

          # stack different feature values column-wise
          if j == 0:
              X_1day = feat_array.reshape(-1, 1)
          else:
              X_1day = np.hstack((X_1day, feat_array.reshape(-1, 1)))

      # stack X and time arrays of all days together
      if i == 0:
          X = X_1day
          times = times_1day.reshape(-1,1)
      else:
          X = np.vstack((X, X_1day))
          times = np.vstack((times, times_1day.reshape(-1,1)))
      date = UTCDateTime(date + 'T00:00:00') + NUM_SECS_DAY
      date = str(date)[:10]

  # save stacked statistical feature and times array to numpy file in experiment folder
  print('Saving statistical feature (X) array and times array...')
  np.save(EXP_PATH / f'X_{num_feats}feats_{args.days}days_{start_day}.npy', X)
  utils.save_file(EXP_PATH / f'times_{args.days}days_{start_day}.pkl', times)
  print('X array shape:', np.shape(X))
  print('times array shape:', np.shape(times))