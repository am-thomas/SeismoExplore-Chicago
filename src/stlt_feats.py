# This file extracts, processes, and computes ST/LT-based features for continous data. 
# It is used within get_clustfeatures.py to compute all 10 features used in the clustering model
# 
# This file is taken directly from https://github.com/am-thomas/EQ-Chicago and thus has additional
# functionality (e.g. filtering before feature computation) than is used by SeismoExplore-Chicago 

import numpy as np
import utils_process
from scipy import stats
from obspy import read_inventory
from obspy import UTCDateTime
from constants import DATA_PATH 
import os 
import argparse
import json 
import utils

def get_stlt(data, samp_rate, start0='2015-04-01T00:00:00', counter=0, s_st=3, s_lt=30,
             dt_step = 0.5, param ='avg', return_time = True):
    '''
    Returns the desired statistical moment of a moving Short Term (ST) and Long Term (LT) window and
    their ratios (ST/LT)
    :param data: array or list of data points (e.g. trace.data attribute of an Obspy stream object)
    :param samp_rate: sampling rate (Hz) of seismic station data
    :param start0: starttime string corresponding to midnight UTC time of the day of interest (e.g. '2015-04-01T00:00:00'),
                    not used if return_time = False
    :param counter: starting index of the given time series, specifying the starttime of the given data.
                    (e.g. counter = 6000 --> utc_starttime = YYYY-MM-DDT00:01:00.0000Z for 100 Hz data)
                    not used if return_time = False
    :param s_st: duration (seconds) of Short Time (ST) window
    :param s_lt: duration (seconds) of Long Time (LT) window
    :param dt_step: time step (seconds) that each ST/LT window advances for next ST/LT computation
    :param param: statistical parameter to be computed, first parameter is for the time domain option and second
     is for the frequency domain ('_fd')
        - Average: 'avg', 'avg_fd'
        - Standard deviation: 'std', 'std_fd'
        - Skewness: 'skw', 'skw_fd'
        - Kurtosis: 'kur', 'kur_fd
    :return: arrays of ST, LT, ST/LT ratio values, and (optionally) times and an updated counter
    '''

    # store initial values
    n_st = int(s_st * samp_rate)                               # number of samples in ST window
    n_lt = int(s_lt * samp_rate)                               # number of samples in LT window
    s_data = len(data)/samp_rate                               # duration (s) of input data
    steps = int((s_data - s_lt - s_st) * (1 / dt_step)) + 1    # number of steps or total number of ratios
    n_ltsegs = int(s_lt/s_st)                                  # number of ST windows contained in one LT window
    ST = np.zeros(steps)                                       # empty array of ST values
    LT = np.zeros(steps)                                       # empty array of LT values
    STLTratios = np.zeros(steps)                               # empty array of ST/LT ratios
    dt = 1/samp_rate                                           # temporal resolution of given time data
    ts = []

    # loop through time series to compute all ST/LT ratios
    start = n_lt
    for i in range(steps):
        # gather data segment with ST + LT window
        data_seg = data[start-n_lt:start+n_st]
        # detrend ST+LT window if desired
        # data_seg = detrend(data_seg, type='constant')

        # if frequency-domain-based method, compute FFT of ST and LT windows
        if param.endswith('_fd'):
            # compute FFT amplitudes of ST window
            data_st = abs(np.fft.rfft(data_seg[n_lt:])) * dt

            # For LT window, compute FFTs of subsegments equal in duration to s_st and take average
            lt_segs = []
            start_2 = 0
            for j in range(n_ltsegs):
                data_ltseg = abs(np.fft.rfft(data_seg[start_2:start_2 + n_st]))* dt
                lt_segs.append(data_ltseg)
                start_2 = start_2 + n_st
            lt_segs = np.array(lt_segs)
            data_lt = np.mean(lt_segs, axis=0)
            # set first elements of amplitude distribution to zero
            data_st[0] = 0
            data_lt[0] = 0

        # store ST and LT data arrays for time-domain option
        else:
            data_st = data_seg[n_lt:]
            data_lt = data_seg[:n_lt]

        # compute desired moments in ST and LT window
        if param.startswith('avg'):
            ST[i] = np.mean(abs(data_st))
            LT[i] = np.mean(abs(data_lt))
        elif param.startswith('kur'):
            ST[i] = stats.kurtosis(data_st, nan_policy='propagate', fisher=False)
            LT[i] = stats.kurtosis(data_lt, nan_policy='propagate', fisher=False)
        elif param.startswith('std'):
            ST[i] = np.std(data_st)
            LT[i] = np.std(data_lt)
        elif param.startswith('skw'):
            ST[i] = stats.skew(data_st, nan_policy='propagate')
            LT[i] = stats.skew(data_lt, nan_policy='propagate')
        elif param.startswith('max'):
            ST[i] = np.max(abs(data_st))
            LT[i] = np.max(abs(data_lt))

        # compute and store ratios
        STLTratios[i] = ST[i]/LT[i]
        start = start + int(dt_step*samp_rate)

        # update counter and time
        if return_time:
            utc_time = UTCDateTime(start0) + (counter * dt_step)
            counter += 1
            ts.append(utc_time)

    if return_time:
        return ST, LT, STLTratios, ts, counter
    else:
        return ST, LT, STLTratios


def get_freqparam(filtband):
    # Returns the minimum and maximinum frequency to input for bandpass filtering
    if filtband == 'filtb1':
        freqmin, freqmax = 0.01, 0.05
    elif filtband == 'filtb2':
        freqmin, freqmax = 0.05, 0.5
    elif filtband == 'filtb3':
        freqmin, freqmax = 0.5, 2.0
    elif filtband == 'filtb4':
        freqmin, freqmax = 2.0, 8.0
    elif filtband == 'filtb5':
        freqmin, freqmax = 8.0, 20.0
    elif filtband == 'filt8pole':
        freqmin, freqmax = 11.5, 13.5
    elif filtband == 'filt6pole':
        freqmin, freqmax = 15.5, 17.5
    elif filtband == 'filt4pole':
        freqmin, freqmax = 24, 26
    return freqmin, freqmax 


def compute_features(args): 
    # set duration of segment to process. By default, taper_s == 0 s and duration_ic is two hours
    duration_icext = args.duration_ic + args.s_st + args.s_lt - args.dt_step + (2*args.taper_s)     # duration in seconds
    npts_icext = int(duration_icext * args.samp_rate)                                               # duration in samples

    # set fraction of extracted time series to taper, if needed
    taper_frac = args.taper_s / duration_icext

    # set start time and data path
    starttime_str_0 = args.start_time 
    EXP_PATH = DATA_PATH / args.exp_name 

    # Make sure the stlt directory exists, else create it 
    os.makedirs(EXP_PATH, exist_ok=True)
    # Save parameter info 
    with open(EXP_PATH / 'info.txt', 'a+') as f:
        json.dump(args.__dict__, f, indent=2)

    # store inventories and pref_filter frequencies if response correction is desired
    if args.respcor:
        # store inv object for desired channel
        inv = read_inventory(f'../metadata/{args.sta}inventory_{args.chan}.xml')

        # set prefilter frequencies for response correction
        if args.samp_rate == 100:
            pre_filt = [0.004, 0.006, 35, 50]
        elif args.samp_rate == 40:
            pre_filt = [0.005, 0.023, 16, 19.99]

    if args.stage0_div:
        if args.sta == 'HQIL' and args.net == 'NW' :
            stage0_sensitivity = 9.43695e8
        elif args.sta == 'L44A' and args.net == 'TA':
            stage0_sensitivity = 627192000.0
        else:
            raise ValueError('Division by stage zero sensitivities only capable for station NW.HQIL')

    
    # store frequency parameters for bandpass filtering, if needed
    if args.filt != 'None' and not args.filt.startswith('cusfilt'):
        freqmin, freqmax = get_freqparam(args.filt)
        print(f'Filtering data at {freqmin} - {freqmax} Hz')
    elif args.filt != 'None' and args.filt.startswith('cusfilt'):
        freqmin = args.freqmin
        freqmax = args.freqmax
        print(f'Custom filtering data at {freqmin} - {freqmax} Hz')

    # loop through parameters and days to save files of ST, LT, and ST/LT ratios
    j_it = int(args.duration / args.duration_ic)                              # number of iterations in for loop for each day
    taper_idx = int(args.taper_s * args.samp_rate)                            # index of data segment when un-tapered segment begins
    for param_index, param in enumerate(args.param_list):
        starttime_str_ext = str(UTCDateTime(starttime_str_0)-args.s_lt-args.taper_s)
        counter = 0
        # loop through duration of data at a time to save ratios to file
        for i in range(args.days):
            times = []
            print(f"Computing {param} for day {i}")
            starttime = str(UTCDateTime(starttime_str_ext) + args.s_lt + args.taper_s)

            # store empy arrays for features and time variables for one day
            sts = np.array([])
            lts = np.array([])
            ratios = np.array([])
            ts = np.array([])

            # loop through duration_ic of data per day to process and store ratios
            for j in range(j_it):
                # data retrieval and processing: get waveform, resample, linear detrend, remove response, cut tapered
                # portions, and filter as desired.
                try:
                    st = utils_process.get_rawdata(args.net, args.sta, args.loc, args.chan, starttime_str_ext, duration_icext,
                    args.samp_rate, plot_wave=False, save=False)

                # continue to next iteration if data could not be retrieved properly
                except:
                    starttime_str_ext = UTCDateTime(starttime_str_ext) + (args.duration_ic)
                    starttime_str_ext = str(starttime_str_ext)
                    if args.samp_rate == 100:
                        counter = counter + 2880
                    else:
                        raise ValueError(f'Program does not currently accommodate {args.samp_rate} Hz data with gaps')
                    continue

                # continue if stream doesn't contain traces, incomplete traces, or traces with masked values
                if len(st) == 0 or len(st[0]) < npts_icext or np.ma.is_masked(st[0].data):
                    starttime_str_ext = UTCDateTime(starttime_str_ext) + (args.duration_ic)
                    starttime_str_ext = str(starttime_str_ext)
                    if args.samp_rate == 100:
                        counter = counter + 2880
                    else:
                        raise ValueError(f'Program does not currently accommodate {args.samp_rate} Hz data with gaps')
                    continue

                # fill gaps if needed
                get_gaps = st.get_gaps()
                if len(get_gaps) != 0:
                    print('Gaps present. Merging traces using interpolation...')
                    st.merge(fill_value='interpolate')

                # downsample if needed
                if st[0].stats.sampling_rate > args.samp_rate:
                    st.decimate(int(st[0].stats.sampling_rate/args.samp_rate))

                # linear detrend
                st.detrend('linear')

                # response correction if desired
                if args.respcor:
                    st.remove_response(inventory=inv, output=args.units, water_level=None, pre_filt=pre_filt,
                                       zero_mean=True, taper=True, taper_fraction=taper_frac, plot=False, fig=None)
                    st.detrend('linear')

                # filtering
                if args.filt != 'None':
                    st.taper(max_percentage=taper_frac, type='cosine')
                    st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=True)

                # cut to desired length of data and cut off tapered portions
                data = st[0].data[:npts_icext]
                data = data[taper_idx:-taper_idx]

                # by default for HQIL, divide by stage zero sensitivities to convert from raw data to velocity units
                if args.stage0_div:
                    data = data / stage0_sensitivity
                    
                # Append to a list of ST, LT, and ST/LT ratios for each day
                sts_ap, lts_ap, ratios_ap, ts, counter = get_stlt(data, args.samp_rate, dt_step=args.dt_step, s_st=args.s_st, s_lt=args.s_lt, 
                                                    param=param, start0=args.start_time, counter=counter)

                sts = np.append(sts,sts_ap)
                lts = np.append(lts, lts_ap)
                ratios = np.append(ratios, ratios_ap)
                
                times.extend(ts)   
                starttime_str_ext = UTCDateTime(starttime_str_ext) + (args.duration_ic)
                starttime_str_ext = str(starttime_str_ext)

            # continue to next day if no features were retrieved
            if len(ts) == 0:
                continue

            # print('Number of ratios', len(ratios))
            
            # set name of file to save computed features for each day
            if args.chan in ['HHZ','BHZ']:
                filename_pre = f'{param}_Zcomp'
            elif args.chan in ['HH1', 'BHN', 'HHN']:     #HH1 in HQIL is mostly aligned in North direction
                filename_pre = f'{param}_H1comp'
            elif args.chan in ['HH2', 'BHE', 'HHE']:     #HH2 in HQIL is mostly aligned in East direction
                filename_pre = f'{param}_H2comp'
            if args.filt != 'None':
                filename_pre = filename_pre + '_' + args.filt
            rfilename = EXP_PATH / f'{filename_pre}_{starttime[0:10]}'

            # save text files of data if desired
            if args.savetxt:
                with open(rfilename.with_suffix('.txt'), "w+") as f:
                    for i in range(len(sts)):
                        f.write(str(times[i]) + '\t' + str(sts[i]) + '\t' + str(lts[i]) + '\t' + str(ratios[i]) + '\n')

            # save npy files of data
            np.save(rfilename.with_suffix('.npy'), np.array([sts, lts, ratios]))

            # only save times pickle file once (for one parameter)
            if param_index == 0 and args.save_timespkl:
                utils.save_file(EXP_PATH / f'times_{starttime[0:10]}.pkl', times)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Station Parameters 
    parser.add_argument("--net", default='NW', type=str, help='network code')
    parser.add_argument("--sta", default='HQIL', type=str, help='station code') 
    parser.add_argument("--loc", default='00', type=str, help='location code')
    parser.add_argument("--chan", default='HHZ', type=str, help='channel code')
    parser.add_argument("--samp_rate", default=100, type=int, help='station sampling rate (Hz)')

    # Time and algorithm parameters
    parser.add_argument("--start_time", default='2015-04-01T00:00:00', type=str,
                        help='start time of desired time series (midnight of UTC time)')
    parser.add_argument("--s_st", default=10, type=float, help='duration (s) of Short Time (ST) window')
    parser.add_argument("--s_lt",default=40, type=float, help=' duration (s) of Long Time (ST) window')
    parser.add_argument("--dt_step", default=5, type=float, help='time step (s) to move SST/LT windows')
    parser.add_argument("--taper_s", default=2.5, type=float, help='duration (s) of window to taper before filering.')
    parser.add_argument("--duration", default=60 * 60 * 24, type=float, help='duration (usually 24 h) of data to save to file')
    parser.add_argument("--duration_ic", default= 60 * 60 * 2, type=float, help='duration of data to detrend at a time')
    parser.add_argument("--days", default=7, type=int)
    parser.add_argument("--filt", default='None', type=str, help='''Frequency band to filter seismic waveforms using a Fourth Order Bandpass filter.
                        # 0.01 - 0.05 Hz band, characteristic of surface waves (filt = 'filtb1')
                        # 0.05 - 0.5 Hz band, characteristic of microseismic noise (filt = 'filtb2')
                        # 0.5 - 2.0 Hz band, characteristic low-noise band (filt = 'filtb3')
                        # 2.0 - 8.0 Hz band. characteristic of local quakes (filt = 'filtb4')
                        # 8.0 - 20.0 Hz band, cultural noise (filt = 'filtb5')
                        # 11.5 - 13.5 Hz, narrow band centered on dominant freq of 8 pole motor (filt = 'filt8pole')
                        # 15.5 - 17.5 Hz, narrow band centered on dominant freq of 6 pole motor (filt = 'filt6pole')
                        # 24.0 - 26.9 Hz, narrow band centered on dominant freq of 4 pole motor (filt = 'filt4pole')
                        # custom band (filt = 'cusfilt#' where # can be any number)
                        ''')
    parser.add_argument("--freqmin", default=15, type=float, help='minimum frequency for custom bandpass filtering')
    parser.add_argument("--freqmax", default=18, type=float, help='maximum frequency for custom bandpass filtering')
    parser.add_argument("-p", '--param_list',  nargs='+', required=True,
                        help="""list of parameters to compute 
                        List of parameters which can be computed
                        # Average of the time domain amplitude distribution (param = 'avg')
                        # Average of the frequency domain amplitude distribution (param = 'avg_fd')
                        # Standard deviation of the time domain amplitude distribution (param = 'std')*
                        # Standard deviation of the frequency domain amplitude distribution (param = 'std_fd')*
                        # Skewness of the time domain amplitude distribution (param = 'skw')
                        # Skewness of the frequency domain amplitude distribution (param = 'skw_fd')
                        # Kurtosis of the time domain amplitude distribution (param = 'kur')
                        # Kurtosis of the frequency domain amplitude distribution (param = 'kur_fd')
                        """)

    # Parameters if instrument response correction is desired
    parser.add_argument("--stage0_div", default=True, action=argparse.BooleanOptionalAction,
                        help='True if dividing data by stage zero sensitivity to convert to velocity units. Only applicable for HQIL and ')
    parser.add_argument("--respcor", default=False, action=argparse.BooleanOptionalAction,
                        help='True if data should be corrected for instrument response using response file' )
    parser.add_argument("--units", default='VEL', type=str,
                        help='''
                        output to apply when correcting for instrument response. "counts" for counts, "DISP" for displacement, 
                        "VEL" for velocity, or "ACC" for acceleration) 
                        ''')

    # Saving parameters
    parser.add_argument("--exp_name", default='stlt', type=str)
    parser.add_argument("--savetxt", default=False, action=argparse.BooleanOptionalAction,
                        help='True of you wish to save a a textfile of features and times')
    parser.add_argument("--save_timespkl", default=False, action=argparse.BooleanOptionalAction,
                        help='True if you wish to save a times pickle file. False if otherwise')

    args = parser.parse_args()


    compute_features(args)

