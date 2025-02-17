# Computes and saves the misfit between the PSD of a given window (usually 10 s) and a dynamic
# background PSD. The backrgound PSDs were computed using backrgound_PSDs.py, which processes PSD segments
# over 50 random days in the study period

import numpy as np
import utils_process
import matplotlib.pyplot as plt
from obspy import read_inventory, UTCDateTime
from obspy.clients.fdsn import Client, header
client = Client('IRIS')
from constants import DATA_PATH, METADATA_PATH, NUM_SECS_HOUR, NUM_SECS_DAY
import os 
import argparse
import json
from utils import get_dst_dates
from utils_process import classicsmooth


def get_PSDmf(data, samp_rate, smoothpsddict, psdstddict, counter_marks, dst_info, start0, counter=0, wlen_s=10, dt_step = 2.5):
    # Returns the power spectral misfit between a given piece of data and the "background noise" sepctra 

    # store initial values
    night_start, morn_start, aft_start, eve_start = counter_marks     # counter vals that correspond to the start of night, morning, afternoon, and evening times
    npts = int(wlen_s * samp_rate)                                    # number of samples in window
    s_data = len(data)/samp_rate                                      # duration (s) of input data
    steps = int((s_data - wlen_s) * (1 / dt_step))                    # number of steps or total number of ratios
    misfits_dyn = np.zeros(steps)                                     # PSD misfits relative to dynamic window
    ts = []

    # loop through time series to compute all PSD misfits
    start = 0
    utc_time = str(UTCDateTime(start0) + (counter * dt_step))
    for i in range(steps):
        # gather data segment with the desired window and compute PSD
        data_seg = data[start:start+npts]
        freq, psd = utils_process.my_PSD(data_seg, args.chan, 10, samp_rate=100, dt_sgmt=10, dt_subgmt=2.5, ovlap_sgmt=0.50,
                                 n_subsgmts=13, ovlap_subsgmt=0.75, tpr_alpha=0.20, tpr_factor=1.142857,
                                 octavefrac=1.0 / 8.0, resp_correct=False, smooth=False)

        # accomadate for daylight saving days
        if dst_info['check']:
            multiplier = dst_info['mul']
            if utc_time[:19] == f"{dst_info['start']}T08:00:00":
                print('Updating counter markers for daylight savings...')
                night_start, morn_start, aft_start, eve_start = 5*multiplier, 11*multiplier, 17*multiplier, 23*multiplier
            elif utc_time[:19] == f"{dst_info['end']}T07:00:00":
                print('Updating counter markers for daylight savings...')
                night_start, morn_start, aft_start, eve_start = 6*multiplier, 12*multiplier, 18*multiplier, 24*multiplier

        # get appropriate background noise psd
        if counter >= night_start and counter < morn_start:
            background_psd_dyn = smoothpsddict['Night']
            background_std_dyn = psdstddict['Night']
        elif counter >= morn_start and counter < aft_start:
            background_psd_dyn = smoothpsddict['Morning']
            background_std_dyn = psdstddict['Morning']
        elif counter >= aft_start and counter < eve_start:
            background_psd_dyn = smoothpsddict['Afternoon']
            background_std_dyn = psdstddict['Afternoon']
        else:
            background_psd_dyn = smoothpsddict['Evening']
            background_std_dyn = psdstddict['Evening']

        # compute misfit compared to dynamic window
        # plt.plot(psd)
        # plt.plot(background_psd_dyn)
        # plt.show()
        sq_diff_dyn = (psd - background_psd_dyn) / background_std_dyn
        sq_diff_dyn = np.where(sq_diff_dyn<=1, 0, sq_diff_dyn)
        misfits_dyn[i] = np.mean(sq_diff_dyn)

        # update start indices, times, counter, and save times to list
        start = start + int(dt_step*samp_rate)
        utc_time = UTCDateTime(start0) + (counter * dt_step)
        ts.append(utc_time)
        utc_time = str(utc_time)
        counter += 1

    counter_marks = [night_start, morn_start, aft_start, eve_start]
    return misfits_dyn, ts, counter, counter_marks


def compute_PSDfeatures(args, smoothpsddict, psdstddict):
    EXP_PATH = DATA_PATH / args.exp_name

    # duration of segment to correct for instrument response (2 hours)
    duration_ic = 60 * 60 * 2
    duration_icext =  duration_ic + args.wlen_s + (2*args.taper_s)  # process in segments of 2 hours + duration for window and tapering
    npts_durext = int(duration_icext*args.samp_rate)
    # fraction of extracted time series to taper
    taper_frac = args.taper_s / duration_icext

    # Make sure the data directory exists, else create it
    os.makedirs(EXP_PATH, exist_ok=True)
    # Save parameter info 
    with open(EXP_PATH / 'info.txt', 'a+') as f:
        json.dump(args.__dict__, f, indent=2)

    # store inventory for response correction
    inv = read_inventory(METADATA_PATH / f'{args.sta}.xml')
    # set prefilter frequencies for response correction
    pre_filt = [0.004, 0.006, 35, 50]

    # get list of days to compute PSD misfits
    days_list = [args.start_time[:10]]
    dtobj = UTCDateTime(args.start_time)
    for k in range(1,args.days):
        days_list.append( str(dtobj + (k*NUM_SECS_DAY))[:10] )

    # Get the daylight start and end dates for the given year 
    dst_start, dst_end = get_dst_dates(year=args.year, timezone=args.timezone)
    print(f"For year: {args.year}, Timezone: {args.timezone}, DST starts: {dst_start} and DST ends: {dst_end}")

    # check if we need to check for days for daylight savings change
    check_daylight = False
    if dst_start in days_list or dst_end in days_list:
        check_daylight = True

    # set counter marks/indices to identify when a datasegment belongs to morning, afternoon, evening, or night local time
    # example: 5*multiplier corresponds to YYYY-MM-DDT05:00:00 in UTC time (which is YYYY-MM-DDT00:00:00 in CDT)
    # it may be less confusing to convert back and forth between UTC-local times using the utils functions but this is more computational efficient
    multiplier = NUM_SECS_HOUR/args.dt_step 
    dst_info = {'check': check_daylight, 'start': dst_start, 'end': dst_end, 'mul': multiplier}
    if UTCDateTime(args.start_time).julday > UTCDateTime(dst_start).julday and UTCDateTime(args.start_time).julday <= UTCDateTime(dst_end).julday:
        night_start, morn_start, aft_start, eve_start = 5*multiplier, 11*multiplier, 17*multiplier, 23*multiplier
    else:
        night_start, morn_start, aft_start, eve_start = 6*multiplier, 12*multiplier, 18*multiplier, 24*multiplier
    counter_marks = [night_start, morn_start, aft_start, eve_start]

    # number of iterations in for loop for each day
    j_it = int(args.duration / duration_ic)
    # index of data segment when tapered segment ends                         
    taper_idx = int(args.taper_s * args.samp_rate)

    # loop through days to save files of PSD misfit values
    starttime_str_ext = str(UTCDateTime(args.start_time) - args.taper_s)
    for i in range(args.days):
        d_counter = 0
        misfits_dyn = [None]
        times = []
        print(f"Computing PSD misfit for day {i}")
        starttime = days_list[i] + args.start_time[10:]

        # loop through duration_ic of data per day to correct for instrument response and store ratios
        for j in range(j_it):
            skip = False

            # data processing steps: get waveform, linear detrend, remove response, and cut tapered portions
            try:
                # uses local function that can access local files
                st = utils_process.get_rawdata(args.net, args.sta, args.loc, args.chan, starttime_str_ext, duration_icext,
                    args.samp_rate, plot_wave=False, save=False)
                
            # if there are trimming errors, try getting waveforms directly through the obspy function instead 
            except ValueError as e:
                print('Value error', starttime_str_ext)
                starttime_ext = UTCDateTime(starttime_str_ext)
                try:
                    st = client.get_waveforms(args.net, args.sta, args.loc, args.chan, starttime_ext, starttime_ext+(duration_icext),
                                        attach_response=False)
                    
                # if no data is available, skip to next iteration
                except header.FDSNNoDataException:
                    print('No data available for segment with start time', starttime_str_ext)
                    skip = True
            
            # if no data is available, skip to next iteration
            except header.FDSNNoDataException:
                print('No data available for segment with start time', starttime_str_ext)
                skip = True

            # update counters and start times if skipping
            if skip or len(st) == 0:
                d_counter = d_counter + 1440     # this counter step is specific for 100 Hz sampling rates
                starttime_str_ext = UTCDateTime(starttime_str_ext) + (duration_ic)
                starttime_str_ext = str(starttime_str_ext)
                continue

            st.detrend('linear')
            st.remove_response(inventory=inv, output=args.units, water_level=None, pre_filt=pre_filt,
                               zero_mean=False, taper=True, taper_fraction=taper_frac, plot=False, fig=None)
            st.detrend('linear')
            data = st[0].data[:npts_durext]
            data = data[taper_idx:-taper_idx]

            # create and append to a list of PSD misfits for each day
            if misfits_dyn[0]==None:
                misfits_dyn, ts, d_counter, counter_marks = get_PSDmf(data, args.samp_rate, smoothpsddict, psdstddict,
                                                                                    counter_marks, dst_info=dst_info, start0=starttime,
                                                                                    counter=d_counter, wlen_s=args.wlen_s, dt_step=args.dt_step)
            else:
                misfits_dyn_ap, ts, d_counter, counter_marks = get_PSDmf(data, args.samp_rate, smoothpsddict, psdstddict,
                                                                                          counter_marks, dst_info=dst_info, start0=starttime,
                                                                                          counter=d_counter, wlen_s=args.wlen_s, dt_step=args.dt_step)

                misfits_dyn = np.append(misfits_dyn, misfits_dyn_ap)

            times.extend(ts)
            starttime_str_ext = UTCDateTime(starttime_str_ext) + (duration_ic)
            starttime_str_ext = str(starttime_str_ext)

        if misfits_dyn[0]!=None:
            print('len of misfits', len(misfits_dyn))

            # save to file the computed features for each day
            rfilename = EXP_PATH / f'PSDmf_{args.chan}_{starttime[0:10]}_{args.units}'
            with open(rfilename.with_suffix('.txt'), "w+") as f:
                for i in range(len(misfits_dyn)):
                    f.write(str(times[i]) + '\t' + str(misfits_dyn[i]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    # Station Parameters 
    parser.add_argument("--net", default='NW', type=str, help='network code')
    parser.add_argument("--sta", default='HQIL', type=str, help='station code') 
    parser.add_argument("--loc", default='00', type=str, help='location code') 
    parser.add_argument("--chan", default='HHZ', type=str, help='desired channel to compute PSD misfit')
    parser.add_argument("--samp_rate", default=100, type=int, help='station sampling rate (Hz)') 
    parser.add_argument("--units", default='ACC', type=str,
                        help='output (displacement, velocity, or acceleration) to apply when correcting for instrument response')

    # time and algorithm parameters
    parser.add_argument("--start_time", default='2015-04-01T00:00:00', type=str,
                        help='start time of desired time series (midnight of UTC time). If hour-time-second is not 00:00:00, change countermarkers manually')
    parser.add_argument("--days", default=7, type=int)
    parser.add_argument("--year", default=2017, type=int)
    parser.add_argument("--wlen_s", default=10, type=float, help='duration (s) of window')
    parser.add_argument("--dt_step", default=5, type=float, help='time step (s) to move SST/LT windows')
    parser.add_argument("--taper_s", default=2.5, type=float,  help='duration (s) of window to taper before response correction')
    parser.add_argument("--duration", default=60 * 60 * 24, type=float, help='duration (usually 24 h) of data to save to file')
    parser.add_argument("--timezone", default='US/Central', type=str, help="Timezone to use for calculating daylight savings dates")

    # plotting and saving parameters
    parser.add_argument("--plotbackground", default=False, action=argparse.BooleanOptionalAction, help='flag if you want to plot background spectra')
    parser.add_argument("--exp_name", default='test', type=str, help='name of experiment folder to save files to')
    args = parser.parse_args()

    # create a dictionary of background PSDs for each channel and day segment
    dayseg_list = ['Morning', 'Afternoon', 'Evening', 'Night']
    freq_10s = np.arange(0.78125, 50.78125, 0.78125)    # frequencies of a 10-s window
    smoothpsddict = dict()
    psdstddict = dict()
    for dayseg in dayseg_list:
        psd_file = np.load(DATA_PATH / f'PSDs/{args.sta}_{args.chan}_{dayseg}.npz')
        smoothpsd_avg = psd_file['smoothpsd_avg']
        psd_std = psd_file['psd_std']
        freq = psd_file['freq']

        # compute smoothed version of PSD standard deviation
        std_power = 10 ** (psd_std / 10)
        smoothpsd_powerstd = classicsmooth(std_power, octaveFraction=(1.0 / 8.0))
        smoothpsd_std = 10 * np.log10(smoothpsd_powerstd)

        # since the frequency resolution of 3-month average (typically using 1-hour segments) will be higher
        # than the 10-s PSD, include only the power values at the frequencies resolved in the 10-s window
        indices = []
        for val in freq_10s:
            indices.append(np.argwhere(np.isclose(freq, val))[0][0])
        smoothpsddict[dayseg] = smoothpsd_avg[indices]
        psdstddict[dayseg] = smoothpsd_std[indices]


    # Plot the background noise spectra
    if args.plotbackground:
        plt.figure()
        plt.plot(freq_10s, smoothpsddict['Morning'], label='Morning', color='indianred')
        plt.plot(freq_10s, smoothpsddict['Afternoon'], label='Afternoon', color='mediumpurple')
        plt.plot(freq_10s, smoothpsddict['Evening'], label='Evening', color ='steelblue')
        plt.plot(freq_10s, smoothpsddict['Night'], label='Night', color='salmon')
        plt.ylabel('Power [10log$_{10}$(m$^2$/s$^4$/Hz)] (db)', fontsize = '12')
        plt.xlabel('Frequency (Hz)', fontsize = '12')
        plt.xscale('log')
        plt.xlim([freq_10s[0],freq_10s[-1]])
        plt.ylim([-145,-85])
        plt.legend(fontsize='10', loc='upper left')
        plt.title(args.chan)
        plt.show()

    compute_PSDfeatures(args, smoothpsddict, psdstddict)