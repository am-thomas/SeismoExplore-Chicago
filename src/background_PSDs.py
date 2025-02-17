# Computes and saves averaged PSDs for 50 random days in the period July 2014 - December 2019

import argparse
from constants import DATA_PATH, METADATA_PATH, NUM_SECS_HOUR
import numpy as np
import matplotlib.pyplot as plt
import utils_process
from obspy import read_inventory, UTCDateTime, clients
from obspy.clients.fdsn import Client
client = Client('IRIS')
import utils
from datetime import datetime, timedelta

def classicsmooth(x, octaveFraction=1.0/8.0):
    # apply 1/8 octave smoothing on a given array (x)
    smooth = []
    rightFactor = 2.0 ** (octaveFraction / 2.0)
    leftFactor = 1.0 / rightFactor
    for n in range(len(x)):
        left = int(n * leftFactor)
        right = int(n * rightFactor)
        smooth_val = sum(x[left: right + 1]) / (right - left + 1)
        smooth.append(smooth_val)
    return smooth

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Station Parameters
    parser.add_argument("--net", default='NW', type=str, help='network code')
    parser.add_argument("--sta", default='HQIL', type=str, help='station code')
    parser.add_argument("--loc", default='00', type=str, help='location code')
    parser.add_argument("--chan", required=True, type=str, help='channel code')
    parser.add_argument("--samp_rate", default=100, type=int, help='station sampling rate (Hz)')
    parser.add_argument("--units", default='ACC', type=str,
                        help='output (displacement, velocity, or acceleration) to apply when correcting for instrument response')
    parser.add_argument("--days", default=50, type=int)
    parser.add_argument("--dayseg", type=str, required=True, help='''6-hour day segment over which to average spectra
                    # Night: midnight - 6 am local time
                    # Morning: 6 am - 12 pm local time
                    # Afternoon: 12 pm - 6 pm local time
                    # Evening: 6 pm - midnight local time
                    ''')
    args = parser.parse_args()

    # load spectra of Peterson Low Noise Model and High Noise Hodel
    noisemodel = np.loadtxt('noisemodels_IRIS.txt',delimiter=',')
    period_nm = noisemodel[:, 0]
    lnmodel = noisemodel[:,1]
    hnmodel = noisemodel[:,2]

    # set processing parameters
    pre_filt = [0.004, 0.006, 35, 50]
    inv = read_inventory(METADATA_PATH / f'{args.sta}.xml')
    taper_s = 5
    taper_idx = int(taper_s * args.samp_rate)
    duration_dayseg = NUM_SECS_HOUR * 6

    # set duration to process at a time (20 min) and compute additional parameters
    duration_proc = 60*20
    duration_procext = duration_proc + (2*taper_s)
    npts_procext = duration_procext * args.samp_rate
    taper_frac = taper_s / duration_procext

    # create dictionary to define afternoon, evening, morning, and night time starting boundaries (in Us/Central Time zone)
    hmstime_dict = {'Night': '00:00:00', 'Morning': '06:00:00', 'Afternoon':'12:00:00', 'Evening':'18:00:00'}
    starthour = hmstime_dict[args.dayseg]

    # randomly select 300 days between 2014-07-01 - 2019-12-24 (2003 days in total).
    # We will only use 50 days but getting a larger collection, because we will exclude days with gaps and poor data
    # rand_days = np.random.choice(np.arange(0, 2003), 300, replace=False)
    # np.save(DATA_PATH/'PSDs/randomized_days.npy', rand_days)

    # load random days between 2014-07-01 - 2019-12-24 (2003 days in total)
    rand_days = np.load(DATA_PATH/'PSDs/randomized_days.npy')

    # store list of days to exclude: 2016-12-09 to 2017-03-31 and 2017-10-12 to 2017-10-25
    exclude_days = np.concatenate((np.arange(892, 1005), np.arange(1199, 1213)))
    
    # Get PSDs for segments of duration_dayseg in each day, averaged over input number of days
    freqs = []
    psds = []
    smooth_psds = []
    counter = 0
    idx = 0

    print('***************')
    print("NOTE: PSDs will be calculated faster if you have data saved as local files")
    print('***************')
    print('Calculating PSDs for the following segments:')

    while counter < args.days:
        
        # skip excluded days
        day = rand_days[idx]
        if day in exclude_days:
            print('Skipping excluded day...')
            idx = idx + 1
            continue

        # set start time and end time of  in UTC format
        starttime_str = datetime.strptime('2014-07-01 ' + starthour, "%Y-%m-%d %H:%M:%S") + timedelta(days=int(day))
        starttime_str = str(utils.local_to_utc(str(starttime_str), 'US/Central'))     
        endtime_str = str(UTCDateTime(starttime_str) + duration_dayseg)
        print(f'{counter}: {starttime_str} to {endtime_str}')

        try:
            # retrieve 6 hour data and process 20 minutes at a time to remove long-period noise
            starttime_proc = UTCDateTime(starttime_str)-taper_s  
            starttime_str_proc = str(starttime_proc)
            for j in range( int(duration_dayseg/duration_proc) ):
                endtime_proc = starttime_proc + duration_procext
                st = utils_process.get_rawdata(args.net, args.sta, args.loc, args.chan, starttime_str_proc, duration_procext,
                                args.samp_rate, plot_wave=False, save = False)
                if len(st[0].data) < npts_procext:
                        raise ValueError('Retrieved data does not have the expected length')
                
                st.detrend('linear')
                st.remove_response(inventory=inv, output=args.units, water_level=None, pre_filt=pre_filt,
                                    zero_mean=True, taper=True, taper_fraction=taper_frac, plot=False, fig=None)
                st.detrend('linear')

                #cut off tapered portions
                procdata_subseg = st[0].data[:npts_procext]
                procdata_subseg = procdata_subseg[taper_idx:-taper_idx]

                # append to array of processed data for 1 day
                if j == 0:
                    proc_data = procdata_subseg
                else:
                    proc_data = np.append(proc_data, procdata_subseg)

                starttime_str_proc = str(UTCDateTime(starttime_str_proc)+duration_proc)

        # skip and store days where no data is available
        except (clients.fdsn.header.FDSNNoDataException, ValueError, IndexError) as error:
            print('No or missing data for this segment:', starttime_str)
            idx = idx + 1
            continue

        # calc PSDs
        freq1, psd1 = utils_process.my_PSD(proc_data, args.chan, duration_dayseg, samp_rate=args.samp_rate, dt_sgmt=60 * 60,
                                            ovlap_sgmt=0.50, dt_subgmt=60 * 15, n_subsgmts=13, ovlap_subsgmt=0.75,
                                            tpr_alpha=0.20, resp_correct=False, smooth=False)
        freqs.append(freq1)
        psds.append(psd1)

        # update idx and counter
        idx = idx + 1
        counter = counter + 1

    # compute the mean, median, 10th percentile, 90th percentile of PSDs
    print('n', len(psds))
    psd_avg = np.mean(psds, axis=0)
    psd_med = np.median(psds, axis=0)
    psd_10 = np.percentile(psds, 10, axis=0)
    psd_90 = np.percentile(psds, 90, axis=0)
    psd_std = np.std(psds, axis=0)

    # compute smooth psd of averaged PSD
    power = 10 ** (psd_avg / 10)
    smoothpwr = classicsmooth(power, octaveFraction=(1.0 / 8.0))
    smoothpsd_avg = 10 * np.log10(smoothpwr)

    # save to file
    np.savez(DATA_PATH / f'PSDs/{args.sta}_{args.chan}_{args.dayseg}.npz', freq=freq1, psd_med=psd_med,
                psd_avg=psd_avg, psd_std=psd_std, perc10 = psd_10, perc90 = psd_90, smoothpsd_avg=smoothpsd_avg)

    # load PSD file
    psd_file = np.load(DATA_PATH / f'PSDs/{args.sta}_{args.chan}_{args.dayseg}.npz')
    smoothpsd_avg = psd_file['smoothpsd_avg']
    psd_std = psd_file['psd_std']
    freq = psd_file['freq']

    # compute smoothed version of PSD standard deviation
    std_power = 10 ** (psd_std / 10)
    smoothpsd_powerstd = classicsmooth(std_power, octaveFraction=(1.0 / 8.0))
    smoothpsd_std = 10 * np.log10(smoothpsd_powerstd)

    # decimate smoothed averaged: since the frequency resolution of 3-month average (typically using 1-hour segments) will be higher
    # than the 10-s PSD, include only the power values at the frequencies resolved in the 10-s window
    freq_10s = np.arange(0.78125, 50.78125, 0.78125)
    indices = []
    for val in freq_10s:
        indices.append(np.argwhere(np.isclose(freq, val))[0][0])
    smoothpsd_avg_dec = smoothpsd_avg[indices]
    smoothpsd_std_dec = smoothpsd_std[indices]

    # plot computed PSDs and their standard deviation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    psd_2stdup = smoothpsd_avg_dec + (2 * smoothpsd_std_dec)
    psd_2stdlw = smoothpsd_avg_dec - (2 * smoothpsd_std_dec)
    ax.plot(psd_file['freq'], psd_file['psd_avg'], color='green', label='PSD')
    ax.plot(freq_10s, smoothpsd_avg_dec, color='lightpink', label='Processed PSD')
    ax.fill_between(freq_10s, psd_2stdup, psd_2stdlw, color='lightblue', alpha=0.6)
    # ax.plot(1 / period_nm, lnmodel, color='black', label='NLNM/NHNM', alpha=0.7)
    # ax.plot(1 / period_nm, hnmodel, color='black', alpha=0.7)
    ax.set_xscale('log')
    # ax.set_ylim([-210, -60])
    # ax.set_xlim([0.78125, 50])
    ax.set_xlabel('Frequency (Hz) ')
    ax.set_ylabel('Power [10log$_{10}$(m$^2$/s$^4$/Hz)] (db)')
    plt.legend(loc='upper left', prop={"size": 9})
    plt.show()
    #plt.savefig(DATA_PATH / f'PSD_{args.chan}_{args.dayseg}.png', dpi=300)



