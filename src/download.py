# Download and save 3-component files locally

from obspy import UTCDateTime, clients
import utils_process
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # station parameters
    parser.add_argument("--net", default='NW', type=str, help='network code')
    parser.add_argument("--sta", default='HQIL', type=str, help='station code')
    parser.add_argument("--loc", default='00', type=str, help='location code')
    parser.add_argument('-l', '--chan_list', nargs='+', default=['HH1', 'HH2', 'HHZ'], help='list of channel codes')
    parser.add_argument('--samp_rate', default=100, help='sampling rate of seismic data', type=int)

    # time series parameters
    parser.add_argument('-s', '--start', help='Starting date in UTC (e.g. 2014-04-01)',
                        type=str, required=True)
    parser.add_argument('-d', '--end', help='Ending date in UTC (e.g. 2014-04-05)',
                        type=str, required=True)
    args = parser.parse_args()

    # accommodate for stations that have a location code of ''
    if args.loc == 'None':
        loc = ''
    else:
        loc = args.loc

    # download data from desired times and channels
    days  = int ((UTCDateTime(args.end) - UTCDateTime(args.start)) / 86400) + 1
    print(f'Downloading data for {days} days')
    duration = 60*60*24
    starttime_str = args.start + 'T00:00:00'
    for i in range(days):
        for j in range(len(args.chan_list)):
            print(starttime_str)
            try:
                st = utils_process.get_rawdata(args.net, args.sta, loc, args.chan_list[j], starttime_str, duration, args.samp_rate, plot_wave=False, save=True)
                print('len of stream', len(st[0].data))
            except (clients.fdsn.header.FDSNNoDataException) as error:
                print('No data available')
                continue
        starttime_str = UTCDateTime(starttime_str) + (60*60*24)
        starttime_str = str(starttime_str)[0:19]
