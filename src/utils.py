import pickle
import os 
from constants import RAW24H_PATH
import pytz
from datetime import datetime, timedelta



def utc_to_local(utc_str, pytz_timezone):
    '''
    Converts time in UTC to local time zone
    :parameter
        utc_str (string): time in UTC with format yyyy-MM-dd'T'HH:mm:ss
        pytz_timezone (string): local time zone
    :returns
        local_datetime (datetime object): time in Central Standard Time zone
    '''
    utc_datetime = datetime.strptime(utc_str,'%Y-%m-%dT%H:%M:%S')
    local_timezone = pytz.timezone(pytz_timezone)
    local_datetime = utc_datetime.replace(tzinfo=pytz.utc)
    local_datetime = local_datetime.astimezone(local_timezone)
    return local_datetime


def local_to_utc(time_str, pytz_timezone):
    '''
    Converts time in local time zone to UTC
    :parameter
        utc_str (string): time in UTC with format yyyy-MM-dd HH:mm:ss
        pytz_timezone (string): local time zone
    :returns
        local_datetime (datetime object): time in desired local time zone
    '''
    local = pytz.timezone(pytz_timezone)
    utc = pytz.utc
    naive_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    local_time = local.localize(naive_time)
    utc_time = local_time.astimezone(utc)
    return utc_time.strftime('%Y-%m-%d %H:%M:%S')


def save_file(filename, file):
    """
    Save file in pickle format
    Args:
        file (any object): Can be any Python object. We would normally use this to save the
        processed Pytorch dataset
        filename (str): Name of the file
    """

    with open(filename, 'wb') as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(filename):
    """
    Load a pickle file
    Args:
        filename (str): Name of the file
    Returns (Python obj): Returns the loaded pickle file
    """
    with open(filename, 'rb') as f:
        file = pickle.load(f)

    return file


def get_dst_dates(year, timezone="US/Central"): 
    """
    Given a year and timezone return the start and end dates of daylight savings time 
    """
    months = 12 
    days = 31 
    timezone = pytz.timezone(timezone)
    dst_start, dst_end = None, None 

    for m in range(1, months+1): 
        for d in range(1, days+1):
            try: 
                dt = datetime(year, m, d, hour=3, minute=0, second=0)
                timezone_aware_date = timezone.localize(dt, is_dst=None)
                date_is_daylight = timezone_aware_date.tzinfo.dst(timezone_aware_date) != timedelta(0)
                if not dst_start and date_is_daylight: 
                    dst_start = dt 
                elif dst_start and not date_is_daylight and not dst_end:
                    dst_end = dt  
            except: 
                pass 
     
    dst_start = dst_start.strftime('%Y-%m-%d')
    dst_end = dst_end.strftime('%Y-%m-%d')
    
    return dst_start, dst_end


def get_allgapdays(net, sta, chan, gapfilename='NQ_HQIL_gaps_savedfiles.txt'):
    prnt_folder = f'{net}_{sta}'
    # Combine all individual day gap text files into one merged file
    gap_files = []
    for subdir, dirs, files in os.walk(RAW24H_PATH / prnt_folder / chan):
        for file in files:
            if file.endswith('_gaps.txt'):
                gap_files.append(os.path.join(subdir, file))

    with open(RAW24H_PATH / prnt_folder / chan / gapfilename, 'w') as outfile:
        for fname in gap_files:
            with open(fname) as infile:
                outfile.write(infile.read())
    print('Created new merged file of all gaps in RAW_24H')