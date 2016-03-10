##############################################################
# intnx function (replicates sas)
# need to add the weekday and shift operator functionality
#############################################################
from __future__ import division
import datetime
import pandas
import numpy as np
from math import ceil

#--------------------------------------
# a function to determine if a given
# calendar year is a leap year
#--------------------------------------
def leap_year(y):
    if y % 4 != 0:
        return False
    elif y % 4 ==0 and y % 100 == 0 and y % 400 != 0:
        return False
    else:
        return True

#--------------------------------------------------
## interval - declares the frequency to convert to
## shift_index - optional shift index (not in yet)
## start - datetime object or series or a python time index
## increment - number of steps forward
## alignment - beginning (b), middle (m), end (e), same (S)
#------------------------------------------------

def intnx(interval, start, increment, alignment='beginning'):
    
    # convert pandas time indexes to datetime objects
    if type(start)==pandas.tseries.index.DatetimeIndex:
        start = start.to_pydatetime()

    # convert single datetime objects to numpy arrays
    if type(start) in [datetime.datetime, pandas.tslib.Timestamp]:
        start = np.array([start])

    # define the ends of the month
    mnth_ends = [31,28,31,30,31,30,31,31,30,31,30,31]

    # define the shift operator
    if "." in interval:
        shift = int(interval.split('.')[1])
        interval = interval.split('.')[0]
    
    # determine the number of periods in the interval annually
    if interval.lower() in ['year', 'y']:
        periods = 12
    elif interval.lower() in ['quarter', 'q']:
        periods = 3
    elif interval.lower() in ['month', 'm']:
        periods = 1
    elif interval.lower() in ['week', 'w']:
        pass
    else:
        print "Interval must be week(w), month (m), quarter (q), or year (y)"
    
    for i in xrange(len(start)):
        # advance the date forward by weeks:
        if interval.lower() in ['week', 'w']:
            start[i] = start[i] + datetime.timedelta(days=7*increment)

            if start[i].isoweekday()!=shift:
                start[i] = start[i] + datetime.timedelta(days=shift - start[i].isoweekday()) 

        else:    
            # convert the interval to months and find new year and month
            # new year
            new_year = start[i].year + ((start[i].month -1) + increment*periods)//12
            # new month
            if (start[i].month + increment*periods) % 12 ==0:
                new_month = 12
            else:
                new_month = (start[i].month + increment*periods) % 12

            #adjust for changing month ends
            if start[i].day > mnth_ends[new_month-1]:
                start[i] = start[i].replace(year=new_year, month=new_month, day=mnth_ends[new_month-1])
            else:
                start[i] = start[i].replace(year=new_year, month=new_month)

            # now add in the alignment features
            if alignment.lower() in ['beginning', 'b']:
                if interval.lower() in ['quarter', 'q']:
                    start[i] = start[i].replace(month=int(3*ceil(start[i].month/12*4)-2), day=1)
                if interval.lower() in ['year', 'y']:
                    start[i] = start[i].replace(month=1, day=1)
                else:
                    start[i] = start[i].replace(day=1)

            elif alignment.lower() in ['middle', 'm']:
                if interval.lower() in ['quarter', 'q']:
                    start[i] = start[i].replace(month=int(3*ceil(start[i].month/12*4)-1))
                if interval.lower() in ['year', 'y']:
                    start[i] = start[i].replace(month=6)
                if start[i].month==2 and leap_year(start[i].year)==True:
                    start[i] = start[i].replace(int(day=ceil(29/2)))
                else:
                    start[i] = start[i].replace(day=int(ceil(mnth_ends[start[i].month-1]/2)))

            elif alignment.lower() in ['end', 'e']:
                if interval.lower() in ['quarter', 'q']:
                    start[i] = start[i].replace(month=int(3*ceil(start[i].month/12*4)))
                if interval.lower() in ['year', 'y']:
                    start[i] = start[i].replace(month=12)
                if start[i].month==2 and leap_year(start[i].year)==True:
                    start[i] = start[i].replace(day=29)
                else:
                    start[i] = start[i].replace(day=int(ceil(mnth_ends[start[i].month-1])))

            elif alignment.lower() in ['same', 's']:
                pass
    
            else:
                print "alignment must be beginning (b), middle (m), end (e), or same (s)"

    return start
