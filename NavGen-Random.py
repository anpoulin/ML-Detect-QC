import numpy as np
import pandas as pd
import os
from io import StringIO
from datetime import datetime, timedelta
from pyproj import CRS, Transformer
import random
from random import randrange


def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)

start = datetime.strptime('01/01/2024 00:00:00.600', '%m/%d/%Y %H:%M:%S.%f')
stop = datetime.strptime('01/30/2024 00:00:00.800', '%m/%d/%Y %H:%M:%S.%f')

numOfShots = 1000

outputPath = 'C:/Projects/FORGE/NAV'

#inputXLSX   = '/Users/david/Developer/Silixa/CenovusLashburn/Shot Log.xlsx'
path, file = os.path.split(inputCSV)

# # ITRF2014 epsg ITRF2014 and WGS84 are extremly similar, so just use WGS84
# crs_ITRF2014    = CRS.from_epsg(32612)
# # UTM 12N NAD27
# crs_UTM12N      = CRS.from_epsg(26712)
# # Projection from ITRF to UTM
# projectionUTM   = Transformer.from_crs(crs_UTM12N.geodetic_crs, crs_UTM12N)
# projectionITRF  = Transformer.from_crs(crs_ITRF2014.geodetic_crs, crs_ITRF2014)
# projectionRev   = Transformer.from_crs(crs_ITRF2014, crs_ITRF2014.geodetic_crs)

# lat =  53  + 11/60. + 50.969/3600.
# lon = -(109 + 24/60. + 40.822/3600.)
# eastUTM, northUTM   = projectionUTM.transform(lat, lon)
# eastITRF, northITRF = projectionITRF.transform(lat, lon)
# # latRev, lonRev      = projectionRev.transform(eastITRF, northITRF)
# # latUTM, lonUTM      = projectionUTM.transform(latRev, lonRev)
# print(lat, lon)
# print(latRev, lonRev)
# print(eastUTM, northUTM)
# print(eastITRF, northITRF)




my_list = list(df)
colsOut = list()
for i in range(len(numOfShots)):
    cnt = 1
    shotNum = cnt+i
    DTobject = datetime.strptime(str(UTC), '%Y%m%d%H%M%S.%f')
    DTevtRan = print(random_date(start, stop))
    colsOut.append( [shotNum, DTevtRan.strftime('%Y%m%d %H:%M:%S.%f')] )


# Output filenames
fOut = outputPath + '/NAV_16A_9N-Ran.csv'
# Use header='...' and comments='' to output NAV compatible header with columns names (no hash mark)
np.savetxt(fOut, colsOut, fmt='%s',  delimiter=',', comments='', header='shotNum,timestamp (UTC)')