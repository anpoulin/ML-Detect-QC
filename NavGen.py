import numpy as np
import pandas as pd
import os
from io import StringIO
from datetime import datetime, timedelta
from pyproj import CRS, Transformer
import random

#inputXLSX   = '/Users/david/Developer/Silixa/CenovusLashburn/Shot Log.xlsx'
inputCSV    = r'C:\Projects\RTE\Dev\NAV\EventCatalogue_16A_9.csv'
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



df = pd.read_csv(inputCSV) 
my_list = list(df)
colsOut = list()
for i in range(len(df)):
    cnt = 1
    shotNum = cnt+i
    UTC     = df[my_list[0]][i]
    DTobject = datetime.strptime(str(UTC), '%Y%m%d%H%M%S.%f')
    DToffset = 1
    print(DToffset)
    DTevtOT = df[my_list[9]][i]
    DTevtRan = DTobject-timedelta(seconds=DToffset)+timedelta(seconds=DTevtOT)
    # line    = df[my_list[4]][i]
    # station = df[my_list[5]][i]
    # X = df[my_list[6]][i]
    # Y = df[my_list[7]][i]
    # Z = df[my_list[8]][i]
    colsOut.append( [shotNum, DTevtRan.strftime('%Y%m%d %H:%M:%S.%f')] )


# Output filenames
fOut = path + '/NAV_16A_9N.csv'
# Use header='...' and comments='' to output NAV compatible header with columns names (no hash mark)
np.savetxt(fOut, colsOut, fmt='%s',  delimiter=',', comments='', header='shotNum,timestamp (UTC)')