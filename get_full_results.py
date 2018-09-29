import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
import glob
import astropy.table
import pandas as pd
from datetime import datetime
import multiprocessing
sys.path.insert(0,'/Users/parkerf/Research/SkyModel/BOSS_Sky/SkyModel/ContModel/python/')

import cont_model as Model

def main():
    start = datetime.now()

    #Meta = astropy.table.Table.read('/Users/parkerf/Research/SkyModel/BOSS_Sky/good_mean_meta_071718.fits')
    blue_wave_files  = glob.glob('/Volumes/PFagrelius_Backup/sky_data/wave_arrays/blue*.npy')
    red_wave_files  = glob.glob('/Volumes/PFagrelius_Backup/sky_data/wave_arrays/red*.npy')

    global BSMeta
    BSMeta = astropy.table.Table.read('/Volumes/PFagrelius_Backup/sky_data/wave_arrays/blue_sorted_good_mean_meta_071726.fits')
    global RSMeta
    RSMeta = astropy.table.Table.read('/Volumes/PFagrelius_Backup/sky_data/wave_arrays/red_sorted_good_mean_meta_071726.fits')

    #Need to use sorted meta if using wavelength data
    # DarkResults = []
    # TwiResults = []
    # MoonResults = []
    #FullResults = []

    pool = multiprocessing.Pool(processes=4)
    bret = pool.map(make_blue_model_for_wave, blue_wave_files)
    rret = pool.map(make_red_model_for_wave, red_wave_files)
    pool.terminate()

    bret = np.vstack(bret)
    rret = np.vstack(rret)
    ret = np.vstack([bret, rret])
    Ret = []
    for i in ret:
        if i is None:
            pass
        else:
            Ret.append(i)
    MoonResults = pd.DataFrame(Ret)
    #print(MoonResults)

    #MoonResults = pd.concat([bret,rret])
    MoonResults.to_csv('MoonResults.csv', sep=',')
    print("Total Time: ", (datetime.now() - start).total_seconds())


def make_blue_model_for_wave(wave_file):
    wl = os.path.splitext(wave_file)[0][-3:]

    if float(wl) in np.linspace(360,620,(621-360)):
        print(wl)
        yy = np.load(wave_file)
        TM = Model.ContModel(yy, wl, 'blue', BSMeta)
        
        try:
            Results = []
            TM.run_dark_model()
            Results.append(TM.get_params())
            TM.run_twi_model()
            Results.append(TM.get_params())
            TM.run_moon_model()
            Results.append(TM.get_params())
            return np.vstack(Results)
        except:
            print("%d didn't work"%int(wl))


def make_red_model_for_wave(wave_file):
    wl = os.path.splitext(wave_file)[0][-3:]

    if float(wl) in np.linspace(621,1030,(1031-621)):
        print(wl)
        yy = np.load(wave_file)
        TM = Model.ContModel(yy, wl, 'red', RSMeta)

        try:
            Results = []
            TM.run_dark_model()
            Results.append(TM.get_params())
            TM.run_twi_model()
            Results.append(TM.get_params())
            TM.run_moon_model()
            Results.append(TM.get_params())
            return np.vstack(Results)
        except:
            print("%d didn't work"%int(wl))


# for wave in red_wave_files:
#     wl = os.path.splitext(wave)[0][-3:]
#     if float(wl) in np.linspace(621,1030,(1031-621)):
#         yy = np.load(wave)
#         TM = Model.ContModel(yy, wl, 'red', RSMeta)
#         try:
#             TM.run_dark_model()
#             DarkResults.append(TM.get_params())
#             TM.run_twi_model()
#             TwiResults.append(TM.get_params())
#             #TM.run_moon_model()
#             #MoonResults.append(TM.get_params())
#             #TM.run_full_model()
#             #FullResults.append(TM.get_params())
#         except:
#             print(wl)
# DarkResults = pd.concat(DarkResults)
# TwiResults = pd.concat(TwiResults)
# #MoonResults = pd.concat(MoonResults)
# #FullResults = pd.concat(FullResults)

# DarkResults.to_csv('DarkResults.csv', sep=',')
# TwiResults.to_csv('TwiResults.csv', sep=',')
# print("Total Time: ", (datetime.now() - start).total_seconds())
#MoonResults.to_csv('MoonResults.csv', sep=',')
#FullResults.to_csv('FullResults.csv', sep=',')

if __name__ == '__main__':
    main()

