import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
import astropy.table
import pandas as pd 
from scipy.interpolate import interp1d
from astroplan import Observer
import pickle

from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy import units as u
APACHE = EarthLocation.of_site('Apache Point')

from astroplan import download_IERS_A
download_IERS_A()

class SkySpectrum(object):
    def __init__(self, airmass, ecl_lat, SOLARFLUX, tai, gal_lat, gal_lon, sun_alt, sun_sep, moon_phase, moon_ill, moon_sep, moon_alt, no_zodi):
        self.airmass = airmass
        self.ecl_lat = ecl_lat
        self.tai = tai
        self.gal_lat = gal_lat
        self.gal_lon = gal_lon
        self.sun_alt = sun_alt
        self.sun_sep = sun_sep
        self.moon_phase = moon_phase
        self.moon_ill = moon_ill
        self.moon_sep = moon_sep
        self.moon_alt = moon_alt
        self.SOLARFLUX = SOLARFLUX
        self.no_zodi = no_zodi

        Results = pd.DataFrame.from_csv('/Users/parkerf/Research/SkyModel/BOSS_Sky/SkyModel/ContModel/python/MoonResults.csv')
        Results.columns = ['wl','model','data_var','unexplained_var','X2','rX2','c0','c_am','tau','tau2','c_zodi','c_isl','sol','I',
        't0','t1','t2','t3','t4','m0','m1','m2','m3','m4','m5','m6','feb','mar','apr','may','jun','jul','aug','sep','oct','nov',
        'dec','c2','c3','c4','c5','c6']
        self.Results = Results[Results['model'] == 'moon']
        

        THIS_DIR = '/Users/parkerf/Research/SkyModel/BOSS_Sky/SkyModel/ContModel/' #os.path.split(os.path.abspath(os.getcwd()))[0]
        #print(THIS_DIR)
        #calculate albedo
        albedo_file = THIS_DIR+'/files/albedo_constants.csv'
        albedo_table = pd.read_csv(albedo_file, delim_whitespace=True) 
        self.AlbedoConstants = {}
        for constant in list(albedo_table):
            line = interp1d(albedo_table['WAVELENGTH'],albedo_table[constant],bounds_error=False, fill_value=0)
            self.AlbedoConstants[constant] = line 

        #get solar flux data
        solar_data = np.load(THIS_DIR+'/files/solar_flux.npy')
        self.solar_flux = interp1d(solar_data['MJD'], solar_data['fluxobsflux'], bounds_error=False, fill_value = 0)

        #get zenith extinction curve
        self.zen_ext = np.loadtxt('/Users/parkerf/Research/SkyModel/BOSS_Sky/SkyModel/files/ZenithExtinction-KPNO.dat')
        zen_wave = self.zen_ext[:,0]/10.
        ext = self.zen_ext[:,1]
        zext = interp1d(zen_wave, ext, bounds_error=False, fill_value = 'extrapolate')
        k = zext(self.Results['wl'])
        self.tput = 1 - (10**(-0.4*k) - 10**(-0.4*k*self.airmass))

        self.apache = Observer(APACHE)

        zodi_data = pickle.load(open(THIS_DIR+'/files/s10_zodi.pkl','rb'))
        self.zodi = zodi_data(np.abs(self.ecl_lat))

        isl_data = pickle.load(open(THIS_DIR+'/files/isl_map.pkl','rb'))
        self.isl  = isl_data(self.gal_lon,self.gal_lat)[0]

    def albedo(self, moon_phase):
        p1 = 4.06054
        p2 = 12.8802
        p3 = -30.5858
        p4 = 16.7498
        A = []
        for i in range(4):
            A.append(self.AlbedoConstants['a%d'%i](self.Results['wl'])*(moon_phase**i))
        #for j in range(1,4):
        #    A.append(AlbedoConstants['b%s'%str(j)](wave)*(data_table['SOLAR_SELENO']**(2*j-1)))
        A.append(self.AlbedoConstants['d1'](self.Results['wl'])*np.exp(-moon_phase/p1))
        A.append(self.AlbedoConstants['d2'](self.Results['wl'])*np.exp(-moon_phase/p2))
        A.append(self.AlbedoConstants['d3'](self.Results['wl'])*np.cos((moon_phase-p3)/p4))
        lnA = np.sum(A,axis=0)
        Albedo = np.exp(lnA)
        return Albedo

    def create_features(self):
        obs_time = self.tai/86400.
        start_time = Time(obs_time, scale='tai', format='mjd', location=APACHE)
        self.mjd = start_time.mjd
        sun_rise = self.apache.sun_rise_time(start_time, which = 'next')
        sun_set = self.apache.sun_set_time(start_time, which = 'previous')
        hour = ((start_time - sun_set).sec)/3600

        month_frac = start_time.datetime.month + start_time.datetime.day/30.
        hour_frac = hour/((Time(sun_rise, format='mjd') - Time(sun_set,format = 'mjd')).sec/3600.)

        MONTHS = np.zeros(12)

        mm = np.rint(month_frac)
        if mm == 13:
            mm = 1
        MONTHS[int(mm-1)] = 1

        self.MONTHS = np.array(MONTHS)

        HOURS = np.zeros(6)
        levels = np.linspace(0,1,7)

        idx = np.argmin(np.abs(levels-hour_frac))

        HOURS[idx] = 1

        self.HOURS= np.array(HOURS)

    def get_cont_model(self): 
        self.create_features()
        solarF = self.Results['sol']*self.SOLARFLUX #self.solar_flux(self.mjd-self.Results['I'])
        zodi = self.Results['c_zodi']*self.zodi

        airmass = self.Results['c_am']*self.airmass

        months =  self.Results['feb']*self.MONTHS[1]+self.Results['mar']*self.MONTHS[2]+self.Results['apr']*self.MONTHS[3]+self.Results['may']*self.MONTHS[4]+self.Results['jun']*self.MONTHS[5]+self.Results['jul']*self.MONTHS[6]+self.Results['sep']*self.MONTHS[8]+self.Results['oct']*self.MONTHS[9]+self.Results['nov']*self.MONTHS[10]+self.Results['dec']*self.MONTHS[11]
        hours = self.Results['c2']*self.HOURS[1] +self.Results['c3']*self.HOURS[2] + self.Results['c4']*self.HOURS[3] + self.Results['c5']*self.HOURS[4] + self.Results['c6']*self.HOURS[5]

        twi = (self.Results['t0']*np.abs(self.sun_alt) + self.Results['t1']*(np.abs(self.sun_alt))**2 +  self.Results['t2']*np.abs(self.sun_sep) **2 + self.Results['t3']*np.abs(self.sun_sep)) * np.exp(-self.Results['t4']*self.airmass)

        ALB = self.albedo(self.moon_phase)
        moon = (self.Results['m0'] * self.moon_alt**2 + self.Results['m1'] * self.moon_alt + self.Results['m2'] * self.moon_ill**2 + self.Results['m3'] * self.moon_ill + self.Results['m4'] * self.moon_sep**2 + self.Results['m5'] * self.moon_sep ) * np.exp(-self.Results['m6']*self.airmass) * ALB   

        if self.sun_alt > -20:

            self.model = (self.Results['c0'] + solarF +  months + hours + airmass + zodi + self.Results['c_isl']*self.isl)*self.tput + twi
        elif self.no_zodi:
            self.model = (self.Results['c0'] + solarF +  months + hours + airmass + self.Results['c_isl']*self.isl)*self.tput + moon
        else:
            self.model = (self.Results['c0'] + solarF +  months + hours + airmass + zodi + self.Results['c_isl']*self.isl)*self.tput + moon




    def get_cont_spectrum(self):
        self.get_cont_model()
        wl = np.array(self.Results['wl'])
        sort = np.argsort(wl)
        func = interp1d(wl[sort], self.model[sort], bounds_error = False,fill_value = 'extrapolate')

        wave = np.linspace(360,1000, (1001-360))

        return wl[sort], func(wl[sort])






