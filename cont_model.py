import os
import numpy as np
import astropy.table
from astropy.io import fits
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle
import pandas as pd
from scipy import stats
import seaborn as sns
import math

from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy import units as u
APACHE = EarthLocation.of_site('Apache Point')

from lmfit import models, Parameters, Parameter, Model


class ContModel(object):
    def __init__(self, data, wl, color, Meta):

        self.data = data
        self.wl = int(wl) #wavelength for data
        self.Meta = Meta
        self.color = color

        #fit parameters
        self.model_name = None
        self.data_var = None
        self.unexplained_var = None
        self.chisqr = None
        self.redchisqr = None

        #model parameters
        self.c_0 = None
        self.c_am = None
        self.tau = None
        self.tau2 = None
        self.c_zodi = None
        self.c_isl = None
        self.sol = None
        self.i = None
        self.t0 = None
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None
        self.m0 = None
        self.m1 = None
        self.m2 = None
        self.m3 = None
        self.m4 = None
        self.m5 = None
        self.m6 = None
        self.feb = None
        self.mar = None
        self.apr = None
        self.may = None
        self.jun = None
        self.jul = None
        self.aug = None
        self.sep = None
        self.oct = None
        self.nov = None
        self.dec = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.c5 = None
        self.c6 = None

        THIS_DIR = os.path.split(os.path.abspath(os.getcwd()))[0]

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
        self.k = zext(self.wl)

    def albedo(self, moon_phase):
        p1 = 4.06054
        p2 = 12.8802
        p3 = -30.5858
        p4 = 16.7498
        A = []
        for i in range(4):
            A.append(self.AlbedoConstants['a%d'%i](self.wl)*(moon_phase**i))
        #for j in range(1,4):
        #    A.append(AlbedoConstants['b%s'%str(j)](wave)*(data_table['SOLAR_SELENO']**(2*j-1)))
        A.append(self.AlbedoConstants['d1'](self.wl)*np.exp(-moon_phase/p1))
        A.append(self.AlbedoConstants['d2'](self.wl)*np.exp(-moon_phase/p2))
        A.append(self.AlbedoConstants['d3'](self.wl)*np.cos((moon_phase-p3)/p4))
        lnA = np.sum(A,axis=0)
        Albedo = np.exp(lnA)
        return Albedo

    def add_ext(self, airmass,tau):

        tput = 1 - (10**(-0.4*self.k*tau) - 10**(-0.4*self.k*airmass*tau))
        return tput


    def create_feature_list(self):

        obs_time = 0.5*((self.ThisMeta['TAI-BEG']+self.ThisMeta['TAI-END'])/86400.)
        start_time = Time(obs_time, scale='tai', format='mjd', location=APACHE)
        self.ThisMeta['MONTH_FRAC'] = [time.datetime.month + time.datetime.day/30. for time in start_time]
        self.ThisMeta['HOUR_FRAC'] = self.ThisMeta['HOUR']/((Time(self.ThisMeta['SUN_RISE'], format='mjd') - Time(self.ThisMeta['SUN_SET'],format = 'mjd')).sec/3600.)
        all_months = np.zeros((len(self.ThisMeta), 12))
        months = np.array(np.rint(self.ThisMeta['MONTH_FRAC'])-1,dtype=int)

        months_ = []
        for i, month in enumerate(all_months):

            if int(months[i]) == 12:
                x = 0
            else:
                x = int(months[i])
            month[x] = 1
            months_.append(month)
        MM = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
        months_ = np.array(months_)

        for i,M in enumerate(MM):
            self.ThisMeta[M] = astropy.table.Column(months_[:,i])

        hours = np.zeros((len(self.ThisMeta), 6))
        levels = np.linspace(0,1,7)
        blue_hours = np.array(self.ThisMeta['HOUR_FRAC'])
        hours_ = []
        for i, hour in enumerate(hours):
            for x in range(6):
                if (blue_hours[i] > levels[x]) & (blue_hours[i] <= levels[x+1]):
                    xx = x

            hour[xx] = 1
            hours_.append(hour)
        hours_ = np.array(hours_)
        HH = ['H1','H2','H3','H4','H5','H6']
        for i, H in enumerate(HH):
            self.ThisMeta[H] = astropy.table.Column(hours_[:,i])
        
        D = self.ThisMeta.as_array()
        columns = ['JAN','FEB','MAR','APR','MAY',
                   'JUN','JUL','AUG','SEP','OCT','NOV','DEC',
                   'MJD','ECL_LAT','MOON_PHASE','MOON_ILL','MOON_ALT','MOON_SEP','SUN_SEP',
                   'SUN_ALT','AIRMASS', 'ALT','ZODI','ISL','H1','H2','H3','H4','H5','H6']
        df = pd.DataFrame(D[columns])
        df = df.drop(['JAN','H1'], axis = 1)

        self.Features = df

    def clean_data(self):
        idx = np.isfinite(self.ThisData)
        self.ThisMeta = self.ThisMeta[idx]
        self.ThisData = self.ThisData[idx]

    def color_cut(self):
        if self.color == 'blue':
            idx = np.where((self.ThisMeta['CAMERAS'] == 'b1')|(self.ThisMeta['CAMERAS'] == 'b2'))
            self.ThisMeta = self.ThisMeta[idx]
            self.ThisData = self.ThisData[idx]
        elif self.color == 'red':
            idx = np.where((self.ThisMeta['CAMERAS'] == 'r1')|(self.ThisMeta['CAMERAS'] == 'r2'))
            self.ThisMeta = self.ThisMeta[idx]
            self.ThisData = self.ThisData[idx]

    def get_photo_data(self):
        idx = np.where((self.ThisMeta['PHOTO'] == 1))
        self.ThisMeta = self.ThisMeta[idx]
        self.ThisData = self.ThisData[idx]

    def get_params(self):
        NAMES = ('wl','model','data_var','unexplained_var','X2','rX2','c0','c_am','tau','tau2','c_zodi','c_isl','sol','I',
        't0','t1','t2','t3','t4','m0','m1','m2','m3','m4','m5','m6','feb','mar','apr','may','jun','jul','aug','sep','oct','nov',
        'dec','c2','c3','c4','c5','c6')

        Results = [self.wl,self.model_name, self.data_var, self.unexplained_var, self.chisqr, self.redchisqr,
        self.c_0,self.c_am,self.tau,self.tau2,self.c_zodi,self.c_isl,self.sol,self.i,self.t0,self.t1,self.t2,self.t3,self.t4,self.m0,
        self.m1,self.m2,self.m3,self.m4,self.m5,self.m6,self.feb, self.mar,self.apr,self.may,self.jun,self.jul,self.aug,
        self.sep,self.oct,self.nov,self.dec,self.c2,self.c3,self.c4,self.c5,self.c6]

        df = pd.DataFrame([Results], columns = list(NAMES))
        return df

    def run_dark_model(self):
        def dark_cont_model( x, tau, I, c_0, c_am, c_z, c_i, c_sol, c_feb, c_mar, c_apr, c_may, c_jun, c_jul, c_sep, c_oct, c_nov, c_dec, c_2, c_3, c_4, c_5, c_6): 
            FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,MJD,ECL_LAT,MOON_PHASE,MOON_ILL,MOON_ALT,MOON_SEP,SUN_SEP,SUN_ALT,AIRMASS,ALT,ZODI,ISL,H2,H3,H4,H5,H6 = x

            solarF = c_sol*self.solar_flux(MJD-I)
            zodi = c_z*ZODI

            airmass = c_am*AIRMASS
            tput = self.add_ext(AIRMASS,tau)

            months =  c_feb*FEB+c_mar*MAR+c_apr*APR+c_may*MAY+c_jun*JUN+c_jul*JUL+c_sep*SEP+c_oct*OCT+c_nov*NOV+c_dec*DEC
            hours = c_2*H2 + c_3*H3 + c_4*H4 + c_5*H5 + c_6*H6

            model = (c_0 + solarF +  months + hours + airmass + zodi + c_i*ISL)*tput
            return model


        mod = Model(dark_cont_model)

        params = mod.make_params()
        params.add('I', value = 5)
            
        params.add('c_0', value = 2)
        params.add('c_am', value = 0.2,min=0)
        params.add('tau', value = 1, min = 0)
        params.add('c_z', value = 1)
        params.add('c_i', value = 0.069191)
        params.add('c_sol', value = 0.006710)

        params.add('c_feb', value = 0.052769)
        params.add('c_mar', value = 0.016099)
        params.add('c_apr', value = -0.038161)
        params.add('c_may', value = -0.003831)
        params.add('c_jun', value = 0.056229)
        params.add('c_jul', value = -0.093193)
        params.add('c_aug', value = -0.035002)
        params.add('c_sep', value = -0.061367)
        params.add('c_oct', value = 0.082170)
        params.add('c_nov', value = -0.037848)
        params.add('c_dec', value = 0.016476)

        params.add('c_2', value = .1)
        params.add('c_3', value = .1)
        params.add('c_4', value = .1)
        params.add('c_5', value = .1)
        params.add('c_6', value = .1)

        dark_idx = np.where((self.Meta['AIRMASS']<1.4)&(self.Meta['MOON_ALT']<0)&(self.Meta['SUN_ALT']<-20)&(self.Meta['GAL_LAT']>10))
        self.ThisMeta = self.Meta[dark_idx]
        self.ThisData = self.data[dark_idx]
        self.clean_data()
        self.color_cut()
        #self.get_photo_data()
        self.create_feature_list()
        y = self.ThisData
        X = self.Features

        self.dark_result = mod.fit(y, params, x =  np.array(X).T)
        self.dark_resids = y - self.dark_result.best_fit

        #fit params
        self.model_name = 'dark'
        self.data_var = np.std(y)
        self.unexplained_var = np.std(self.dark_resids)
        self.chisqr = self.dark_result.chisqr
        self.redchisqr = self.dark_result.redchi
        
        self.c_0 = self.dark_result.params['c_0'].value
        self.c_am = self.dark_result.params['c_am'].value
        self.tau = self.dark_result.params['tau'].value
        self.c_zodi = self.dark_result.params['c_z'].value
        self.c_isl = self.dark_result.params['c_i'].value
        self.sol = self.dark_result.params['c_sol'].value
        self.i = self.dark_result.params['I'].value
        self.feb = self.dark_result.params['c_feb'].value
        self.mar = self.dark_result.params['c_mar'].value
        self.apr = self.dark_result.params['c_apr'].value
        self.may = self.dark_result.params['c_may'].value
        self.jun = self.dark_result.params['c_jun'].value
        self.jul = self.dark_result.params['c_jul'].value
        self.aug = self.dark_result.params['c_aug'].value
        self.sep = self.dark_result.params['c_sep'].value
        self.oct = self.dark_result.params['c_oct'].value
        self.nov = self.dark_result.params['c_nov'].value
        self.dec = self.dark_result.params['c_dec'].value
        self.c2 = self.dark_result.params['c_2'].value
        self.c3 = self.dark_result.params['c_3'].value
        self.c4 = self.dark_result.params['c_4'].value
        self.c5 = self.dark_result.params['c_5'].value
        self.c6 = self.dark_result.params['c_6'].value

    def run_twi_model(self):
        def twi_cont_model( x, T0, T1, T2, T3, T4, tau, I, c_0, c_am, c_z, c_i, c_sol, c_feb, c_mar, c_apr, c_may, c_jun, c_jul, c_sep, c_oct, c_nov, c_dec, c_2, c_3, c_4, c_5, c_6): 
            FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,MJD,ECL_LAT,MOON_PHASE,MOON_ILL,MOON_ALT,MOON_SEP,SUN_SEP,SUN_ALT,AIRMASS,ALT,ZODI,ISL,H2,H3,H4,H5,H6 = x

            solarF = c_sol*self.solar_flux(MJD-I)
            zodi = c_z*ZODI

            airmass = c_am*AIRMASS
            tput = self.add_ext(AIRMASS, tau)

            months =  c_feb*FEB+c_mar*MAR+c_apr*APR+c_may*MAY+c_jun*JUN+c_jul*JUL+c_sep*SEP+c_oct*OCT+c_nov*NOV+c_dec*DEC
            hours = c_2*H2 + c_3*H3 + c_4*H4 + c_5*H5 + c_6*H6

            twi = (T0*np.abs(SUN_ALT) + T1*(np.abs(SUN_ALT))**2 +  T2*np.abs(SUN_SEP) **2 + T3*np.abs(SUN_SEP)) * np.exp(-T4*AIRMASS)  

            model = (c_0 + solarF +  months + hours + airmass + zodi + c_i*ISL)*tput + twi
            return model


        mod = Model(twi_cont_model)

        params = mod.make_params()
        params.add('c_0', value = 1, vary = True)
        params.add('T0', value = 1)
        params.add('T1', value = 1)
        params.add('T2', value = 1)
        params.add('T3', value = 1)
        params.add('T4', value = 1)

        params.add('I', value = self.i, vary = False)
        params.add('c_am', value = self.c_am, vary = False)
        params.add('tau', value = self.tau, vary = False)
        params.add('c_z', value = self.c_zodi, vary = False)
        params.add('c_i', value = self.c_isl, vary = False)
        params.add('c_sol', value = self.sol, vary = False)
        params.add('c_feb', value = self.feb, vary = False)
        params.add('c_mar', value = self.mar, vary = False)
        params.add('c_apr', value = self.apr, vary = False)
        params.add('c_may', value = self.may, vary = False)
        params.add('c_jun', value = self.jun, vary = False)
        params.add('c_jul', value = self.jul, vary = False)
        params.add('c_aug', value = self.aug, vary = False)
        params.add('c_sep', value = self.sep, vary = False)
        params.add('c_oct', value = self.oct, vary = False)
        params.add('c_nov', value = self.nov, vary = False)
        params.add('c_dec', value = self.dec, vary = False)
        params.add('c_2', value = self.c2, vary = False)
        params.add('c_3', value = self.c3, vary = False)
        params.add('c_4', value = self.c4, vary = False)
        params.add('c_5', value = self.c5, vary = False)
        params.add('c_6', value = self.c6, vary = False)
        
        #define data
        twi_idx = np.where((self.Meta['MOON_ALT']<0)&(self.Meta['SUN_ALT']>-20))
        self.ThisMeta = self.Meta[twi_idx]
        self.ThisData = self.data[twi_idx]
        self.clean_data()
        self.color_cut()
        #self.get_photo_data()
        self.create_feature_list()
        y = self.ThisData
        X = self.Features

        #do fit
        self.twi_result = mod.fit(y, params, x =  np.array(X).T)
        self.twi_resids = y - self.twi_result.best_fit

        #fit params
        self.model_name = 'twilight'
        self.data_var = np.std(y)
        self.unexplained_var = np.std(self.twi_resids)
        self.chisqr = self.twi_result.chisqr
        self.redchisqr = self.twi_result.redchi
        
        self.c_0 = self.twi_result.params['c_0'].value
        self.t0 = self.twi_result.params['T0'].value
        self.t1 = self.twi_result.params['T1'].value
        self.t2 = self.twi_result.params['T2'].value
        self.t3 = self.twi_result.params['T3'].value
        self.t4 = self.twi_result.params['T4'].value


    def run_moon_model(self):
        def moon_cont_model( x, M0, M1, M2, M3, M4, M5, M6,  T0, T1, T2, T3, T4, tau, I, c_0, c_am, c_z, c_i, c_sol, c_feb, c_mar, c_apr, c_may, c_jun, c_jul, c_sep, c_oct, c_nov, c_dec, c_2, c_3, c_4, c_5, c_6): 
            FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,MJD,ECL_LAT,MOON_PHASE,MOON_ILL,MOON_ALT,MOON_SEP,SUN_SEP,SUN_ALT,AIRMASS,ALT,ZODI,ISL,H2,H3,H4,H5,H6 = x

            solarF = c_sol*self.solar_flux(MJD-I)
            zodi = c_z*ZODI

            airmass = c_am*AIRMASS
            tput = self.add_ext(AIRMASS,tau)

            months =  c_feb*FEB+c_mar*MAR+c_apr*APR+c_may*MAY+c_jun*JUN+c_jul*JUL+c_sep*SEP+c_oct*OCT+c_nov*NOV+c_dec*DEC
            hours = c_2*H2 + c_3*H3 + c_4*H4 + c_5*H5 + c_6*H6

            twi = (T0*np.abs(SUN_ALT) + T1*(np.abs(SUN_ALT))**2 +  T2*np.abs(SUN_SEP) **2 + T3*np.abs(SUN_SEP)) * np.exp(-T4*AIRMASS)    # +np.abs(130 - SUN_SEP) 

            ALB = self.albedo(MOON_PHASE)
            moon = (M0 * MOON_ALT**2 + M1 * MOON_ALT + M2 * MOON_ILL**2 + M3 * MOON_ILL + M4 * MOON_SEP**2 + M5 * MOON_SEP ) * np.exp(-M6*AIRMASS) * ALB 

            model = (c_0 + solarF +  months + hours + airmass + zodi + c_i*ISL)*tput + moon
            return model


        mod = Model(moon_cont_model)

        params = mod.make_params()
        params.add('c_0', value = self.c_0, vary = True)
        params.add('M0', value = 1)
        params.add('M1', value = 1)
        params.add('M2', value = 1)
        params.add('M3', value = 1)
        params.add('M4', value = 1)
        params.add('M5', value = 1)
        params.add('M6', value = 1)

        params.add('I', value = self.i, vary = False)
        params.add('c_am', value = self.c_am, vary = False)
        params.add('tau', value = self.tau, vary = False)

        params.add('c_z', value = self.c_zodi, vary = False)
        params.add('c_i', value = self.c_isl, vary = False)
        params.add('c_sol', value = self.sol, vary = False)
        params.add('c_feb', value = self.feb, vary = False)
        params.add('c_mar', value = self.mar, vary = False)
        params.add('c_apr', value = self.apr, vary = False)
        params.add('c_may', value = self.may, vary = False)
        params.add('c_jun', value = self.jun, vary = False)
        params.add('c_jul', value = self.jul, vary = False)
        params.add('c_aug', value = self.aug, vary = False)
        params.add('c_sep', value = self.sep, vary = False)
        params.add('c_oct', value = self.oct, vary = False)
        params.add('c_nov', value = self.nov, vary = False)
        params.add('c_dec', value = self.dec, vary = False)
        params.add('c_2', value = self.c2, vary = False)
        params.add('c_3', value = self.c3, vary = False)
        params.add('c_4', value = self.c4, vary = False)
        params.add('c_5', value = self.c5, vary = False)
        params.add('c_6', value = self.c6, vary = False)
        params.add('T0', value = self.t0, vary = False)
        params.add('T1', value = self.t1, vary = False)
        params.add('T2', value = self.t2, vary = False)
        params.add('T3', value = self.t3, vary = False)
        params.add('T4', value = self.t4, vary = False)

        #define data
        moon_idx = np.where(self.Meta['SUN_ALT']<-18)
        self.ThisMeta = self.Meta[moon_idx]
        self.ThisData = self.data[moon_idx]
        self.clean_data()
        self.color_cut()
        #self.get_photo_data()
        self.create_feature_list()
        y = self.ThisData
        X = self.Features

        self.moon_result = mod.fit(y, params, x =  np.array(X).T)
        self.moon_resids = y - self.moon_result.best_fit

        self.model_name = 'moon'
        self.data_var = np.std(y)
        self.unexplained_var = np.std(self.moon_resids)
        self.chisqr = self.moon_result.chisqr
        self.redchisqr = self.moon_result.redchi

        self.c_0 = self.moon_result.params['c_0'].value
        self.m0 = self.moon_result.params['M0'].value
        self.m1 = self.moon_result.params['M1'].value
        self.m2 = self.moon_result.params['M2'].value
        self.m3 = self.moon_result.params['M3'].value
        self.m4 = self.moon_result.params['M4'].value
        self.m5 = self.moon_result.params['M5'].value
        self.m6 = self.moon_result.params['M6'].value 

      


if __name__ == '__main__':

    Meta = astropy.table.Table.read('/Users/parkerf/Research/SkyModel/BOSS_Sky/good_mean_meta_071718.fits')
    yy = Meta['cont_b_460']
    wl = 460
    TM = ContModel(yy, wl, 'blue', Meta)
    TM.run_dark_model()
    sys.exit()

    RR = []
    for item in ['cont_b_380','cont_b_410','cont_b_425','cont_b_460','cont_b_480','cont_b_510','cont_b_565','cont_b_540','cont_b_602','cont_b_615','cont_b_583']:
        yy = Meta[item]
        wl = int(item[-3:])
        TM = ContModel(yy, wl, 'blue', Meta)
        TM.run_dark_model()
        TM.run_full_model()

        Results = [wl,TM.dark_chisqr,TM.dark_data_var,TM.dark_var,TM.dark_improve,TM.chisqr,TM.data_var,TM.var,TM.improve,
    TM.c_0,TM.c_am,TM.c_zodi,TM.za,TM.c_isl,TM.sol,TM.i,TM.t0,TM.t1,TM.t2,TM.t3,TM.m0,TM.m1,TM.m2,TM.m3,TM.m4,TM.m5,
    TM.feb, TM.mar,TM.apr,TM.may,TM.jun,TM.jul,TM.aug,TM.sep,TM.oct,TM.nov,TM.dec]

        RR.append(Results)

    for item in ['cont_r_833','cont_r_720','cont_r_740','cont_r_710','cont_r_675','cont_r_977','cont_r_642','cont_r_873','cont_r_920','cont_r_825']:
        yy = Meta[item]
        wl = int(item[-3:])
        TM = ContModel(yy, wl, 'red', Meta)
        TM.run_dark_model()
        TM.run_full_model()

        Results = [wl,TM.dark_chisqr,TM.dark_data_var,TM.dark_var,TM.dark_improve,TM.chisqr,TM.data_var,TM.var,TM.improve,
    TM.c_0,TM.c_am,TM.c_zodi,TM.za,TM.c_isl,TM.sol,TM.i,TM.t0,TM.t1,TM.t2,TM.t3,TM.m0,TM.m1,TM.m2,TM.m3,TM.m4,TM.m5,
    TM.feb, TM.mar,TM.apr,TM.may,TM.jun,TM.jul,TM.aug,TM.sep,TM.oct,TM.nov,TM.dec]
        RR.append(Results)

    C = ['wl','dX2','d_dataVar','dVar','d_imp','X2','dataVar','Var','imp','c0','c_am','c_zodi','za','c_isl','sol','I',
    't0','t1','t2','t3','m0','m1','m2','m3','m4','m5','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

    df = pd.DataFrame(np.array(RR))
    df.columns = C
    T = astropy.table.Table.from_pandas(df)
    print(T)
    T.write("output/model_fit_a.fits",format='fits')
