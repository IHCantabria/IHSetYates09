import numpy as np
import xarray as xr
from datetime import datetime
from spotpy.parameter import Uniform
from .yates09 import yates09
from IHSetCalibration import objective_functions

class cal_Yates09(object):
    """
    cal_Yates09
    
    Configuration to calibrate and run the Yates et al. (2009) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        
        
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

        cfg = xr.open_dataset(path+'config.nc')
        wav = xr.open_dataset(path+'wav.nc')
        ens = xr.open_dataset(path+'ens.nc')

        self.cal_alg = cfg['cal_alg'].values
        self.metrics = cfg['metrics'].values
        self.dt = cfg['dt'].values
        self.switch_Yini = cfg['switch_Yini'].values

        if self.cal_alg == 'NSGAII':
            self.n_pop = cfg['n_pop'].values
            self.generations = cfg['generations'].values
            self.n_obj = cfg['n_obj'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, n_pop=self.n_pop, generations=self.generations, n_obj=self.n_obj)
        else:
            self.repetitions = cfg['repetitions'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, repetitions=self.repetitions)

        self.Hs = wav['Hs'].values
        self.Tp = wav['Tp'].values
        self.Dir = wav['Dir'].values
        self.time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)
        self.E = self.Hs ** 2

        self.Obs = ens['Obs'].values
        self.time_obs = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

        self.start_date = datetime(int(cfg['Ysi'].values), int(cfg['Msi'].values), int(cfg['Dsi'].values))
        self.end_date = datetime(int(cfg['Ysf'].values), int(cfg['Msf'].values), int(cfg['Dsf'].values))

        self.split_data()

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]

        cfg.close()
        wav.close()
        ens.close()
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        if self.switch_Yini == 0:
            def model_simulation(par):
                a = -par['a']
                b = par['b']
                cacr = -par['cacr']
                cero = -par['cero']

                Ymd, _ = yates09(self.E_splited,
                                 self.dt,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 self.Yini)
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('a', 1e-3, 2),
                Uniform('b', 1e-1, 1e+3),
                Uniform('cacr', 1e-5, 6e-1),
                Uniform('cero', 1e-5, 6e-1)
            ]
            self.model_sim = model_simulation

        elif self.switch_Yini == 1:
            def model_simulation(par):
                a = -par['a']
                b = par['b']
                cacr = -par['cacr']
                cero = -par['cero']
                Yini = par['Yini']

                Ymd, _ = yates09(self.E_splited,
                                 self.dt,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 Yini)
            
                return Ymd[self.idx_obs_splited]
            self.params = [
                Uniform('a', 1e-3, 2),
                Uniform('b', 1e-1, 1e+3),
                Uniform('cacr', 1e-5, 6e-1),
                Uniform('cero', 1e-5, 6e-1),
                Uniform('Yini', np.min(self.Obs), np.max(self.Obs))
            ]
            self.model_sim = model_simulation

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
        self.idx_calibration = idx
        self.E_splited = self.E[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))
        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited

        # Validation    
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))
        self.idx_validation_obs = idx
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
        self.idx_validation_for_obs = mkIdx(self.time_obs[idx])


        


