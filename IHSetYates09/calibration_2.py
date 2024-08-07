import numpy as np
import xarray as xr
from datetime import datetime
import fast_optimization as fo
from .yates09 import yates09
from numba import jit

class cal_Yates09_2(object):
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
        self.lb = cfg['lb'].values
        self.ub = cfg['ub'].values

        if self.cal_alg == 'NSGAII':
            self.num_generations = cfg['num_generations'].values
            self.population_size = cfg['population_size'].values
            self.cross_prob = cfg['cross_prob'].values
            self.mutation_rate = cfg['mutation_rate'].values
            self.regeneration_rate = cfg['regeneration_rate'].values
            self.objective_function = fo.multi_obj_func(self.metrics)
        elif self.cal_alg == 'NSGAII-ts': 
            self.num_generations = cfg['num_generations'].values
            self.population_size = cfg['population_size'].values
            self.pressure = cfg['pressure'].values
            self.regeneration_rate = cfg['regeneration_rate'].values
            self.objective_function = fo.multi_obj_func(self.metrics)
        elif self.cal_alg == 'SPEA2':
            self.num_generations = cfg['num_generations'].values
            self.population_size = cfg['population_size'].values
            self.pressure = cfg['pressure'].values
            self.regeneration_rate = cfg['regeneration_rate'].values
            self.cross_prob = cfg['cross_prob'].values
            self.mutation_rate = cfg['mutation_rate'].values
            self.mutation_variance = cfg['mutation_variance'].values
            self.objective_function = fo.multi_obj_func(self.metrics)

        self.Hs = wav['Hs'].values
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
            @jit(nopython=True)
            def model_simulation(par):
                a = -1 * (10 ** par[0])
                b = 10 ** par[1]
                cacr = -1 * (10 ** np.exp(par[2]))
                cero = -1 * (10 ** np.exp(par[3]))
                Ymd, _ = yates09(self.E_splited,
                                 self.dt,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 self.Yini)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            @jit(nopython=True)
            def init_par(population_size):
                log_lower_bounds = np.array([np.log10(self.lb[0]), np.log10(self.lb[1]), np.log10(self.lb[2]), np.log10(self.lb[3])])
                log_upper_bounds = np.array([np.log10(self.ub[0]), np.log10(self.ub[1]), np.log10(self.ub[2]), np.log10(self.ub[3])])
                population = np.zeros((population_size, 4))
                for i in range(4):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1:
            @jit(nopython=True)
            def model_simulation(par):
                a = -1 * (10 ** par[0])
                b = 10 ** par[1]
                cacr = -1 * (10 ** par[2])
                cero = -1 * (10 ** par[3])
                Yini = par[4]

                Ymd, _ = yates09(self.E_splited,
                                 self.dt,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 Yini)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            @jit(nopython=True)
            def init_par(population_size):
                log_lower_bounds = np.array([np.log10(self.lb[0]), np.log10(self.lb[1]), np.log10(self.lb[2]), np.log10(self.lb[3]), 0.75*np.min(self.Obs_splited)])
                log_upper_bounds = np.array([np.log10(self.ub[0]), np.log10(self.ub[1]), np.log10(self.ub[2]), np.log10(self.ub[3]), 1.25*np.max(self.Obs_splited)])
                population = np.zeros((population_size, 5))
                for i in range(5):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        ii = np.where(self.time>=self.start_date)[0][0]
        self.E = self.E[ii:]
        self.time = self.time[ii:]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))[0]
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.idx_calibration = idx
        self.E_splited = self.E[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))[0]
        self.idx_validation_obs = idx
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

    def calibrate(self):
        """
        Calibrate the model.
        """
        if self.cal_alg == 'NSGAII':
            self.population, self.objectives = fo.nsgaii_algorithm(self.objective_function, self.model_simulation, self.Obs_splited, self.initialize_population, self.num_generations, self.population_size, self.cross_prob, self.mutation_rate, self.regeneration_rate)
        elif self.cal_alg == 'NSGAII-ts':
            self.population, self.objectives = fo.nsgaii_algorithm_ts(self.objective_function, self.model_simulation, self.Obs_splited, self.initialize_population, self.num_generations, self.population_size, self.pressure, self.regeneration_rate)
        elif self.cal_alg == 'SPEA2':
            self.population, self.objectives = fo.spea2_algorithm(self.objective_function, self.model_simulation, self.Obs_splited, self.initialize_population, self.num_generations, self.population_size, self.pressure, self.regeneration_rate, self.cross_prob, self.mutation_rate, self.mutation_variance)
        


