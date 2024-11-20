import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .yates09 import yates09
import json

class cal_Yates09_2(object):
    """
    cal_Yates09
    
    Configuration to calibrate and run the Yates et al. (2009) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['Yates09'])

        self.cal_alg = cfg['cal_alg']
        self.metrics = cfg['metrics']
        self.dt = cfg['dt']
        self.switch_Yini = cfg['switch_Yini']
        self.lb = cfg['lb']
        self.ub = cfg['ub']

        if self.cal_alg == 'NSGAII': 
            self.num_generations = cfg['num_generations']
            self.population_size = cfg['population_size']
            self.cross_prob = cfg['cross_prob']
            self.mutation_rate = cfg['mutation_rate']
            self.pressure = cfg['pressure']
            self.regeneration_rate = cfg['regeneration_rate']
            self.kstop = cfg['kstop']
            self.pcento = cfg['pcento']
            self.peps = cfg['peps']
            self.indexes = fo.multi_obj_indexes(self.metrics)
        elif self.cal_alg == 'SPEA2':
            self.num_generations = cfg['num_generations']
            self.population_size = cfg['population_size']
            self.pressure = cfg['pressure']
            self.regeneration_rate = cfg['regeneration_rate']
            self.cross_prob = cfg['cross_prob']
            self.mutation_rate = cfg['mutation_rate']
            self.m = cfg['m']
            self.eta_mut = cfg['eta_mut']
            self.kstop = cfg['kstop']
            self.pcento = cfg['pcento']
            self.peps = cfg['peps']
            self.indexes = fo.multi_obj_indexes(self.metrics)
        elif self.cal_alg == 'SCE-UA':
            self.num_generations = cfg['num_generations']
            self.population_size = cfg['population_size']
            # self.magnitude = cfg['magnitude']
            self.cross_prob = cfg['cross_prob']
            self.mutation_rate = cfg['mutation_rate']
            self.regeneration_rate = cfg['regeneration_rate']
            self.eta_mut = cfg['eta_mut']
            self.num_complexes = cfg['num_complexes']
            self.kstop = cfg['kstop']
            self.pcento = cfg['pcento']
            self.peps = cfg['peps']
            self.indexes = fo.multi_obj_indexes(self.metrics)
        elif self.cal_alg == 'Simulated Annealing':
            self.max_iterations = cfg['max_iterations']
            self.initial_temperature = cfg['initial_temperature']
            self.cooling_rate = cfg['cooling_rate']
            self.indexes = fo.multi_obj_indexes(self.metrics)
            

        if cfg['trs'] == 'Average':
            self.hs = np.mean(data.hs.values, axis=1)
            self.time = pd.to_datetime(data.time.values)
            self.E = self.hs ** 2
            self.Obs = data.average_obs.values
            self.Obs = self.Obs[~data.mask_nan_average_obs]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_average_obs]
        else:
            self.hs = data.hs.values[:, cfg['trs']]
            self.time = pd.to_datetime(data.time.values)
            self.E = self.hs ** 2
            self.Obs = data.obs.values[:, cfg['trs']]
            self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]
        
        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])
        
        data.close()

        self.split_data()

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]


        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

        
        self.idx_obs = mkIdx(self.time_obs)

        if self.switch_Yini == 0:
            # @jit
            def model_simulation(par):
                a = -np.exp(par[0])
                b = np.exp(par[1])
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
                Ymd, _ = yates09(self.E_splited,
                                 self.dt,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 self.Yini)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def run_model(par):
                a = -np.exp(par[0])
                b = np.exp(par[1])
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
                Ymd, _ = yates09(self.E,
                                 self.dt,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 self.Yini)
                return Ymd

            self.run_model = run_model

            # @jit
            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), np.log(self.lb[1]), np.log(self.lb[2]), np.log(self.lb[3])])
                log_upper_bounds = np.array([np.log(self.ub[0]), np.log(self.ub[1]), np.log(self.ub[2]), np.log(self.ub[3])])
                population = np.zeros((population_size, 4))
                for i in range(4):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1:
            # @jit
            def model_simulation(par):
                a = -np.exp(par[0])
                b = np.exp(par[1])
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
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

            def run_model(par):
                a = -np.exp(par[0])
                b = np.exp(par[1])
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
                Yini = par[4]

                Ymd, _ = yates09(self.E,
                                 self.dt,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 Yini)
                return Ymd

            self.run_model = run_model


            # @jit
            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), np.log(self.lb[1]), np.log(self.lb[2]), np.log(self.lb[3]), 0.75*np.min(self.Obs_splited)])
                log_upper_bounds = np.array([np.log(self.ub[0]), np.log(self.ub[1]), np.log(self.ub[2]), np.log(self.ub[3]), 1.25*np.max(self.Obs_splited)])
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
            self.solution, self.objectives, self.hist = fo.nsgaii_algorithm_ts(
                self.model_sim, 
                self.Obs_splited, 
                self.init_par, 
                self.num_generations, 
                self.population_size,
                self.cross_prob, 
                self.mutation_rate, 
                self.pressure, 
                self.regeneration_rate,
                self.kstop,
                self.pcento,
                self.peps,
                self.indexes
                )
        elif self.cal_alg == 'SPEA2':
            self.solution, self.objectives, self.hist = fo.spea2_algorithm(
                self.model_sim, 
                self.Obs_splited, 
                self.init_par, 
                self.num_generations, 
                self.population_size, 
                self.cross_prob, 
                self.mutation_rate, 
                self.pressure, 
                self.regeneration_rate,
                self.m,
                self.eta_mut,
                self.kstop,
                self.pcento,
                self.peps,
                self.indexes)
        elif self.cal_alg == 'SCE-UA':
            self.solution, self.objectives, self.hist = fo.sce_ua_algorithm(
                self.model_sim, 
                self.Obs_splited, 
                self.init_par, 
                self.num_generations, 
                self.population_size,
                self.cross_prob,
                self.mutation_rate,
                self.regeneration_rate,
                self.eta_mut,
                self.num_complexes,
                self.kstop,
                self.pcento,
                self.peps,
                self.indexes)
        elif self.cal_alg == 'Simulated Annealing':
            self.solution, self.objectives, self.hist = fo.simulated_annealing(
                self.model_sim, 
                self.Obs_splited, 
                self.init_par, 
                self.max_iterations, 
                self.initial_temperature, 
                self.cooling_rate,
                self.indexes)
            