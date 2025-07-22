# import numpy as np
# import xarray as xr
# import pandas as pd
# import fast_optimization as fo
# from .yates09 import yates09_njit
# import json

# class Yates09_run(object):
#     """
#     Yates09_run
    
#     Configuration to calibrate and run the Yates et al. (2009) Shoreline Evolution Model.
    
#     This class reads input datasets, performs its calibration.
#     """

#     def __init__(self, path):

#         self.path = path
#         self.name = 'Yates et al. (2009)'
#         self.mode = 'standalone'
#         self.type = 'CS'
     
#         data = xr.open_dataset(path)
        
#         cfg = json.loads(data.attrs['run_Yates09'])
#         self.cfg = cfg

#         self.switch_Yini = cfg['switch_Yini']

#         if cfg['trs'] == 'Average':
#             self.hs = np.mean(data.hs.values, axis=1)
#             self.time = pd.to_datetime(data.time.values)
#             self.E = self.hs ** 2
#             self.Obs = data.average_obs.values
#             self.Obs = self.Obs[~data.mask_nan_average_obs]
#             self.time_obs = pd.to_datetime(data.time_obs.values)
#             self.time_obs = self.time_obs[~data.mask_nan_average_obs]
#         else:
#             self.hs = data.hs.values[:, cfg['trs']]
#             self.time = pd.to_datetime(data.time.values)
#             self.E = self.hs ** 2
#             self.Obs = data.obs.values[:, cfg['trs']]
#             self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
#             self.time_obs = pd.to_datetime(data.time_obs.values)
#             self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]


#         self.start_date = pd.to_datetime(cfg['start_date'])
#         self.end_date = pd.to_datetime(cfg['end_date'])
        
#         data.close()

#         self.split_data()

#         if self.switch_Yini == 1:
#             ii = np.argmin(np.abs(self.time_obs - self.time[0]))
#             self.Yini = self.Obs[ii]

#         mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        
#         self.idx_obs = mkIdx(self.time_obs)

#         # Now we calculate the dt from the time variable
#         mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
#         self.dt = mkDT(np.arange(0, len(self.time)-1))

#         if self.switch_Yini == 0:
#             def run_model(par):
#                 a = par[0]
#                 b = par[1]
#                 cacr = par[2]
#                 cero = par[3]
#                 Yini = par[4]

#                 Ymd, _ = yates09_njit(self.E,
#                                     self.dt,
#                                     a,
#                                     b,
#                                     cacr,
#                                     cero,
#                                     Yini)
#                 return Ymd
        
#             self.run_model = run_model
#         else:
#             def run_model(par):
#                 a = par[0]
#                 b = par[1]
#                 cacr = par[2]
#                 cero = par[3]

#                 Ymd, _ = yates09_njit(self.E,
#                                     self.dt,
#                                     a,
#                                     b,
#                                     cacr,
#                                     cero,
#                                     self.Yini)
#                 return Ymd
        
#             self.run_model = run_model
    
#     def run(self, par):
#         self.full_run = self.run_model(par)
#         if self.switch_Yini == 1:
#             self.par_names = [r'a', r'b', r'C+', r'C-']
#             self.par_values = par
#         elif self.switch_Yini == 0:
#             self.par_names = [r'a', r'b', r'C+', r'C-', r'Y_i']
#             self.par_values = par

#         # self.calculate_metrics()

#     def calculate_metrics(self):
#         self.metrics_names = fo.backtot()[0]
#         self.indexes = fo.multi_obj_indexes(self.metrics_names)
#         self.metrics = fo.multi_obj_func(self.Obs, self.full_run[self.idx_obs], self.indexes)

#     def split_data(self):
#         """
#         Split the data into calibration and validation datasets.
#         """
#         ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
#         self.E = self.E[ii]
#         self.time = self.time[ii]

#         ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
#         self.Obs = self.Obs[ii]
#         self.time_obs = self.time_obs[ii]

import numpy as np
from .yates09 import yates09_njit 
from IHSetUtils.CoastlineModel import CoastlineModel

class Yates09_run(CoastlineModel):
    """
    Yates09_run
    
    Configuration to calibrate and run the Yates et al. (2009) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Yates et al. (2009)',
            mode='standalone',
            model_type='CS',
            model_key='run_Yates09'
        )
        self.switch_Yini = self.cfg['switch_Yini']
        self.setup_forcing()

    def setup_forcing(self):
        self.E = self.hs ** 2
        if self.switch_Yini == 1:
            self.Yini = self.Obs[0]

    def run_model(self, par: np.ndarray) -> np.ndarray:
        if self.switch_Yini == 1:
            a = par[0]; b = par[1]
            cacr = par[2]; cero = par[3]
            Ymd, _ = yates09_njit(self.E, self.dt, a, b, cacr, cero, self.Yini)
        else:
            a = par[0]; b = par[1]
            cacr = par[2]; cero = par[3]; Yini = par[4]
            Ymd, _ = yates09_njit(self.E, self.dt, a, b, cacr, cero, Yini)
        return Ymd
    
    def _set_parameter_names(self):
        if self.switch_Yini == 1:
            self.par_names = [r'a', r'b', r'C+', r'C-']
        elif self.switch_Yini == 0:
            self.par_names = [r'a', r'b', r'C+', r'C-', r'Y_i']
