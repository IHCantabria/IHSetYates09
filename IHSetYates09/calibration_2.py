import numpy as np
from .yates09 import yates09
from IHSetUtils.CoastlineModel import CoastlineModel

class cal_Yates09_3(CoastlineModel):
    """
    Shoreline model Yates et al. (2009).
    """
    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Yates et al. (2009)',
            mode='calibration',
            model_type='CS',
            model_key='Yates09'
        )
        self.switch_Yini = self.cfg['switch_Yini']
        self.setup_forcing()

    def setup_forcing(self):
        self.E = self.hs ** 2
        self.E_s = self.hs_s ** 2
        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]

    def init_par(self, population_size: int):
        if self.switch_Yini == 0:
            lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
            uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])
        else:
            lowers = np.array([
                np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]),
                np.log(self.lb[3]), 0.75 * np.min(self.Obs_splited)
            ])
            uppers = np.array([
                np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]),
                np.log(self.ub[3]), 1.25 * np.max(self.Obs_splited)
            ])
        pop = np.zeros((population_size, len(lowers)))
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers

    def model_sim(self, par: np.ndarray) -> np.ndarray:
        if self.switch_Yini == 0:
            a = -np.exp(par[0]); b = par[1]
            cacr = -np.exp(par[2]); cero = -np.exp(par[3])
            Ymd, _ = yates09(self.E_s, self.dt_s, a, b, cacr, cero, self.Yini)
        else:
            a = -np.exp(par[0]); b = par[1]
            cacr = -np.exp(par[2]); cero = -np.exp(par[3]); Yini = par[4]
            Ymd, _ = yates09(self.E_s, self.dt_s, a, b, cacr, cero, Yini)
        return Ymd[self.idx_obs_splited]

    def run_model(self, par: np.ndarray) -> np.ndarray:
        if self.switch_Yini == 0:
            a = par[0]; b = par[1]
            cacr = par[2]; cero = par[3]
            Ymd, _ = yates09(self.E, self.dt, a, b, cacr, cero, self.Yini)
        else:
            a = par[0]; b = par[1]
            cacr = par[2]; cero = par[3]; Yini = par[4]
            Ymd, _ = yates09(self.E, self.dt, a, b, cacr, cero, Yini)
        return Ymd

    def _set_parameter_names(self):
        if self.switch_Yini == 1:
            self.par_names = ['a', 'b', 'C+', 'C-', 'Y_i']
        else:
            self.par_names = ['a', 'b', 'C+', 'C-']
        for idx in [0, 2, 3]:
            self.par_values[idx] = -np.exp(self.par_values[idx])

        