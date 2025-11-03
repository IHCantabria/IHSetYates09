import numpy as np
from .yates09 import yates09
from IHSetUtils.CoastlineModel import CoastlineModel

class assimilate_Yates09(CoastlineModel):
    """
    Shoreline model Yates et al. (2009).
    """
    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Yates et al. (2009)',
            mode='assimilation',
            model_type='CS',
            model_key='Yates09'
        )
        self.setup_forcing()

    def setup_forcing(self):
        self.E = self.hs ** 2
        self.E_s = self.hs_s ** 2
        self.Yini = self.Obs_splited[0]
        self.y_old = self.Yini

    def init_par(self, population_size: int):
        lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
        uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])

        pop = np.zeros((population_size, len(lowers)))
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers

    def model_step(self, par: np.ndarray, t_idx: int) -> np.ndarray:
        a = -np.exp(par[0]); b = par[1]
        cacr = -np.exp(par[2]); cero = -np.exp(par[3])
        idx = self.idx_obs_splited[t_idx-1:t_idx]
        Ymd, _ = yates09(self.E_s[idx], self.dt_s[idx], a, b, cacr, cero, self.y_old)
        self.y_old = Ymd[-1]
        return Ymd[-1]

    def run_model(self, par: np.ndarray) -> np.ndarray:
        a = par[0]; b = par[1]
        cacr = par[2]; cero = par[3]
        Yini = self.Yini
        Ymd, _ = yates09(self.E, self.dt, a, b, cacr, cero, Yini)
        return Ymd

    def _set_parameter_names(self):
        self.par_names = ['a', 'b', 'C+', 'C-']
        for idx in [0, 2, 3]:
            self.par_values[idx] = -np.exp(self.par_values[idx])

        