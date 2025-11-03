import numpy as np
from .yates09 import yates09
from IHSetUtils.CoastlineModel import CoastlineModel
from typing import Any

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

    def init_par(self, population_size: int):
        lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
        uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])

        pop = np.zeros((population_size, len(lowers)))
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers

    def model_step(self, par: np.ndarray, t_idx: int, context: Any | None = None) -> np.ndarray:
        a = -np.exp(par[0]); b = par[1]
        cacr = -np.exp(par[2]); cero = -np.exp(par[3])
        idx = self.idx_obs_splited[t_idx-1:t_idx]
        if context is None or ('y_old' not in context):
            y0 = float(self.Yini)   # first step starts from initial shoreline
        else:
            y0 = float(context['y_old'])
        Ymd, _ = yates09(self.E_s[idx], self.dt_s[idx], a, b, cacr, cero, y0)
        context = {'y_old': Ymd[-1]}
        return Ymd[-1], context
    
    def model_step_batch(self,
                        pop: np.ndarray,             # (N, D)
                        t_idx: int,
                        contexts: list[dict] | None  # len N
                        ):
        N = pop.shape[0]
        y_out   = np.empty((N,), dtype=float)
        new_ctx = [None] * N

        i0, i1 = self.idx_obs_splited[t_idx-1], self.idx_obs_splited[t_idx]
        E_seg  = self.E_s[i0:i1]
        dt_seg = self.dt_s[i0:i1]

        for j in range(N):
            par = pop[j]
            a    = -np.exp(par[0])
            b    =  par[1]
            cacr = -np.exp(par[2])
            cero = -np.exp(par[3])

            y0 = float(self.Yini) if (contexts is None or contexts[j] is None
                                    or 'y_old' not in contexts[j]) else float(contexts[j]['y_old'])

            Ymd, _ = yates09(E_seg, dt_seg, a, b, cacr, cero, y0)
            y_last = float(Ymd[-1])

            y_out[j]   = y_last
            new_ctx[j] = {'y_old': y_last}

        # EnKF expects (N, p). For scalar p=1, return shape (N, 1) or (N,) is okay
        return y_out, new_ctx

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

        