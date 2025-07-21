import numpy as np
from .yates09 import yates09
from IHSetUtils import CoastlineModel

class cal_Yates09_3(CoastlineModel):
    """
    cal_Yates09
    
    Configuration to calibrate and run the Yates et al. (2009) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        super().__init__(
            path = path,
            model_name = 'Yates et al. (2009)',
            mode = 'calibration',
            type = 'CS',
            model_key = 'Yates09',
        )

        self.switch_Yini = self.cfg['switch_Yini']

        self.setup_forcing()

    def setup_forcing(self):
        """
        Set up the forcing data for the model.
        """

        self.E = self.hs ** 2
        self.E_s = self.hs_s ** 2

        if self.switch_Yini == 0:
            self.Yini = self.Obs_s[0]

    def _simulation(self):
        """
        Simulate the model based on the parameters.
        """

        if self.switch_Yini == 0:
            # @jit
            def model_simulation(par):
                a = -np.exp(par[0])
                b = par[1]
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
                Ymd, _ = yates09(self.E_s,
                                 self.dt_s,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 self.Yini)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def run_model(par):
                a = -np.exp(par[0])
                b = par[1]
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

            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
                log_upper_bounds = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])
                population = np.zeros((population_size, 4))
                for i in range(4):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1:
            def model_simulation(par):
                a = -np.exp(par[0])
                b = par[1]
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
                Yini = par[4]

                Ymd, _ = yates09(self.E_splited,
                                 self.dt_splited,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 Yini)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def run_model(par):
                a = -np.exp(par[0])
                b = par[1]
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

            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3]), 0.75*np.min(self.Obs_splited)])
                log_upper_bounds = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3]), 1.25*np.max(self.Obs_splited)])
                population = np.zeros((population_size, 5))
                for i in range(5):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

    def calibrate(self):
        """
        Calibrate the model.
        """
        self.solution, self.objectives, self.hist = self.calibr_cfg.calibrate(self)

        self.full_run = self.run_model(self.solution)

        if self.switch_Yini == 1:
            self.par_names = [r'a', r'b', r'C+', r'C-', r'Y_i']
            self.par_values = self.solution.copy()
            self.par_values[0] = -np.exp(self.par_values[0])
            self.par_values[2] = -np.exp(self.par_values[2])
            self.par_values[3] = -np.exp(self.par_values[3])
        elif self.switch_Yini == 0:
            self.par_names = [r'a', r'b', r'C+', r'C-']
            self.par_values = self.solution.copy()
            self.par_values[0] = -np.exp(self.par_values[0])
            self.par_values[2] = -np.exp(self.par_values[2])
            self.par_values[3] = -np.exp(self.par_values[3])
