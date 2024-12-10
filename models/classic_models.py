from scipy.stats import gamma
from scipy.special import beta
from scipy.optimize import minimize
import numpy as np


class CSGD():
    def __init__(self, verbose = True) -> None:
        self.verbose = verbose

    def calc_initial_values(self,ground_truth, method = 'scipy'):
        """
        Calculate initial values for optimization.

        Parameters:
        - ground_truth: np.ndarray, array of ground truth values
        - method: str, 'scipy' or 'paper' to determine initialization method

        Returns:
        - list of initial values [mean, std_dev, shift]
        """
        if method == 'paper':
            # Calculamos el valor inicial para m y s asumiendo distribución exponencial
            ppop = np.mean(ground_truth > 0)
            #print(ppop)
            if ppop < 0.005:
                # Valores ad hoc para puntos de rejilla extremadamente secos
                m, s, d = 0.0005, 0.0182, 0.00049
            else:
                m = np.mean(ground_truth[ground_truth > 0])
                s = m.copy() #Como m = s, => k = 1.0. Distribución exponencial inicialmente
                k = 1.
                d = -m * np.log(ppop)
                
                restr_check = (m - d) > 0
                #print('Cumple la restricción cuarta?',restr_check)
                # Ajuste para cumplir con la restricción d <= m/2
                
                while (d >= 0.9*m):
                    m *= 0.9  # Reducir m gradualmente
                    d = + m / (k * gamma.ppf(1 - ppop, k))
                    k = m**2/s**2

            return [m, s, d]
        
        elif method == 'scipy':
            shape_0, shift_neg_0, scale_0 = gamma._fitstart(ground_truth)
            m = shape_0 * scale_0
            s = np.sqrt(shape_0)*scale_0
            d = -shift_neg_0

            return [m, s, d]
            

    def crps(self, ground_truth: np.ndarray, params:list = None) -> float:
        """
        Calculate the CRPS value.

        Parameters:
        - ground_truth: np.ndarray, array of ground truth values
        - params: list, [mean, std_dev, shift]. If None, initial values are calculated using Scipy method.

        Returns:
        - float, mean CRPS value
        """
        if params is None:
            print('Calculating initial values') if self.verbose else None
            mean, std_dev, shift = self.calc_initial_values(ground_truth)
        else:
            mean, std_dev, shift = params

        #loss = np.zeros(len(ground_truth))
        shape = (mean**2) / (std_dev**2)
        scale = (std_dev**2) / (mean)
        target = ground_truth
        sum1 = (target + shift) * ( 2*gamma.cdf(target + shift, shape, scale = scale) - 1)
        sum2 = - (shape * scale / np.pi) * beta(0.5, shape + 0.5) * (1 - gamma.cdf(+2 * shift, 2 * shape, scale = scale))
        sum3 = scale * shape * (1 + 2*gamma.cdf(+shift,shape,scale = scale)*gamma.cdf(+shift,shape + 1,scale = scale) - gamma.cdf(+shift,shape,scale = scale)**2 - 2*gamma.cdf(target+shift,shape + 1, scale = scale))
        sum4 = -shift * gamma.cdf(+shift,shape, scale = scale)**2
        loss = (sum1 + sum2 + sum3 + sum4)

        return np.mean(loss)
    
    def mse(self, ground_truth: np.ndarray, params:list = None) -> float:
        if params is None:
            print('Calculating initial values') if self.verbose else None
            mean, std_dev, shift = self.calc_initial_values(ground_truth)
        else:
            mean, std_dev, shift = params

        shape = (mean**2) / (std_dev**2)
        scale = (std_dev**2) / (mean)
        
        cdf_gamma = gamma.cdf(shift, shape, scale = scale)
        expected_value = mean * (1 - cdf_gamma) * (1 - gamma.cdf(shift, shape + 1, scale = scale)) - shift * (1 - cdf_gamma)**2
        #expected_value = np.round(expected_value, decimals = 1)
        mse =  (expected_value - ground_truth)**2
        
        return np.mean(mse)
    
    def rmse(self, ground_truth: np.ndarray, params:list = None) -> float:
        if params is None:
            print('Calculating initial values') if self.verbose else None
            mean, std_dev, shift = self.calc_initial_values(ground_truth)
        else:
            mean, std_dev, shift = params
        #print(f'mean {mean} std {std_dev} shift {shift}')
        shape = (mean**2) / (std_dev**2)
        scale = (std_dev**2) / (mean)
        
        cdf_gamma = gamma.cdf(shift, shape, scale = scale)
        expected_value = mean * (1 - cdf_gamma) * (1 - gamma.cdf(shift, shape + 1, scale = scale)) - shift * (1 - cdf_gamma)**2
        #expected_value = np.round(expected_value,decimals = 2)
        #print(f'Expected value: {expected_value}')
        #print(f'shape {shape} shift {shift} scale {scale} mean {mean}')
        #print(f'Expected value v2: {gamma.mean(shape, loc = shift, scale = scale)}')
        #expected_value = np.round(expected_value, decimals = 1)
        rmse =  np.sqrt((expected_value - ground_truth)**2)
        
        return np.mean(rmse)
    
    def brier_score(self, ground_truth: np.ndarray, params:list = None, threshold = 1):
        if params is None:
            print('Calculating initial values') if self.verbose else None
            mean, std_dev, shift = self.calc_initial_values(ground_truth)
        else:
            mean, std_dev, shift = params
        
        shape = (mean**2) / (std_dev**2)
        scale = (std_dev**2) / (mean)

        target_binarized = ground_truth <= threshold
        cdf_gamma_at_thr = gamma.cdf(shift + threshold, shape, scale = scale)
        brier_score = (cdf_gamma_at_thr - target_binarized)**2
        
        return np.mean(brier_score)

    
    def _fit_climat(self, ground_truth: np.ndarray, init_values: list) -> minimize:
        """
        Fit the climatological distribution.

        Parameters:
        - ground_truth: np.ndarray, array of ground truth values
        - init_values: list, initial values for optimization

        Returns:
        - scipy.optimize.OptimizeResult, result of the optimization
        """
        constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},  # mean > 0
        {'type': 'ineq', 'fun': lambda x: x[1]},  # std_dev > 0
        {'type': 'ineq', 'fun': lambda x: x[2]},  # shift >= 0
        {'type': 'ineq', 'fun': lambda x: x[0] - x[2]}  #(mean - shift) >= 0
        ]
        crps_csgd_climat_optim = lambda params: self.crps(params = params, ground_truth = ground_truth)
        res = minimize(fun = crps_csgd_climat_optim, x0 = init_values,method = 'SLSQP', constraints = constraints, jac = '3-point') #trust-const
        return res
    
    def fit_climatological(self, ground_truth: np.ndarray, init_params: list = None, calc_init: str = None) -> list:
        """
        Fit climatological distribution minimizing CRPS. 
        self.param_values is NOT updated with new fitted values.

        Parameters:
        - ground_truth: np.ndarray, array of ground truth values
        - init_params: list, [mean_0, std_dev_0, shift_0]. If None, scipy and paper initialization methods are tried if calc_init not specificied.
        - calc_init: str, specify the method to initialize params. 'scipy' or 'paper' are the possible options.

        Returns:
        - list of fitted values [mean, std_dev, shift]
        """
        print('------Calculating climatological csgd----------------') if self.verbose else None
        if init_params is not None:
            init_values = init_params
            res = self._fit_climat(ground_truth, init_values = init_values)

        elif calc_init in ['paper','scipy']:
            init_values = self.calc_initial_values(ground_truth, calc_init)
            res = self._fit_climat(ground_truth, init_values = init_values)

        else:
            print('Calculating initial climat params')
            init_values_paper = self.calc_initial_values(ground_truth, 'paper')
            init_values_scipy = self.calc_initial_values(ground_truth, 'scipy')
            
            res_paper = self._fit_climat(ground_truth, init_values = init_values_paper)
            res_scipy = self._fit_climat(ground_truth, init_values = init_values_scipy)

            if res_paper.fun > res_scipy.fun:
                print('Scipy initialization is the best encountered') if self.verbose else None
                res = res_scipy
            else:
                print('Paper initialization is the best encountered') if self.verbose else None
                res = res_paper

        print(res) if self.verbose else None
        self.fitted_params = res.x
        return res.x
    
    def predict(self, ground_truth: np.ndarray, metric = 'crps') -> float:
        """
        Predict CRPS using fitted parameters.

        Parameters:
        - ground_truth: np.ndarray, array of ground truth values
        - metric: string, indicate 'crps' or 'all'. 'all' includes crps, rmse and brier score (in that order).

        Returns:
        - float, CRPS value
        """
        if metric == 'crps':
            return self.crps(ground_truth, self.fitted_params)
        elif metric == 'all':
            crps = self.crps(ground_truth, self.fitted_params)
            mse = self.mse(ground_truth, self.fitted_params)
            rmse = self.rmse(ground_truth, self.fitted_params)
            brier_score = self.brier_score(ground_truth, self.fitted_params)
            return  (crps, rmse, mse, brier_score)

class PredictiveCSGD():
    def __init__(self, params = None, forecast_climat_mean = None, climat_params = None, verbose = True) -> None:
        self.csgd = CSGD(verbose = verbose)
        self.verbose = verbose
        self.fitted_params = params
        self.fitted_climat_params = climat_params
        self.forecast_climat_mean = forecast_climat_mean
        self.csgd.fitted_params = climat_params

    def obtain_defining_parameters(self, ground_truth, wrf_ensemble,forecast_climat_mean = None, params = None, climat_params = None):
        a1, a2, a3, a4 = params if (params is not None) else [0.1,1,0.1,0.2]
        mean_climat, std_climat, shift_climat = self.csgd.fit_climatological(ground_truth) if (climat_params is None) else climat_params

        forecast_climat_mean =  np.mean(wrf_ensemble) if (forecast_climat_mean is None) else forecast_climat_mean #LO CAMBIÉ, EN VEZ DE GROUND TRUTH, PUSE WRF_ENSEMBLE
        #wrf_ensemble = np.round(wrf_ensemble,decimals=1)
        forecast_mean = np.mean(wrf_ensemble, axis = 1)
        #forecast_mean = np.round(forecast_mean, decimals=1)
        #forecast_mean[forecast_mean <= 0.05] = 0
        #assert len(forecast_mean) == len(ground_truth)
        mean = (mean_climat/a1) * np.log(1 + (np.exp(a1)-1)*(a2 + a3*(forecast_mean/forecast_climat_mean)) )
        std = a4 * std_climat * np.sqrt(mean/mean_climat)
        shift = shift_climat

        return mean, std, shift

    
    def crps(self, ground_truth, wrf_ensemble,forecast_climat_mean = None, params = None, climat_params = None):
        """
        Calculate the CRPS for the predictive model.

        Parameters:
        - ground_truth: np.ndarray [n,], array of ground truth values
        - wrf_ensemble: np.ndarray [n,k], ensemble forecast
        - forecast_climat_mean: float, climatological mean of the forecast
        - params: list, parameters [a1, a2, a3, a4]
        - climat_params: list, climatological parameters [mean, std_dev, shift]

        Returns:
        - float, CRPS value
        """
        mean, std, shift = self.obtain_defining_parameters(ground_truth, wrf_ensemble, forecast_climat_mean, params, climat_params)
        crps = self.csgd.crps(ground_truth = ground_truth, params = [mean, std, shift])
        #crps = crps_csgd(params = [mean,std,shift], ground_truth = ground_truth)
        return crps
    
    def fit(self,ground_truth, wrf_ensemble,forecast_climat_mean = None, params_0 = None, climat_params = None):
        """
        Fit [a1,a2,a3,a4] params minimizing crps of predictive CSGD.
        """
        constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 0.001},  # a1 > 0
        {'type': 'ineq', 'fun': lambda x: x[1] - 0.01 },  # a2 > 1
        {'type': 'ineq', 'fun': lambda x: x[2]},  # 1.5 - a3 >= 0
        {'type': 'ineq', 'fun': lambda x: x[3]- 0.001}  #a4 >= 0.1
        ]
        bounds = [(0,None) for _ in range(4)]
        if params_0 is None:
            # Define multiple sets of initial parameters to test
            #initial_params_list = [(l,m,n,p) for l in np.arange(0.01,0.35,0.05) for m in np.arange(0.01,0.8,0.05) for n in np.arange(0.7,2,0.2) for p in np.arange(0.1,1.1,0.1)]#[(l,m,n,p) for l in np.arange(0.01,0.35,0.05) for m in np.arange(0.01,0.8,0.05) for n in np.arange(0.1,2,0.2) for p in np.arange(0.1,0.5,0.05)]
             
            initial_params_list = [
                [0.01, 0.02, 1.5, 0.1],
                [0.2, 0.2, 0.2, 0.3],
                [0.05, 0.9, 0.05, 0.15],
                [0.15, 1.1, 0.15, 0.25],
                [0.12, 1.05, 0.12, 0.22],
                [0.08, 0.7, 1.08, 0.18],
                [0.2, 1.3, 0.2, 0.35],
                [0.1, 0.4, 1.1, 0.2],
                [0.3, 1.5, 0.3, 0.4],
                [0.25, 1.25, 0.25, 0.35]
            ]
            
        else:
            initial_params_list = [params_0]

        forecast_climat_mean =  np.mean(wrf_ensemble) if (forecast_climat_mean is None) else forecast_climat_mean #cambié de ground_truth a wrf_ensemble
        init_climat_params = self.csgd.calc_initial_values(ground_truth,'paper')
        climat_params = climat_params if (climat_params is not None) else self.csgd.fit_climatological(ground_truth,init_climat_params)
        
        best_params = None
        best_crps = float('inf')

        print('---------Calculating predictive csgd------------------') if self.verbose else None
        for params in initial_params_list:
            crps_predictive_CSGD_optim = lambda A_params: self.crps(ground_truth, wrf_ensemble, forecast_climat_mean, A_params, climat_params)
            res = minimize(fun=crps_predictive_CSGD_optim, x0=params, method='SLSQP', constraints=constraints, jac='3-point')#, options={'maxiter':200})
            #res = minimize(fun=crps_predictive_CSGD_optim, x0=params,method = 'TNC', bounds = bounds, jac='3-point')
            if res.fun < best_crps:
                best_crps = res.fun
                best_params = res.x
                best_res = res

        print(best_res) if self.verbose else None
        self.fitted_params = best_res.x
        self.fitted_climat_params = climat_params
        self.forecast_climat_mean = forecast_climat_mean
        return res.fun
    
    def predict(self,ground_truth, wrf_ensemble, metrics = 'crps'):
        if metrics == 'crps':
            return self.crps(ground_truth, wrf_ensemble, params = self.fitted_params, forecast_climat_mean = self.forecast_climat_mean, climat_params = self.fitted_climat_params)
        elif metrics == 'all':
            mean, std, shift = self.obtain_defining_parameters(ground_truth, wrf_ensemble, self.forecast_climat_mean, self.fitted_params, self.fitted_climat_params)
            crps = self.csgd.crps(ground_truth = ground_truth, params = [mean, std, shift])
            mse = self.csgd.mse(ground_truth, params = [mean, std, shift])
            rmse = self.csgd.rmse(ground_truth, params = [mean, std, shift])
            brier_score = self.csgd.brier_score(ground_truth, [mean, std, shift])
            return crps, rmse, mse, brier_score
        else:
            raise NotImplementedError
class ensembleCSGD():
    def __init__(self, verbose = True) -> None:
        self.csgd = CSGD(verbose = verbose)
        self.verbose = verbose
    
    def crps(self, ground_truth, wrf_ensemble, params = None):
        """
        Calculate the CRPS for the wrf model (Baran and Nemoda, 2016).

        Parameters:
        - ground_truth: np.ndarray [n,], array of ground truth values
        - wrf_ensemble: np.ndarray [n,k], ensemble forecast
        - params: list, parameters [a, b1, ..., bk, c, d, shift] used to calculate CSGD parameters.

        Returns:
        - float, CRPS value
        """
        n_members = wrf_ensemble.shape[1]
        a = params[0]
        B = np.array(params[1:n_members + 1])
        c = params[n_members + 1]
        d = params[n_members + 2]
        shift = params[n_members + 3]
        assert params[n_members + 3] == params[-1]

        mean = a + np.dot(wrf_ensemble, B)
        #print(f'mean {mean}')
        var = c + d * np.mean(wrf_ensemble, axis=1)

        crps = self.csgd.crps(ground_truth = ground_truth, params = [mean, np.sqrt(var), shift])

        return crps
    
    def fit(self,ground_truth, wrf_ensemble, params_0 = None):
        """
        Fit [a,B,c,d, shift] params minimizing crps of predictive CSGD. Initial values of the params are given by params_0.
        """
        n_members = wrf_ensemble.shape[1]
        bounds = [(1e-2,None) for _ in range(n_members + 4)]
        bounds[0] = (1e-7,None)   #a
        bounds[-2] = (0.1, None)  #d
        bounds[-3] = (0, 2)       #c
        bounds[-1] = (0,None)     #q
        """
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0] - 1e-7},  #a >= 1e-7
            {'type': 'ineq', 'fun': lambda x: x[-3]},  # c >= 0
            {'type': 'ineq', 'fun': lambda x: x[-2] - 0.15},  # d >= 0.15
            {'type': 'ineq', 'fun': lambda x: x[-1]}  # q >= 0
            #{'type': 'ineq', 'fun': lambda x: x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] - 0.9}  # sum_w_ensemble >= 0.9
            #{'type': 'ineq', 'fun': lambda x: -sum(x[i] for i in range(1, n_members + 1)) + 3}  # sum_w_ensemble <= 3
        ] + [{'type': 'ineq', 'fun': lambda x: x[i]} for i in range(1, n_members + 1)]
        """
        if params_0 is None:
            initial_params_list = [[0.1] + [1/n_members for _ in range(n_members)] + [0.1, 0.2, 0.1]]
        else:
            initial_params_list = [params_0]

        best_crps = float('inf')
        best_res = None
        print('---------Calculating ensemble csgd------------------') if self.verbose else None
        for params in initial_params_list:
            crps_ensemble_CSGD_optim = lambda A_params: self.crps(ground_truth, wrf_ensemble, A_params)
            res = minimize(fun=crps_ensemble_CSGD_optim, x0=params, method = 'SLSQP', bounds = bounds, jac='3-point', options={'maxiter':2000})
            
            if (res.fun < best_crps) and res.success:
                best_crps = res.fun
                best_res = res

        self.fitted_params = best_res.x if best_res is not None else params_0
        return best_res.fun if best_res is not None else self.crps(ground_truth, wrf_ensemble, params_0)
    
    def predict(self,ground_truth, wrf_ensemble):
        return self.crps(ground_truth, wrf_ensemble, params = self.fitted_params)
    

import numpy as np
import pandas as pd
from properscoring import crps_ensemble,threshold_brier_score

class AnalogEnsemble:
    def __init__(self, weights = None, n_members = None, t_window = 3, X_train = None, y_train = None):
        self.best_weights = weights
        self.best_n_members = n_members
        self.X_train = X_train
        self.y_train = y_train
        assert t_window in [1,3,5]
        middle_t_window = t_window // 2
        self.array_twindow = np.arange(-middle_t_window, middle_t_window + 1)

    def _calc_dist(self, target, analogs, weights):
        """
        Calcula la distancia ponderada entre la predicción target y un conjunto de predicciones históricas (analogs).
        :param target: Array con las predicciones objetivo.
        :param analogs: Matriz con las predicciones históricas.
        :param weights: Array de pesos para cada variable.
        :return: Distancia ponderada entre el target y cada análogo.
        """
        sq_dif = (analogs - target)**2
        temporal_weights = np.array([0.1,0.8,1,0.8,0.1]).reshape(5,1)
        rot_sum_sqdif = np.sqrt(np.sum(temporal_weights*sq_dif,axis = 1))
        dist = np.sum(weights * rot_sum_sqdif, axis=1)

        return dist

    def _obtain_neighbourhood(self,X,i):
        """
        Extrae las predicciones en los tiempos t-1, t, t+1 para el índice i.
        Si está en un borde (inicio o fin), reutiliza los valores cercanos.
        :param X: DataFrame de predicciones.
        :param i: Índice actual.
        :return: Array con las predicciones en los instantes [t-1, t, t+1].
        """
        neigb_indices = self.array_twindow + i 
        neigb_indices[neigb_indices < 0] = 0
        neigb_indices[neigb_indices >= len(X)] = len(X) - 1

        return X.iloc[neigb_indices].values
        """
        if self.t_window == 3:
            if i == 0:  # Si estamos en el primer elemento, no hay t-1, así que duplicamos t
                return np.array([X.iloc[i].values, X.iloc[i].values, X.iloc[i+1].values])
            elif i == len(X) - 1:  # Si estamos en el último elemento, no hay t+1, así que duplicamos t
                return np.array([X.iloc[i-1].values, X.iloc[i].values, X.iloc[i].values])
            else:
                return np.array([X.iloc[i-1].values, X.iloc[i].values, X.iloc[i+1].values])
        elif self.t_window == 5:
        """

    def _grid_search(self, X_train, y_train, X_val, y_val, n_members_options, weights_options):
        """
        Realiza una búsqueda en cuadrícula (grid search) para encontrar los mejores pesos y número de miembros del ensemble.
        :param X_train: Conjunto de predicciones de entrenamiento.
        :param y_train: Conjunto de observaciones de entrenamiento.
        :param n_members_options: Lista de opciones para el número de miembros del ensemble.
        :param weights_options: Lista de opciones de pesos para los miembros del ensemble.
        :return: Mejores pesos y número de miembros del ensemble.
        """
        best_score = float('inf')
        best_weights = None
        best_n_members = None

        hist_predicts = X_train.values
        hist_predicts = np.array([self._obtain_neighbourhood(X_train, j) for j in range(len(X_train))])
        hist_observs = y_train.values

        for n_members in n_members_options:
            for weights in weights_options:
                # Validar el rendimiento con los pesos y miembros actuales
                total_score = 0
                for i in range(len(X_val)):
                    curr_predicts = self._obtain_neighbourhood(X_val,i)
                    # Calcular distancias
                    #print(curr_predicts.shape, hist_predicts.shape)
                    distancias = self._calc_dist(curr_predicts, hist_predicts, weights)
                    indices_analogos = np.argsort(distancias)[:n_members]

                    # Predecir como el promedio de las observaciones análogas
                    #prediccion_anen = np.mean(hist_observ[indices_analogos])

                    # Sumar el error cuadrático medio (MSE)
                    score_i = crps_ensemble(y_val.iloc[i], hist_observs[indices_analogos])
                    total_score += score_i

                avg_score = total_score / len(X_val)

                # Guardar los mejores parámetros
                if avg_score < best_score:
                    best_score = avg_score
                    best_weights = weights
                    best_n_members = n_members

        self.hist_predicts = hist_predicts
        self.hist_observs = hist_observs

        return best_weights, best_n_members

    def fit(self, X_train, y_train, X_val, y_val, n_members_options=[15, 20, 25, 30, 35], weights_options=None):
        """
        Ajusta los pesos y el número de miembros del ensemble utilizando grid search.
        :param X_train: Conjunto de predicciones de entrenamiento.
        :param y_train: Conjunto de observaciones de entrenamiento.
        :param n_members_options: Opciones para el número de miembros del ensemble.
        :param weights_options: Opciones para los pesos de cada miembro del ensemble.
        """
        # Si no se proporcionan pesos, asumimos igualdad entre los miembros del ensemble.
        if weights_options is None:
            mses_per_member = []
            nmembers = X_train.shape[1]
            for i in range(nmembers):
                mse = np.mean((y_train - X_train.iloc[:,i])**2)
                mses_per_member.append(mse)
            w_wrt_mse = 1/np.array(mses_per_member)
            w_wrt_mse *= (nmembers/np.sum(w_wrt_mse))

            weights_options = [w_wrt_mse,np.array([1.02627918, 0.91947047, 1.12229769, 1.00021187, 0.96396852,
                                0.97513522, 0.85310349, 1.11108683, 0.91859867, 0.9378145 ,
                                0.82044969, 1.03512921, 0.89711854, 1.04068204, 0.99662037,
                                1.14107022, 1.11126958, 1.15881912, 0.99888437, 1.05144258,
                                1.14991208, 0.94344106, 1.00676572, 0.9633264 , 0.85710258]),
                                np.ones(nmembers)]

        # Realizar la búsqueda en cuadrícula (grid search)
        self.best_weights, self.best_n_members = self._grid_search(X_train, y_train, X_val, y_val, n_members_options, weights_options)


        print(f"Mejores pesos: {self.best_weights}")
        print(f"Mejor número de miembros: {self.best_n_members}")

    def predict(self, X_test, y_test):
        """
        Realiza la predicción para un nuevo conjunto de datos usando los mejores parámetros ajustados.
        :param X_train: Conjunto de predicciones históricas (entrenamiento).
        :param y_train: Conjunto de observaciones históricas (entrenamiento).
        :param X_test: Conjunto de predicciones objetivo para las cuales se desea hacer la predicción.
        :return: Predicciones del ensemble.
        """
        #ensemble_predictions = []
        metrics = np.array([0,0,0,0], dtype = np.float64)
        n = len(X_test)
        for i in range(n):
            prediccion_actual = self._obtain_neighbourhood(X_test,i)
            observ_actual = y_test.iloc[i]
            
            # Calcular distancias usando los mejores pesos ajustados
            distancias = self._calc_dist(prediccion_actual, self.hist_predicts, self.best_weights)

            # Seleccionar los mejores análogos
            indices_analogos = np.argsort(distancias)[:self.best_n_members]

            # Predecir como el promedio de las observaciones análogas
            prediccion_anen = np.mean(self.hist_observs[indices_analogos])
            
            mse = (prediccion_anen - observ_actual)**2
            rmse = np.sqrt(mse) 
            # Guardar la predicción
            #ensemble_predictions.append(prediccion_anen)
            #crps.append(crps_ensemble(observ_actual,self.hist_observs[indices_analogos]))
            crps = crps_ensemble(observ_actual,self.hist_observs[indices_analogos])
            bs = threshold_brier_score(observ_actual, self.hist_observs[indices_analogos], threshold = 1)
            #print(f'metrics {metrics.shape}')
            #print([crps,rmse,mse,bs])
            #print(f'result_i {np.array([crps, rmse, mse, bs]).shape}')
            metrics += np.array([crps, rmse, mse, bs])

        metrics /= n
        return metrics
        #return np.array(ensemble_predictions), np.array(crps)
