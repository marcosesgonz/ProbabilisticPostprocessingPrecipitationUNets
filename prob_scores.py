
import numpy as np
from scipy.stats import rankdata
import pandas as pd

class SkillScorer:
    def __init__(self, climat_metric, metric_name = 'metric') -> None:
        self.climat_metric = climat_metric
        self.metric_name = metric_name
        self.values = []

    def calc(self, metric_val):
        metricss = 1 - metric_val/self.climat_metric
        return metricss

    def register(self,name,crps):
        metricss = self.calc(crps)
        self.values.append({'name':name, self.metric_name : metricss})
    
    def show(self):
        print(pd.DataFrame(self.values).sort_values(self.metric_name,ascending=False))


#Adaptado de: https://github.com/oliverangelil/rankhistogram/blob/master/ranky.py
class rankz():
    def __init__(self,ties_method = 'randomize'):
        self.ties_method = ties_method
        self.number_ties = 0 #Nºcasos en los que hay un empate con al menos 1 miembro del ensemble

    def _apply_mask(self,obs,ensemble,mask):
        mask = np.bool_(mask)
        #Al coger indices con los booleanos se pierden las dimensiones. Se quedarán arrays de la forma (N,).
        obs = obs[mask]
        ensemble = ensemble[:,mask] #La máscara se aplica en todos los miembros del ensemble
        return obs, ensemble

    def __call__(self,obs,ensemble,mask=None):
        ''' Parameters
        ----------
        obs : numpy array of observations 
        ensemble : numpy array of ensemble, with the first dimension being the 
            ensemble member and the remaining dimensions being identical to obs
        mask : boolean mask of shape of obs, with zero/false being where grid cells are masked.  

        Returns
        -------
        histogram data for ensemble.shape[0] + 1 bins. 
        The first dimension of this array is the height of 
        each histogram bar, the second dimension is the histogram bins. 
            '''
        
        if mask is None:
            obs = obs.flatten()
            ensemble = ensemble.reshape(ensemble.shape[0],-1)
        else:
            obs,ensemble = self._apply_mask(obs,ensemble,mask)

        #Concatenas en la primera dimensión los ensembles con las observaciones. Con obs[np.newaxis]: (N,) -> (1,N)
        combined = np.vstack((obs[np.newaxis],ensemble))  #En la primera dimensión tendrás el número de miembros del ensemble + 1.

        # print('computing ranks')
        if self.ties_method == 'randomize':
            #Hago el ranking para cada instancia, e.d, A LO LARGO del eje 0.
            ranks=np.apply_along_axis(lambda x: rankdata(x,method='min'),0,combined) #En caso de empates se pone el valor mínimo del ranking que les tocaría a todos los valores empatados 
            # print('computing ties')
            #ranks[0] corresponde a los puestos de ranking correspondientes a las observaciones, que son las que interesan
            ties = np.sum(ranks[0]==ranks[1:], axis=0) #Calcula el nºempates de el ranking de la observación con el ranking asignado a los miembros del ensemble
            self.number_ties = np.sum(ties>0)                #Número de casos en los que hay empate de al menos con 1 miembro del ensemble
            ranks = ranks[0]                           #Nos quedamos solo con el ranking de las observac
            tie = np.unique(ties)                      #Los valores de nºempates que existen en los datos
            
            for i in range(1,len(tie)): #Se empieza en 1 ya que el valor 0 del índice corresponde a no empate.
                #Array de valores de ranking de las observaciones que cumplen haber empatado un número de tie[i] veces 
                index=ranks[ties==tie[i]]
                # print('randomizing tied ranks for ' + str(len(index)) + ' instances where there is ' + str(tie[i]) + ' tie/s. ' + str(len(tie)-i-1) + ' more to go')
                #Sustituyo el valor del ranking dado al empatar por method='min' por uno aleatorio comprendido entre los posibles valores que ocupan los empatados.
                ranks[ties==tie[i]]=[np.random.randint(index[j],index[j]+tie[i]+1,tie[i])[0] for j in range(len(index))]

        else:
            #Hago el ranking para cada instancia, e.d, A LO LARGO del eje 0.
            ranks=np.apply_along_axis(lambda x: rankdata(x,method=self.ties_method),0,combined) #En caso de empates se pone el valor mínimo del ranking que les tocaría a todos los valores empatados
            self.number_ties = np.sum(np.any(ranks[0] == ranks[1:], axis=0))
            ranks = ranks[0]

        return np.histogram(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5, combined.shape[0]+1))
    


class CRPS_mine():
    def __init__(self):
        self.crps =   {"CRPS_sum": 0.0, "n": 0.0} #Initialize a CRPS object.

    def _reset(self):
        self.__init__()

    def CRPS_accum(self, X_f, X_o):
        """
        Compute the average continuous ranked probability score (CRPS) for a set
        of forecast ensembles and the corresponding observations and accumulate the
        result to the given CRPS object.

        Parameters
        ----------
        CRPS: dict
        The CRPS object.
        X_f: array_like
        Array of shape (k,m,n,...) containing the values from an ensemble
        forecast of k members with shape (m,n,...).
        X_o: array_like
        Array of shape (m,n,...) containing the observed values corresponding
        to the forecast.

        References
        ----------
        :cite:`Her2000`
        """
        X_f = np.vstack([X_f[i, :].flatten() for i in range(X_f.shape[0])]).T
        X_o = X_o.flatten()

        mask = np.logical_and(np.all(np.isfinite(X_f), axis=1), np.isfinite(X_o))

        X_f = X_f[mask, :].copy()
        X_f.sort(axis=1)
        X_o = X_o[mask]

        n = X_f.shape[0]
        m = X_f.shape[1]

        alpha = np.zeros((n, m + 1))
        beta = np.zeros((n, m + 1))

        for i in range(1, m):
            mask = X_o > X_f[:, i]
            alpha[mask, i] = X_f[mask, i] - X_f[mask, i - 1]
            beta[mask, i] = 0.0

            mask = np.logical_and(X_f[:, i] > X_o, X_o > X_f[:, i - 1])
            alpha[mask, i] = X_o[mask] - X_f[mask, i - 1]
            beta[mask, i] = X_f[mask, i] - X_o[mask]

            mask = X_o < X_f[:, i - 1]
            alpha[mask, i] = 0.0
            beta[mask, i] = X_f[mask, i] - X_f[mask, i - 1]

        mask = X_o < X_f[:, 0]
        alpha[mask, 0] = 0.0
        beta[mask, 0] = X_f[mask, 0] - X_o[mask]

        mask = X_f[:, -1] < X_o
        alpha[mask, -1] = X_o[mask] - X_f[mask, -1]
        beta[mask, -1] = 0.0

        p = 1.0 * np.arange(m + 1) / m
        res = np.sum(alpha * p**2.0 + beta * (1.0 - p) ** 2.0, axis=1)
        #print(f'Sum(res) object: {np.sum(res)}')
        self.crps["CRPS_sum"] += np.sum(res)
        self.crps["n"] += len(res)

    def _compute(self):
        """
        Compute the averaged values from the given CRPS object.

        Parameters
        ----------
        CRPS: dict
        A CRPS object created with CRPS_init.

        Returns
        -------
        out: float
        The computed CRPS.
        """
        return 1.0 * self.crps["CRPS_sum"] / self.crps["n"]

    def compute(self,X_f, X_o):
        """
        Compute the continuous ranked probability score (CRPS).
        
        Parameters
        ----------
        X_f: array_like
        Array of shape (k,m,n,...) containing the values from an ensemble
        forecast of k members with shape (m,n,...).
        X_o: array_like
        Array of shape (m,n,...) containing the observed values corresponding
        to the forecast.

        Returns
        -------
        out: float
        The computed CRPS.

        References
        ----------
        :cite:`Her2000`
        """
        X_f = X_f.copy()
        X_o = X_o.copy()
        self.CRPS_accum(X_f, X_o)
        crps_score = self._compute()
        self._reset()
        return crps_score

        

def CRPS(X_f, X_o):
    """
    Compute the continuous ranked probability score (CRPS).

    Parameters
    ----------
    X_f: array_like
      Array of shape (k,m,n,...) containing the values from an ensemble
      forecast of k members with shape (m,n,...).
    X_o: array_like
      Array of shape (m,n,...) containing the observed values corresponding
      to the forecast.

    Returns
    -------
    out: float
      The computed CRPS.

    References
    ----------
    :cite:`Her2000`
    """

    X_f = X_f.copy()
    X_o = X_o.copy()
    crps = CRPS_init()
    CRPS_accum(crps, X_f, X_o)
    return CRPS_compute(crps)


def CRPS_init():
    """
    Initialize a CRPS object.

    Returns
    -------
    out: dict
      The CRPS object.
    """
    return {"CRPS_sum": 0.0, "n": 0.0}


def CRPS_accum(CRPS, X_f, X_o):
    """
    Compute the average continuous ranked probability score (CRPS) for a set
    of forecast ensembles and the corresponding observations and accumulate the
    result to the given CRPS object.

    Parameters
    ----------
    CRPS: dict
      The CRPS object.
    X_f: array_like
      Array of shape (k,m,n,...) containing the values from an ensemble
      forecast of k members with shape (m,n,...).
    X_o: array_like
      Array of shape (m,n,...) containing the observed values corresponding
      to the forecast.

    References
    ----------
    :cite:`Her2000`
    """
    X_f = np.vstack([X_f[i, :].flatten() for i in range(X_f.shape[0])]).T
    X_o = X_o.flatten()

    mask = np.logical_and(np.all(np.isfinite(X_f), axis=1), np.isfinite(X_o))

    X_f = X_f[mask, :].copy()
    X_f.sort(axis=1)
    X_o = X_o[mask]

    n = X_f.shape[0]
    m = X_f.shape[1]

    alpha = np.zeros((n, m + 1))
    beta = np.zeros((n, m + 1))

    for i in range(1, m):
        mask = X_o > X_f[:, i]
        alpha[mask, i] = X_f[mask, i] - X_f[mask, i - 1]
        beta[mask, i] = 0.0

        mask = np.logical_and(X_f[:, i] > X_o, X_o > X_f[:, i - 1])
        alpha[mask, i] = X_o[mask] - X_f[mask, i - 1]
        beta[mask, i] = X_f[mask, i] - X_o[mask]

        mask = X_o < X_f[:, i - 1]
        alpha[mask, i] = 0.0
        beta[mask, i] = X_f[mask, i] - X_f[mask, i - 1]

    mask = X_o < X_f[:, 0]
    alpha[mask, 0] = 0.0
    beta[mask, 0] = X_f[mask, 0] - X_o[mask]

    mask = X_f[:, -1] < X_o
    alpha[mask, -1] = X_o[mask] - X_f[mask, -1]
    beta[mask, -1] = 0.0

    p = 1.0 * np.arange(m + 1) / m
    res = np.sum(alpha * p**2.0 + beta * (1.0 - p) ** 2.0, axis=1)

    CRPS["CRPS_sum"] += np.sum(res)
    CRPS["n"] += len(res)


def CRPS_compute(CRPS):
    """
    Compute the averaged values from the given CRPS object.

    Parameters
    ----------
    CRPS: dict
      A CRPS object created with CRPS_init.

    Returns
    -------
    out: float
      The computed CRPS.
    """
    return 1.0 * CRPS["CRPS_sum"] / CRPS["n"]


import numpy as np

class CRPS_R_Equivalent:
    def __init__(self):
        self.crps_ens = None
    
    def crps_func(self, ensemble_data, obs):
        """
        Calculate the CRPS for ensemble forecasts and observations.

        Parameters
        ----------
        ensemble_data: array_like
            Array of shape (m, n) containing the values from an ensemble forecast of m members with n samples.
        obs: array_like
            Array of shape (n,) containing the observed values corresponding to the forecast.

        Returns
        -------
        crps_ens: array
            The CRPS for the ensemble forecast.
        """
        n_members, n_samples = ensemble_data.shape

        # Calculate crpsEns1: Mean absolute error between ensemble forecasts and observations
        print(np.abs(ensemble_data - obs))
        crps_ens1 = np.mean(np.abs(ensemble_data - obs), axis=0)
        print(crps_ens1)

        # Calculate crpsEns2: Sum of pairwise absolute differences between ensemble members
        crps_ens2 = np.zeros(n_samples)
        for i in range(n_members):
            for j in range(n_members):
                crps_ens2 += np.abs(ensemble_data[i, :] - ensemble_data[j, :])


        # Normalize crpsEns2
        crps_ens2 /= (2 * (n_members ** 2))

        # Calculate crpsEns
        crps_ens = crps_ens1 - crps_ens2

        return crps_ens

    def compute(self, ensemble_data, obs):
        """
        Compute the continuous ranked probability score (CRPS) for the ensemble forecast.

        Parameters
        ----------
        ensemble_data: array_like
            Array of shape (n, m) containing the values from an ensemble forecast of n members with m samples.
        obs: array_like
            Array of shape (m,) containing the observed values corresponding to the forecast.

        Returns
        -------
        crps: float
            The computed CRPS for the ensemble forecast.
        """
        crps_ens = self.crps_func(ensemble_data, obs)
        return np.mean(crps_ens)

