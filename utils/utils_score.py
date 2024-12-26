
import numpy as np
from scipy.stats import rankdata
import pandas as pd

#Create skill scores of any metric
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