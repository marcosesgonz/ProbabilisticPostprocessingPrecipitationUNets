import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

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


class ReliabilityDiagram:
    def __init__(self, threshold=1.0, n_bins=10, hist_max = None):
        """
        :param threshold: exceedance threshold
        :param n_bins: number of probability bins
        :param hist_max: manual maximum for histogram y-axis (log-scale). If None, auto-scale.
        """
        assert threshold > 0
        self.threshold = threshold
        self.n_bins = n_bins
        self.hist_max = hist_max
        self.prob_bins = np.linspace(0, 1, n_bins + 1)
        self.bin_centers = (self.prob_bins[:-1] + self.prob_bins[1:]) / 2

    def compute_probabilities_csgd(self, mean, std, shift):
        shape = mean**2 / std**2
        scale = std**2 / mean
        return 1 - gamma.cdf(self.threshold + shift, a=shape, scale=scale)

    def compute_probabilities_ensemb(self, ensemble_values):
        try:
            ensemble_values = np.array(ensemble_values)
            predicted_prob = (ensemble_values > self.threshold).mean(axis= 1)
        except:
            if isinstance(ensemble_values,list): #The number of ensemble members could change in each batch
                predicted_prob = []
                for ensemble_val in ensemble_values: 
                    ensemble_val = np.array(ensemble_val)
                    #print(f'Batch ensemble shape: {ensemble_val.shape}')
                    predicted_prob.append((ensemble_val > self.threshold).mean())
                predicted_prob = np.array(predicted_prob)
            else:
                raise NotImplementedError

        return predicted_prob

    def calculate_observed_frequency(self, predicted_prob, observed):
        # Bin indices
        binned = np.digitize(predicted_prob, self.prob_bins) - 1
        obs_freq = []
        counts = []
        for i in range(self.n_bins):
            mask = (binned == i)
            count = mask.sum()
            counts.append(count)
            freq = (observed[mask] > self.threshold).mean() if count > 0 else np.nan
            obs_freq.append(freq)
        return np.array(obs_freq), np.array(counts)
    
    def compute_calibration_errors(self, obs_freq, counts):
        # Bin centers as forecast probabilities
        p_i = self.bin_centers
        N = counts.sum()
        # Avoid division by zero
        weights = counts / N if N > 0 else np.zeros_like(counts)
        # Compute absolute errors only for bins with counts>0
        abs_err = np.abs(obs_freq - p_i)
        ece = np.nansum(weights * abs_err)
        mce = np.nanmax(abs_err)
        return ece, mce

    def plot(self, predicted_prob, observed, ax=None, path=None):
        # Compute observed frequency and histogram counts
        obs_freq, counts = self.calculate_observed_frequency(predicted_prob, observed)
        # Compute ECE and MCE
        ece, mce = self.compute_calibration_errors(obs_freq, counts)
        
        # Main axes
        standalone = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            standalone = True
        else:
            fig = ax.figure
        ax.plot(self.bin_centers, obs_freq, '--', label="Observed Frequency")
        ax.scatter(self.bin_centers, obs_freq, s=40, alpha=0.7, edgecolor='k')
        ax.plot([0, 1], [0, 1], 'k--', label="Perfect Reliability")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.legend(loc='upper left')
        ax.grid(True)

        # Add ECE & MCE text above inset
        ax.text(
            x=0.02, y=0.85,
            s=f"ECE={ece:.4f}\nMCE={mce:.4f}",
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )

        # inset histogram
        ax_hist = inset_axes(ax, width="30%", height="30%", loc='lower right', borderpad=1)
        width = self.prob_bins[1] - self.prob_bins[0]
        # bar plot with log scale y-axis
        ax_hist.bar(self.bin_centers, counts, width=width * 0.9, align='center')
        ax_hist.set_xbound(0, 1)
        ax_hist.set_yscale('log')
        ax_hist.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        ax_hist.yaxis.set_major_formatter(LogFormatterMathtext())
        # apply manual y-limit if provided
        if self.hist_max is not None:
            ax_hist.set_ylim(bottom=1, top=self.hist_max)
        ax_hist.set_xticks([])
        # leave y-ticks for powers of 10
        ax_hist.set_title("Counts", fontsize=8)
        # Save if standalone
        if standalone and path:
            fig.savefig(path)
            plt.close(fig)

    def evaluate_csgd(self, mean, std, shift, observed, path):
        predicted_prob = self.compute_probabilities_csgd(mean, std, shift)
        self.plot(predicted_prob, observed, path)

    def evaluate_ensemble(self, ensemble_values, observed, path):
        try:
            ensemble_arr = np.array(ensemble_values)
            predicted_prob = self.compute_probabilities_ensemb(ensemble_arr)
        except:
            predicted = []
            for batch in ensemble_values:
                predicted.extend(self.compute_probabilities_ensemb(batch))
            predicted_prob = np.array(predicted)
        self.plot(predicted_prob, observed, path)