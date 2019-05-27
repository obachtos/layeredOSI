import numpy as np

def rate_means(rates):

    prn = np.array(map(np.mean, rates))
    
    print prn
    return prn


def populations_means(vec, n_neur):

    splt_pts = np.cumsum(n_neur)[:-1]
    prn = np.array(map(np.mean, np.split(vec, splt_pts)))

    print prn
    return prn

def populations_fun(fun, vec, n_neur):

    splt_pts = np.cumsum(n_neur)[:-1]
    prn = np.array(map(fun, np.split(vec, splt_pts)))

    print prn
    return prn



def is_outlier(points, thresh=3.5):
    """ median-absolute-deviation (MAD) 
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def Q_mul(q, x, n_neurons):

    if q.shape != (8,8):
        raise UserWarning('shape of q should be (8,8)')

    splt_pts = np.cumsum(n_neurons)[:-1]
    x_splt = np.split(x, splt_pts)
    x_sums = np.array(map(np.sum, x_splt))

    y = np.dot(q, x_sums)

    return np.repeat(y, n_neurons)

def S_mul(W, q, x, n_neurons):
    return W.dot(x) - Q_mul(q, x, n_neurons)
