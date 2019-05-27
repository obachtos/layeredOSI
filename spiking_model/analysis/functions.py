import sys
import numpy as np

def clearPath():

	i = 0
	while i < len(sys.path):

		if sys.path[i].find('/layered_simulations/data/') != -1:
			rm = sys.path.pop(i)

			print 'removed path entry: \n' + rm
		else:
			i += 1


def rate2nspike(rates, t_measure):

    n_spikes = []
    n_spikes_total = []
    for p in range(len(rates)):
        n_spikes.append(rates[p] * t_measure*1e-3)
        n_spikes_total.append(np.sum(n_spikes[p], axis=1))

    return n_spikes, n_spikes_total



def calcOS(rates, angles, rect=True):

    mean_OSIs = np.empty(8)
    POs, OSIs = [], []
    OSVs = []
    for p in range(len(rates)):

        rec_rates = rates[p].copy()
        if rect:
            rec_rates[rec_rates<0.] = 0.

        OSV = np.sum(rec_rates * np.exp(2.*np.pi*1j*angles/180.), axis=1)
        nOSV = np.sum(rec_rates, axis=1)

        OSV[nOSV==0] = 0.
        nOSV[nOSV==0] = 1.

        OSVs_pop = OSV/nOSV

        PO = np.angle(OSVs_pop)        
        PO[PO<0] = PO[PO<0] + 2.*np.pi
        PO = PO/(2.*np.pi)*180.

        OSI = np.abs(OSVs_pop)

        mean_OSIs[p] = np.mean(OSI)
        POs.append(PO)
        OSIs.append(OSI)
        OSVs.append(OSVs_pop)

    return mean_OSIs, POs, OSIs, OSVs