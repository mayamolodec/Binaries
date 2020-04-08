#!/usr/bin/python3
# coding: utf-8

import numpy as np
from astropy.io import fits
from scipy import interpolate
from scipy.interpolate import griddata
import argparse
import os
import matplotlib.pyplot as plt


def sort(J, H, K, KS, B, V, Teff):

    mask = Teff.argsort()
    J1 = J[mask]
    H1 = H[mask]
    K1 = K[mask]
    KS1 = KS[mask]
    B1 = B[mask]
    V1 = V[mask]
    Teff1 = Teff[mask]

    return J1, H1, K1, KS1, B1, V1, Teff1


def interp_1D(new_grid, old_grid, old_func):
    res = np.interp(new_grid, old_grid, old_func)
    # mas = np.argmin(np.abs(new_grid - val))

    # return np.round(res[mas], 3)
    return res


path = os.path.dirname(os.path.abspath(__file__))
a = fits.open(os.path.join(path, 'ABS_VAL.fits'))

Teff = np.array(a[1].data.field(0))
log_g = np.array(a[1].data.field(1))
log_z = np.array(a[1].data.field(2))
J = np.array(a[1].data.field(3))
H = np.array(a[1].data.field(4))
K = np.array(a[1].data.field(5))
KS = np.array(a[1].data.field(6))
B = np.array(a[1].data.field(7))
V = np.array(a[1].data.field(8))


Teff = Teff[(log_g == 5) * (log_z == 0)]
J = J[(log_g == 5) * (log_z == 0)]
H = H[(log_g == 5) * (log_z == 0)]
K = K[(log_g == 5) * (log_z == 0)]
KS = KS[(log_g == 5) * (log_z == 0)]
B = B[(log_g == 5) * (log_z == 0)]
V = V[(log_g == 5) * (log_z == 0)]

J, H, K, KS, B, V, Teff = sort(J, H, K, KS, B, V, Teff)
teff_new_grid = np.arange(2300, 12001, 1)


Jnew = interp_1D(teff_new_grid, Teff, J)
Hnew = interp_1D(teff_new_grid, Teff, H)
Knew = interp_1D(teff_new_grid, Teff, K)
KSnew = interp_1D(teff_new_grid, Teff, KS)
Bnew = interp_1D(teff_new_grid, Teff, B)
Vnew = interp_1D(teff_new_grid, Teff, V)

I_j = pow(10, (-0.4 * (Jnew)))
I_h = pow(10, (-0.4 * (Hnew)))
I_k = pow(10, (-0.4 * (Knew)))
I_ks = pow(10, (-0.4 * (KSnew)))
I_b = pow(10, (-0.4 * (Bnew)))
I_v = pow(10, (-0.4 * (Vnew)))

np.savetxt('fluxes_vega_normed.txt', (I_j, I_h, I_k, I_ks, I_b, I_v, teff_new_grid), delimiter=',')

# plt.plot(teff_new_grid, Jnew, label='J')
# plt.plot(teff_new_grid, Hnew, label='H')
# plt.plot(teff_new_grid, Knew, label='K')
# plt.plot(teff_new_grid, KSnew, label='KS')
# plt.plot(teff_new_grid, Bnew, label='B')
# plt.plot(teff_new_grid, Vnew, label='V')

plt.plot(teff_new_grid, I_j, label='J')
plt.plot(teff_new_grid, I_h, label='H')
plt.plot(teff_new_grid, I_k, label='K')
plt.plot(teff_new_grid, I_ks, label='KS')
plt.plot(teff_new_grid, I_b, label='B')
plt.plot(teff_new_grid, I_v, label='V')
plt.legend()
plt.show()
