import numpy as np
from numpy import sin, cos, pi, sqrt  # makes the code more readable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, Column
from scipy.optimize import newton


def roche(r, eta, phi, pot, q):
    lam, nu = cos(eta), sin(eta) * sin(phi)
    return (pot - (1. / r + q * (1. / sqrt(1. - 2 * lam * r + r**2) - lam * r) + 0.5 * (q + 1) * r**2 * (1 - nu**2)))


def r_get(roche, r_init, eta, phi):
    r1 = [newton(roche, r_init, args=(th, ph, pot1, q)) for th, ph in zip(eta.ravel(), phi.ravel())]
    r1 = np.array(r1)
    return r1


def decart_get(r1, lam, mu, nu):
    x1 = r1 * lam
    y1 = r1 * mu
    z1 = r1 * nu
    return (x1, y1, z1)


def dir_cos(eta, phi):
    lam = cos(eta)
    mu = cos(phi) * sin(eta)
    nu = sin(phi) * sin(eta)
    return(lam, mu, nu)


def grad_xyz(x1, y1, z1, q):
    grad_x = -x1 / ((x1**2 + y1**2 + z1**2)**1.5) + q * (1 - x1) / (((1 - x1)**2 + y1**2 + z1**2)**1.5) + (1 + q) * x1 - q
    grad_y = -y1 * (1 / ((x1**2 + y1**2 + z1**2)**1.5) + q / (((1 - x1)**2 + y1**2 + z1**2)**1.5) - (1 + q))
    grad_z = -z1 * (1 / ((x1**2 + y1**2 + z1**2)**1.5) + q / (((1 - x1)**2 + y1**2 + z1**2)**1.5))
    return grad_x, grad_y, grad_z


def T_get(grad_x, grad_y, grad_z):
    g_xyz = sqrt(grad_x ** 2 + grad_y**2 + grad_z**2)
    g_0 = np.mean(g_xyz)
    T = T_0 * pow((g_xyz / g_0), 0.25)
    T = np.round(T, 0)
    return T


def normal_vector(grad_x, grad_y, grad_z):
    g_xyz = sqrt(grad_x ** 2 + grad_y**2 + grad_z**2)
    n1 = np.array([-grad_x / g_xyz, -grad_y / g_xyz, -grad_z / g_xyz])
    return n1


def s_elementary(r1, eta, n, n1, lam, mu, nu):
    ds = ((r1**2 * sin(eta) * 2 * np.pi * np.pi / (n * n)) / (lam * n1[0] + mu * n1[1] + nu * n1[2]))
    return ds


def spectra(lamda, d_lamda, T, n):
    h = 1
    c = 1
    k = 1
    spectrum = np.zeros((n**2))
    for i in range(n**2):
        spectrum[i] = np.sum((2 * h * c**2 / (lamda ^ 5) * 1 / (np.exp(h * c / lamda * k * T[i]) - 1)) * d_lamda)
    return spectrum


def model_spectra(I_j, I_h, I_k, I_ks, I_b, I_v, teff, T, n):
    j = np.zeros((n**2))
    h = np.zeros((n**2))
    k = np.zeros((n**2))
    k = np.zeros((n**2))
    ks = np.zeros((n**2))
    b = np.zeros((n**2))
    v = np.zeros((n**2))

    for i in range(n**2):
        # mas = np.argmin(np.abs(teff - T[i]))
        mas = np.argwhere(teff == T[i])
        j[i] = I_j[mas]
        h[i] = I_h[mas]
        k[i] = I_k[mas]
        ks[i] = I_ks[mas]
        b[i] = I_b[mas]
        v[i] = I_v[mas]
    return (j, h, k, ks, b, v)


def flux(m, n, n1, i, ds, spectrum):
    rot_angle = np.linspace(0, 4 * np.pi, m)
    flux = np.zeros((m))
    a = np.zeros((3, m))
    for d in range(m):
        a0 = np.array([-sin(i) * cos(rot_angle[d]), -sin(i) * sin(rot_angle[d]), cos(i)])
        a[:, d] = a0
        cos_gamma = np.zeros((n**2))

        for j in range(n**2):
            cos_gamma[j] = a0[0] * n1[0, j] + a0[1] * n1[1, j] + a0[2] * n1[2, j]

        spectrum_d = spectrum[(cos_gamma > 0)]
        ds_d = ds[(cos_gamma > 0)]
        cos_gamma = cos_gamma[(cos_gamma > 0)]

        dI = (spectrum_d * (1 - x * (1 - cos_gamma)) * cos_gamma * ds_d)
        # d_flux = -2.5 * np.log10(np.sum(dI))
        d_flux = np.sum(dI)
        flux[d] = d_flux
    return(flux, rot_angle * 180 / np.pi)


I_j, I_h, I_k, I_ks, I_b, I_v, teff = np.loadtxt('fluxes_vega_normed.txt', delimiter=',')

# STEPS:
dots_lc = 50
dots_andle_step = 50
# rot_angle = np.linspace(0, 2 * np.pi, dots_lc)
# rot_angle = np.round(rot_angle * 180 / np.pi, 0)

# PARAMETRS:
G = 6.67408 * pow(10, -11)
M = 2 * pow(10, 30)
pot1, pot2 = 2.88, 10.
q = 0.5
T_0 = 5800
r_init = 1e-5
i = np.pi / 2
x = 0.5
lamda = np.arange(500, 1000, 1)
d_lamda = 1

eta, phi = np.mgrid[0.000001:np.pi:50j, -0.5 * np.pi: 1.5 * np.pi:50j]
eta = np.reshape(eta, dots_andle_step**2)
phi = np.reshape(phi, dots_andle_step**2)

r1 = r_get(roche, r_init, eta, phi)

lam, mu, nu = dir_cos(eta, phi)
x1, y1, z1 = decart_get(r1, lam, mu, nu)

grad_x, grad_y, grad_z = grad_xyz(x1, y1, z1, q)
T = T_get(grad_x, grad_y, grad_z)

n1 = normal_vector(grad_x, grad_y, grad_z)
ds = s_elementary(r1, eta, dots_andle_step, n1, lam, mu, nu)
B = spectra(lamda, d_lamda, T, dots_andle_step)
j, h, k, ks, b, v = model_spectra(I_j, I_h, I_k, I_ks, I_b, I_v, teff, T, dots_andle_step)
flux1, phase1 = flux(dots_lc, dots_andle_step, n1, i, ds, B)

flux2, phase2 = flux(dots_lc, dots_andle_step, n1, i, ds, j)
flux3, phase3 = flux(dots_lc, dots_andle_step, n1, i, ds, h)
flux4, phase4 = flux(dots_lc, dots_andle_step, n1, i, ds, k)
flux5, phase5 = flux(dots_lc, dots_andle_step, n1, i, ds, ks)
flux6, phase6 = flux(dots_lc, dots_andle_step, n1, i, ds, b)
flux7, phase7 = flux(dots_lc, dots_andle_step, n1, i, ds, v)


# fig = plt.figure()
# ax = fig.gca(projection='3d')

# for rot, x, y, z in zip(rot_angle, a[0, :], a[1, :], a[2, :]):
#     label = '%.f' % rot
#     ax.text(x, y, z, label)

# ax.scatter(x1, y1, z1, c=T, cmap="inferno", alpha=0.5)
# ax.scatter(a[0, :], a[1, :], a[2, :])
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# plt.show()


plt.scatter(phase1, flux1, label='PLANK')
plt.scatter(phase2, flux2, label='J')
plt.scatter(phase3, flux3, label='H')
plt.scatter(phase4, flux4, label='K')
plt.scatter(phase5, flux5, label='KS')
plt.scatter(phase6, flux6, label='B')
plt.scatter(phase7, flux7, label='V')

plt.ylabel('Flux')
plt.xlabel('Phase')
plt.legend()
plt.show()

# plt.plot(teff, I_j, label='J')
# plt.plot(teff, I_h, label='H')
# plt.plot(teff, I_k, label='K')
# plt.plot(teff, I_ks, label='KS')
# plt.plot(teff, I_b, label='B')
# plt.plot(teff, I_v, label='V')
# plt.plot(teff,)
# plt.legend()
# plt.show()
