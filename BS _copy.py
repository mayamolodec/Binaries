import numpy as np
from numpy import sin, cos, pi, sqrt  # makes the code more readable
# import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
# from mayavi import mlab  # or from enthought.mayavi import mlab
from scipy.optimize import newton
# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot


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
    return T


def normal_vector(grad_x, grad_y, grad_z):
    g_xyz = sqrt(grad_x ** 2 + grad_y**2 + grad_z**2)
    n1 = np.array([-grad_x / g_xyz, -grad_y / g_xyz, -grad_z / g_xyz])
    return n1


def s_elementary(r1, eta, n, n1, lam, mu, nu):
    ds = ((r1**2 * sin(eta) * 2 * np.pi * np.pi / (n * n)) / (lam * n1[0] + mu * n1[1] + nu * n1[2]))
    return ds


def spectra(lamda, T):
    h = 1
    c = 1
    k = 1
    spectrum = (2 * h * c**2 / (lamda ^ 5) * 1 / (np.exp(h * c / lamda * k * T) - 1))
    return spectrum


def flux(m, n, n1, ds, spectrum):
    rot_angle = np.linspace(0, 2 * np.pi, m)
    flux = np.zeros((50))

    for d in range(m):
        a0 = np.array([-sin(i) * cos(rot_angle[d]), -sin(i) * sin(rot_angle[d]), cos(i)])
        cos_gamma = np.zeros((n**2))

        for n in range(n**2):
            cos_gamma[n] = a0[0] * n1[0, n] + a0[1] * n1[0, n] + a0[2] * n1[0, n]

        spectrum_d = spectrum[(cos_gamma > 0)]
        ds_d = ds[(cos_gamma > 0)]
        cos_gamma = cos_gamma[(cos_gamma > 0)]

        dI = (spectrum_d * (1 - x * (1 - cos_gamma)) * cos_gamma * ds_d)
        d_flux = np.sum(dI)

        flux[d] = d_flux
    return(flux, rot_angle * 180 / np.pi)


eta, phi = np.mgrid[0.000001:np.pi:50j, -0.5 * np.pi: 1.5 * np.pi:50j]

m = 50
n = 50
G = 6.67408 * pow(10, -11)
M = 2 * pow(10, 30)
pot1, pot2 = 2.88, 10.
q = 0.5
T_0 = 5800
r_init = 1e-5
i = np.pi / 3

x = 0.5
lamda = 1000

eta = np.reshape(eta, n**2)
phi = np.reshape(phi, n**2)

r1 = r_get(roche, r_init, eta, phi)
# r2 = [newton(roche, r_init, args=(th, ph, pot2, 1. / q)) for th, ph in zip(eta.ravel(), phi.ravel())]


lam, mu, nu = dir_cos(eta, phi)
x1, y1, z1 = decart_get(r1, lam, mu, nu)

# r2 = np.array(r2).reshape(eta.shape)


# gradX = q * (-1. - (x1 - 1.) / (x1**2 + 2 * x1 + y1**2 + z1**2 + 1)**1.5) + (q + 1.) * x1 * (1. - (z1**2) * (1. - x1**2 / (r1**2)) / (y1**2 * (z1**2 / y1**2 + 1.))) - (q + 1.) * z1**2 * r1**2 * (2 * x1**3 / (r1**4) - 2 * x1 / (r1)) / (2 * y1**2 * (z1**2 / y1**2) + 1.) - x1 / (r1**3)
# gradY = (q + 1.) * y1 * (1. - (z1**2 * (1. - x1**2 / (r1**2))) / (y1**2 * (z1**2 / y1**2 + 1.))) - (q * y1) / ((r1**2 - 2 * x1 + 1.)**1.5) + 0.5 * (q + 1.) * (r1**2) * (-(2 * x1**2 * z1**2) / (y1 * (z1**2 / y1**2 + 1.) * (x1**2 + y1**2 + z1**2)**2) - (2 * z1**4 * (1. - x1**2 / (x1**2 + y1**2 + z1**2))) / (y1**5 * (z1**2 / y1**2 + 1.)**2) + (2 * z1**2 * (1. - x1**2 / (x1**2 + y1**2 + z1**2))) / (y1**3 * (z1**2 / y1**2 + 1))) - y1 / (r1**2)**1.5
# gradZ = (q + 1.) * z1 * (1. - (z1**2 * (1. - x1**2 / (r1**2))) / (y1**2 * (z1**2 / y1**2 + 1.))) - (q * z1) / ((r1**2 - 2 * x1 + 1.)**1.5) + 0.5 * (q + 1.) * (r1**2) * (-(2 * z1 * (1. - x1**2 / (r1**2))) / (y1**2 * (z1**2 / y1**2 + 1.)) - (2 * x1**2 * z1**3) / (y1**2 * (z1**2 / y1**2 + 1.) * (x1**2 + y1**2 + z1**2)**2) + (2 * z1**3 * (1. - x1**2 / (x1**2 + y1**2 + z1**2))) / (y1**4 * (z1**2 / y1**2 + 1)**2)) - z1 / (r1**2)**1.5
grad_x, grad_y, grad_z = grad_xyz(x1, y1, z1, q)

# gradR = q * (-lam - (r1 - lam) / (r1**2 - 2 * r1 * lam + 1)**1.5) + (q + 1) * r1 * (1 - nu**2) - 1 / (r1**2)
# gradETA = q * (r1 * sin(eta) - (r1 * sin(eta)) / (r1**2 - 2 * r1 * lam + 1)**1.5) - (q + 1) * (r1**2) * nu * lam * sin(phi)
# gradFI = -(q + 1) * (r1**2) * nu * mu
#g = np.sqrt(gradR**2)
# g = np.sqrt(gradR**2 + (gradETA / r1)**2 + (gradFI / (r1 * sin(eta)))**2)

T = T_get(grad_x, grad_y, grad_z)


n1 = normal_vector(grad_x, grad_y, grad_z)
ds = s_elementary(r1, eta, n, n1, lam, mu, nu)
B = spectra(lamda, T)

flux, phase = flux(m, n, n1, ds, B)


# x2 = r2*np.sin(theta)*np.cos(phi)
# y2 = r2*np.sin(theta)*np.sin(phi)
# z2 = r2*np.cos(theta)

# x2_ = -x2
# x2_+= 1

# T = T_0*pow((g/g_pol)*1/4)

# rot_angle = pi
# Rz = np.array([[cos(rot_angle),-sin(rot_angle),0],
#                [sin(rot_angle), cos(rot_angle),0],
#                [0,             0,              1]])
# B = np.dot(Rz,np.array([x2,y2,z2]).reshape((3,-1))) # we need to have a 3x3 times 3xN array
# x2,y2,z2 = B.reshape((3,x2.shape[0],x2.shape[1])) # but we want our original shape back
# x2 += 1 # simple translation

# mlab.figure()
# mlab.mesh(x1,y1,z1,scalars=r1)
# mlab.mesh(x2,y2,z2,scalars=r2)

print('Flux', flux, 'phase', phase)
fig = plt.figure()
#ax = fig.gca()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')
# ax.scatter(gradETA, gra)
# ax.scatter(nx, ny, nz, color='pink')
# ax.scatter(cos(n1[1, :]), cos(n1[2, :]) * sin(n1[1, :]), sin(n1[2, :]) * sin(n1[1, :]), color='gold')
ax.scatter(x1, y1, z1, c=g_xyz, cmap="inferno", alpha=0.5)
# ax.scatter(n1[0, :], n1[1, :], n1[2, :])
# plt.scatter(rot_angle * 180 / np.pi, dots)
# plt.ylabel('Flux')
# plt.xlabel('Phase')

plt.show()
