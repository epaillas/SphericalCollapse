import numpy as np
from scipy.integrate import quad, simps
from scipy.special import hyp2f1, legendre, eval_legendre
from scipy.interpolate import InterpolatedUnivariateSpline

class Utilities:
    def __init__(self):
        return

    @staticmethod
    def next_pow_two(n):
        '''
        Returns the largest power of two
        smaller than a given positive integer.
        '''
        i = 1
        while i < n:
            i = i << 1
        i = i >> 1
        return i

    @staticmethod
    def truncate_covmat(input_cm, k):
        """
        Method to truncate composite covariance matrix for multipole data vector (monopole and quadrupole), to keep
        only the first k indices in each
        :param input_cm: numpy array, input covariance matrix to resize
        :param k: integer, largest index to keep
        :return: the truncated covariance matrix
        """

        bin_range = input_cm.shape[0] / 2
        output_cm = np.zeros((2 * k, 2 * k))
        for i in range(input_cm.shape[0]):
            for j in range(input_cm.shape[0]):
                if i < k and j < k:
                    output_cm[i, j] = input_cm[i, j]
                elif bin_range <= i < bin_range + k and j < k:
                    output_cm[i + k - bin_range, j] = input_cm[i, j]
                elif i < k and bin_range <= j < bin_range + k:
                    output_cm[i, j + k - bin_range] = input_cm[i, j]
                elif bin_range <= i < bin_range + k and bin_range <= j < bin_range + k:
                    output_cm[i + k - bin_range, j + k - bin_range] = input_cm[i, j]

        return output_cm

    @staticmethod
    def ReadData_TwoDims(fname):
        data = np.genfromtxt(fname)
        dim1 = np.unique(data[:,0])
        dim2 = np.unique(data[:,1])

        vary_dim2 = False
        if data[0,0] == data[1,0]:
            vary_dim2 = True

        result = np.zeros([len(dim1), len(dim2)])
        counter = 0
        if vary_dim2:
            for i in range(len(dim1)):
                for j in range(len(dim2)):
                    result[i, j] = data[counter, 2]
                    counter += 1
        else:
            for i in range(len(dim2)):
                for j in range(len(dim1)):
                    result[j, i] = data[counter, 2]
                    counter += 1
        return dim1, dim2, result

    @staticmethod
    def getMultipole(ell, s, mu, xi_smu):
        multipole = np.zeros(xi_smu.shape[0])
        if mu.min() < 0:
            factor = 2
            mumin = -1
        else:
            factor = 1
            mumin=0
        for i in range(xi_smu.shape[0]):
            mufunc = InterpolatedUnivariateSpline(mu, xi_smu[i, :], k=3, ext=3)
            xaxis = np.linspace(mumin, 1, 1000)
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            multipole[i] = simps(yaxis, xaxis)
        return s, multipole

    @staticmethod
    def getMonopole(s, mu, xi_smu):
        monopole = np.zeros(xi_smu.shape[0])
        if mu.min() < 0:
            factor = 2
            mumin = -1
        else:
            factor = 1
            mumin=0
        for i in range(xi_smu.shape[0]):
            mufunc = InterpolatedUnivariateSpline(mu, xi_smu[i, :], k=3, ext=3)
            xaxis = np.linspace(mumin, 1, 1000)
            yaxis = mufunc(xaxis) / factor
            monopole[i] = simps(yaxis, xaxis)
        return s, monopole

    @staticmethod
    def getQuadrupole(s, mu, xi_smu):
        quadrupole = np.zeros(xi_smu.shape[0])
        if mu.min() < 0:
            factor = 2
            mumin = -1
        else:
            factor = 1
            mumin = 0
        for i in range(xi_smu.shape[0]):
            mufunc = InterpolatedUnivariateSpline(mu, xi_smu[i, :], k=3, ext=3)
            xaxis = np.linspace(mumin, 1, 1000)
            yaxis = mufunc(xaxis) * 5 / 2 * (3 * xaxis**2 - 1) / factor
            quadrupole[i] = simps(yaxis, xaxis)

        return s, quadrupole

    @staticmethod
    def correlation_multipoles(xirmu, r, ell=0):
        """
        Calculate the Legendre multipoles of a correlation function xi(r, mu)
        :param xirmu: an instance of scipy.interpolate.interp2d providing an interpolation for xi(r, mu)
        :param r: array of r values at which to calculate the multipoles
        :param ell: integer, multipole order
        Note that xi(r, mu) is assumed an even function of mu, so integral is evaluated over 0<mu<1 and then doubled
        Returns:
            array of values at the input r
        """

        output = np.zeros(len(r))
        lmu = legendre(ell)
        for i in range(len(r)):
            output[i] = quad(lambda x: xirmu(r[i], x) * lmu(x) * (2 * ell + 1), 0, 1, full_output=1)[0]

        return output

class Cosmology:
    def __init__(self, om_m=0.308, h=0.676):
        c = 299792.458
        om_l = 1.0 - om_m
        ztab = np.linspace(0, 4, 1000)
        rtab = np.zeros_like(ztab)
        for i in range(len(ztab)):
            rtab[i] = quad(lambda x: 0.01 * c / np.sqrt(om_m * (1 + x) ** 3 + om_l), 0, ztab[i])[0]

        self.h = h
        self.c = c
        self.om_m = om_m
        self.om_l = om_l
        self.ztab = ztab
        self.rtab = rtab

    def get_H(self, z):
        return 100 * np.sqrt(self.om_m * (1 + z) ** 3 + self.om_l)

    # comoving distance in Mpc/h
    def get_comoving_distance(self, z):
        return np.interp(z, self.ztab, self.rtab)

    # angular diameter distance in Mpc/h
    def get_angular_diameter_distance(self, z):
        return np.interp(z, self.ztab, self.rtab) / (1 + z)

    # redshift at a given comoving distance
    def get_redshift(self, r):
        return np.interp(r, self.rtab, self.ztab)

    # growth factor at a given redshift
    def get_growth(self, eff_z):
        az = 1. / (1 + eff_z)
        growth = az ** 2.5 * np.sqrt(self.om_l + self.om_m * az ** (-3.)) * \
                hyp2f1(5. / 6, 3. / 2, 11. / 6, -(self.om_l * az ** 3.) / self.om_m) / \
                hyp2f1(5. / 6, 3. / 2, 11. / 6, -self.om_l / self.om_m)
        return growth

    # linear growth rate at a given redshift
    def get_f(self, eff_z):
        f = ((self.om_m * (1 + eff_z)**3.) / (self.om_m * (1 + eff_z)**3 + self.om_l))**0.55
        return f