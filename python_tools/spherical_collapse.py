import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from utilities import Cosmology, Utilities
from scipy.integrate import quad, simps, odeint
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline, interp1d, interp2d
from scipy.optimize import fsolve
from scipy.signal import savgol_filter
from scipy.stats import norm
from scipy.special import eval_legendre, legendre

class SingleFit:
    def __init__(self,
                 xi_r_filename,
                 xi_smu_filename,
                 covmat_filename=None,
                 sv_filename=None,
                 vr_filename=None,
                 full_fit=1,
                 smin=0,
                 smax=150,
                 model=1,
                 const_sv=0,
                 model_as_truth=0,
                 om_m=0.285,
                 s8=0.828,
                 eff_z=0.57,
                 vr_coupling='autocorr'):

        self.xi_r_filename = xi_r_filename
        self.sv_filename = sv_filename
        self.vr_filename = vr_filename
        self.xi_smu_filename = xi_smu_filename
        self.covmat_filename = covmat_filename
        self.smin = smin
        self.smax = smax
        self.const_sv = const_sv
        self.model = model
        self.model_as_truth = model_as_truth
        self.vr_coupling = vr_coupling

        # full fit (monopole + quadrupole)
        self.full_fit = bool(full_fit)

        print("Setting up redshift-space distortions model.")

        # cosmology for Minerva
        self.om_m = om_m
        self.s8 = s8
        self.cosmo = Cosmology(om_m=self.om_m)
        self.nmocks = 299 # hardcoded for Minerva

        self.eff_z = eff_z
        self.dA = self.cosmo.get_angular_diameter_distance(self.eff_z)

        self.growth = self.cosmo.get_growth(self.eff_z)
        self.f = self.cosmo.get_f(self.eff_z)
        self.b = 2.01
        self.beta = self.f / self.b
        self.s8norm = self.s8 * self.growth 

        eofz = np.sqrt((self.om_m * (1 + self.eff_z) ** 3 + 1 - self.om_m))
        self.iaH = (1 + self.eff_z) / (100. * eofz)

        if self.vr_coupling == 'true':
            print('Using true radial velocity profile.')
        elif self.vr_coupling == 'autocorr':
            print('Using velocity-to-density coupling for autocorrelation.')
        elif self.vr_coupling == 'crosscorr':
            print('Using velocity-to-density coupling for cross-correlation')
        else:
            sys.exit('Velocity-to-density coupling not recognized.')

        # read real-space galaxy monopole
        data = np.genfromtxt(self.xi_r_filename)
        self.r_for_xi = data[:,0]
        xi_r = data[:,1]
        self.xi_r = InterpolatedUnivariateSpline(self.r_for_xi, xi_r, k=3, ext=0)

        int_xi_r = np.zeros_like(self.r_for_xi)
        dr = np.diff(self.r_for_xi)[0]
        for i in range(len(int_xi_r)):
            int_xi_r[i] = 1./(self.r_for_xi[i]+dr/2)**3 * (np.sum(xi_r[:i+1]*((self.r_for_xi[:i+1]+dr/2)**3
                                                        - (self.r_for_xi[:i+1] - dr/2)**3)))
        self.int_xi_r = InterpolatedUnivariateSpline(self.r_for_xi, int_xi_r, k=3, ext=0)

        int2_xi_r = np.zeros_like(self.r_for_xi)
        dr = np.diff(self.r_for_xi)[0]
        for i in range(len(int2_xi_r)):
            int2_xi_r[i] = 1./(self.r_for_xi[i]+dr/2)**5 * (np.sum(xi_r[:i+1]*((self.r_for_xi[:i+1]+dr/2)**5
                                                        - (self.r_for_xi[:i+1] - dr/2)**5)))
        self.int2_xi_r = InterpolatedUnivariateSpline(self.r_for_xi, int2_xi_r, k=3, ext=0)

        # read los velocity dispersion profile
        self.r_for_v, self.mu_for_v, sv = Utilities.ReadData_TwoDims(self.sv_filename)
        self.sv_converge = sv[-1, -1]
        sv /= self.sv_converge # normalize velocity dispersion
        self.sv = RectBivariateSpline(self.r_for_v, self.mu_for_v, sv)

        if self.vr_coupling == 'true':
            # read radial velocity profile
            data = np.genfromtxt(self.vr_filename)
            self.r_for_v = data[:,0]
            vr = data[:,1]
            self.vr = InterpolatedUnivariateSpline(self.r_for_v, vr, k=3, ext=0)

        # read redshift-space correlation function
        self.s_for_xi, self.mu_for_xi, self.xi_smu = Utilities.ReadData_TwoDims(self.xi_smu_filename)

        if self.model_as_truth:
            print('Using the model prediction as the measurement.')
            if self.model == 1:
                fs8 = self.f * self.s8norm
                sigma_v = self.sv_converge
                alpha = 1.0
                epsilon = 1.0
                alpha_para = alpha * epsilon ** (-2/3)
                alpha_perp = epsilon * alpha_para

                self.xi0_s, self.xi2_s, self.xi4_s = self.model1_theory(fs8,
                                                            sigma_v,
                                                            alpha_perp,
                                                            alpha_para,
                                                            self.s_for_xi,
                                                            self.mu_for_xi)
        else:
            s, self.xi0_s = Utilities.getMultipole(0, self.s_for_xi, self.mu_for_xi, self.xi_smu)
            s, self.xi2_s = Utilities.getMultipole(2, self.s_for_xi, self.mu_for_xi, self.xi_smu)
            s, self.xi4_s = Utilities.getMultipole(4, self.s_for_xi, self.mu_for_xi, self.xi_smu)

        # read covariance matrix
        if os.path.isfile(self.covmat_filename):
            print('Reading covariance matrix: ' + self.covmat_filename)
            self.cov = np.load(self.covmat_filename)
            self.icov = np.linalg.inv(self.cov)
        else:
            sys.exit('Covariance matrix not found.')


        # restrict measured vectors to the desired fitting scales
        if (self.smax < self.s_for_xi.max()) or (self.smin > self.s_for_xi.min()):

            scales = (self.s_for_xi >= self.smin) & (self.s_for_xi <= self.smax)

            # truncate redshift-space data vectors
            self.s_for_xi = self.s_for_xi[scales]
            self.xi0_s = self.xi0_s[scales]
            self.xi2_s = self.xi2_s[scales]
            self.xi4_s = self.xi4_s[scales]

        # build data vector
        if self.full_fit:
            self.datavec = np.concatenate((self.xi0_s, self.xi2_s))
        else:
            self.datavec = self.xi2_s
        
    def model1_theory(self, fs8, sigma_v, alpha_perp, alpha_para, s, mu):
        '''
        Gaussian streaming model (Fisher 1995)
        '''
        beta = fs8 / (self.b * self.s8norm)
        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        hexadecapole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 100)
        r = self.r_for_xi
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y2 = self.int_xi_r(r)
        y3 = self.sv(r, self.mu_for_v)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=0)
        rescaled_int_xi_r = InterpolatedUnivariateSpline(x, y2, k=3, ext=0)
        rescaled_sv = RectBivariateSpline(x, self.mu_for_v, y3)
        sigma_v = alpha_para * sigma_v

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                rpar = true_spar

                # build the integration variable
                sy = 500 * self.iaH
                y = np.linspace(-5 * sy, 5 * sy, 200)

                rpary = rpar - y
                rr = np.sqrt(true_sperp ** 2 + rpary ** 2)
                mur = rpary / rr
                sy = sigma_v * rescaled_sv.ev(rr, mur) * self.iaH

                if self.vr_coupling == 'true':
                    vrmu = self.vr(rr) * mur * self.iaH
                elif self.vr_coupling == 'autocorr':
                    alpha = 0.5
                    int2_xi_r = rescaled_int_xi_r(rr) / (1 + rescaled_xi_r(rr))
                    vrmu = -2/3 * beta * rr * int2_xi_r * (1 + alpha * int2_xi_r) * mur
                else:
                    vrmu = -1/3 * beta * rr * rescaled_int_xi_r(rr) * mur

                los_pdf = norm.pdf(y, loc=vrmu, scale=sy)

                integrand = los_pdf * (1 + rescaled_xi_r(rr))

                xi_model[j] = simps(integrand, y) - 1


            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                  xi_model[np.argsort(true_mu)],
                                                  k=3, ext=0)

            if true_mu.min() < 0:
                mumin = -1
                factor = 2
            else:
                mumin = 0
                factor = 1
        
            # get multipoles
            xaxis = np.linspace(mumin, 1, 1000)

            ell = 0
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            monopole[i] = simps(yaxis, xaxis)

            ell = 2
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            quadrupole[i] = simps(yaxis, xaxis)

            ell = 4
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            hexadecapole[i] = simps(yaxis, xaxis)

        return monopole, quadrupole, hexadecapole


    def model2_theory(self, fs8, alpha_perp, alpha_para, s, mu):
        '''
        Linear model (Eq. 44-46 from Hernandez-Aguayo et at. 2018)
        '''
        beta = fs8 / (self.b * self.s8norm)
        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        hexadecapole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 100)
        r = self.r_for_xi
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y3 = self.int_xi_r(r)
        y4 = self.int2_xi_r(r)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=0)
        rescaled_int_xi_r = InterpolatedUnivariateSpline(x, y3, k=3, ext=0)
        rescaled_int2_xi_r = InterpolatedUnivariateSpline(x, y4, k=3, ext=0)

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                r = true_s

                xi_model[j] = eval_legendre(0, true_mu[j]) * (1 + 2/3*beta + 1/5 * beta**2) * rescaled_xi_r(r) \
                            + eval_legendre(2, true_mu[j]) * (4/3 * beta + 4/7 * beta**2) * (rescaled_xi_r(r) - rescaled_int_xi_r(r)) \
                            + eval_legendre(4, true_mu[j]) * (8/35 * beta**2) * (rescaled_xi_r(r) + 5/2 * rescaled_int_xi_r(r) - 7/2 * rescaled_int2_xi_r(r))

            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                  xi_model[np.argsort(true_mu)],
                                                  k=3)

            if true_mu.min() < 0:
                mumin = -1
                factor = 2
            else:
                mumin = 0
                factor = 1

            # get multipoles
            xaxis = np.linspace(mumin, 1, 1000)

            ell = 0
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            monopole[i] = simps(yaxis, xaxis)

            ell = 2
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            quadrupole[i] = simps(yaxis, xaxis)

            ell = 4
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            hexadecapole[i] = simps(yaxis, xaxis)
            
        return monopole, quadrupole, hexadecapole


    def log_likelihood(self, theta):
        if self.model == 1:
            fs8, sigma_v, epsilon = theta
        else:
            fs8, epsilon = theta

        alpha = 1.0
        alpha_para = alpha * epsilon ** (-2/3)
        alpha_perp = epsilon * alpha_para

        if self.model == 1:
            xi0, xi2, xi4 = self.model1_theory(fs8,
                                          sigma_v,
                                          alpha_perp,
                                          alpha_para,
                                          self.s_for_xi,
                                          self.mu_for_xi)
        else:
            xi0, xi2, xi4 = self.model2_theory(fs8,
                                          alpha_perp,
                                          alpha_para,
                                          self.s_for_xi,
                                          self.mu_for_xi)

        if self.full_fit:
            modelvec = np.concatenate((xi0, xi2))
        else:
            modelvec = xi2

        chi2 = np.dot(np.dot((modelvec - self.datavec), self.icov), modelvec - self.datavec)
        loglike = -self.nmocks/2 * np.log(1 + chi2/(self.nmocks-1))
        return loglike

    def log_prior(self, theta):
        if self.model == 1:
            fs8, sigma_v, epsilon = theta
            if 0.1 < fs8 < 2.0 and 10 < sigma_v < 700 and 0.8 < epsilon < 1.2:
                return 0.0
        else:
            fs8, epsilon = theta
            if 0.1 < fs8 < 2.0 and 0.8 < epsilon < 1.2:
                return 0.0
        
        return -np.inf

class JointFit:
    def __init__(self,
                 xi_r_filename,
                 xi_smu_filename,
                 covmat_filename,
                 smin,
                 smax,
                 vr_filename=None,
                 sv_filename=None,
                 full_fit=1,
                 model=1,
                 model_as_truth=0,
                 const_sv=0,
                 om_m=0.285,
                 s8=0.828,
                 eff_z=0.57,
                 vr_coupling='autocorr'):

        self.vr_coupling = vr_coupling
        xi_r_filenames = xi_r_filename.split(',')
        sv_filenames = sv_filename.split(',')
        xi_smu_filenames = xi_smu_filename.split(',')
        smins = [int(i) for i in smin.split(',')]
        smaxs = [int(i) for i in smax.split(',')]
        if self.vr_coupling == 'true':
            vr_filenames = vr_filename.split(',')
            vr_filename = {}

        self.ndenbins = len(xi_r_filenames)
        xi_r_filename = {}
        xi_smu_filename = {}
        sv_filename = {}
        smin = {}
        smax = {}

        for j in range(self.ndenbins):
            xi_r_filename['den{}'.format(j)] = xi_r_filenames[j]
            sv_filename['den{}'.format(j)] = sv_filenames[j]
            xi_smu_filename['den{}'.format(j)] = xi_smu_filenames[j]
            smin['den{}'.format(j)] = smins[j]
            smax['den{}'.format(j)] = smaxs[j]
            if self.vr_coupling == 'true':
                vr_filename['den{}'.format(j)] = vr_filenames[j]
            

        # full fit (monopole + quadrupole)
        self.full_fit = full_fit
        self.model = model
        self.model_as_truth = model_as_truth
        self.const_sv = const_sv

        print("Setting up redshift-space distortions model.")

        # cosmology for Minerva
        self.om_m = om_m
        self.s8 = s8
        self.cosmo = Cosmology(om_m=self.om_m)
        self.nmocks = 299 # hardcoded for Minerva
        self.eff_z = eff_z

        self.growth = self.cosmo.get_growth(self.eff_z)
        self.f = self.cosmo.get_f(self.eff_z)
        self.b = 2.01
        self.s8norm = self.s8 * self.growth 

        eofz = np.sqrt((self.om_m * (1 + self.eff_z) ** 3 + 1 - self.om_m))
        self.iaH = (1 + self.eff_z) / (100. * eofz) 

        # read covariance matrix
        if os.path.isfile(covmat_filename):
            print('Reading covariance matrix: ' + covmat_filename)
            self.cov = np.load(covmat_filename)
            self.icov = np.linalg.inv(self.cov)
        else:
            sys.exit('Covariance matrix not found.')

        if self.vr_coupling == 'true':
            print('Using true radial velocity profile.')
        elif self.vr_coupling == 'autocorr':
            print('Using velocity-to-density coupling for autocorrelation.')
        elif self.vr_coupling == 'crosscorr':
            print('Using velocity-to-density coupling for cross-correlation')
        else:
            sys.exit('Velocity-to-density coupling not recognized.')

        self.r_for_xi = {}
        self.s_for_xi = {}
        self.mu_for_xi = {}
        self.xi_r = {}
        self.int_xi_r = {}
        self.int2_xi_r = {}
        self.xi0_s = {}
        self.xi2_s = {}
        self.xi4_s = {}

        if self.model == 1:
            self.r_for_v = {}
            self.mu_for_v = {}
            self.sv = {}
            self.sv_converge = {}

        if self.vr_coupling == 'true':
            self.vr = {}

        self.datavec = np.array([])

        for j in range(self.ndenbins):
            denbin = 'den{}'.format(j)
            # read real-space monopole
            data = np.genfromtxt(xi_r_filename[denbin])
            self.r_for_xi[denbin] = data[:,0]
            xi_r = data[:,-2]
            self.xi_r[denbin] = InterpolatedUnivariateSpline(self.r_for_xi[denbin], xi_r, k=3, ext=0)

            r = self.r_for_xi[denbin]
            int_xi_r = np.zeros_like(r)
            dr = np.diff(r)[0]
            for i in range(len(int_xi_r)):
                int_xi_r[i] = 1./(r[i]+dr/2)**3 * (np.sum(xi_r[:i+1]*((r[:i+1]+dr/2)**3
                                                            - (r[:i+1] - dr/2)**3)))
            self.int_xi_r[denbin] = InterpolatedUnivariateSpline(r, int_xi_r, k=3, ext=0)

            int2_xi_r = np.zeros_like(r)
            dr = np.diff(r)[0]
            for i in range(len(int2_xi_r)):
                int2_xi_r[i] = 1./(r[i]+dr/2)**5 * (np.sum(xi_r[:i+1]*((r[:i+1]+dr/2)**5
                                                            - (r[:i+1] - dr/2)**5)))
            self.int2_xi_r[denbin] = InterpolatedUnivariateSpline(r, int2_xi_r, k=3, ext=0)

            if self.vr_coupling == 'true':
                # read radial velocity
                data = np.genfromtxt(vr_filename[denbin])
                self.r_for_v[denbin] = data[:,0]
                vr = data[:,-2]
                self.vr[denbin] = InterpolatedUnivariateSpline(self.r_for_v[denbin], vr, k=3, ext=0)


            if self.model == 1:
                self.r_for_v[denbin], self.mu_for_v[denbin], sv = Utilities.ReadData_TwoDims(sv_filename[denbin])
                self.sv_converge[denbin] = sv[-1, -1]

                if self.const_sv:
                    sv = np.ones(len(self.r_for_v[denbin]))
                else:
                    sv /= self.sv_converge[denbin]

                self.sv[denbin] = RectBivariateSpline(self.r_for_v[denbin],
                                                      self.mu_for_v[denbin],
                                                      sv)

            # read redshift-space correlation function
            self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu_obs = Utilities.ReadData_TwoDims(xi_smu_filename[denbin])

            if self.model_as_truth:
                print('Using the model prediction as the measurement.')
                if self.model == 1:
                    fs8 = self.f * self.s8norm
                    sigma_v = self.sv_converge[denbin]
                    alpha = 1.0
                    epsilon = 1.0
                    alpha_para = alpha * epsilon ** (-2/3)
                    alpha_perp = epsilon * alpha_para

                    self.xi0_s[denbin], self.xi2_s[denbin], self.xi4_s[denbin] = self.model1_theory(fs8,
                                                                                sigma_v,
                                                                                alpha_perp,
                                                                                alpha_para,
                                                                                self.s_for_xi[denbin],
                                                                                self.mu_for_xi[denbin],
                                                                                denbin)

            else:
                s, self.xi0_s[denbin] = Utilities.getMultipole(0, self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu_obs)
                s, self.xi2_s[denbin] = Utilities.getMultipole(2, self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu_obs)
                s, self.xi4_s[denbin] = Utilities.getMultipole(4, self.s_for_xi[denbin], self.mu_for_xi[denbin], xi_smu_obs)


            # restrict measured vectors to the desired fitting scales
            scales = (self.s_for_xi[denbin] >= smin[denbin]) & (self.s_for_xi[denbin] <= smax[denbin])

            self.s_for_xi[denbin] = self.s_for_xi[denbin][scales]
            self.xi0_s[denbin] = self.xi0_s[denbin][scales]
            self.xi2_s[denbin] = self.xi2_s[denbin][scales]
            self.xi4_s[denbin] = self.xi4_s[denbin][scales]

            if self.full_fit:
                self.datavec = np.concatenate((self.datavec, self.xi0_s[denbin], self.xi2_s[denbin]))
            else:
                self.datavec = np.concatenate((self.datavec, self.xi2_s[denbin]))

        

    def log_likelihood(self, theta):
        if self.model == 1:
            if self.ndenbins == 2:
                fs8, sigma_v1, sigma_v2, epsilon = theta
                sigmalist = [sigma_v1, sigma_v2]

            if self.ndenbins == 3:
                fs8, sigma_v1, sigma_v2, sigma_v3, epsilon = theta
                sigmalist = [sigma_v1, sigma_v2, sigma_v3]
                
            if self.ndenbins == 4:
                fs8, sigma_v1, sigma_v2, sigma_v3, sigma_v4, epsilon = theta
                sigmalist = [sigma_v1, sigma_v2, sigma_v3, sigma_v4]

            if self.ndenbins == 5:
                fs8, sigma_v1, sigma_v2, sigma_v3, sigma_v4, sigma_v5, epsilon = theta
                sigmalist = [sigma_v1, sigma_v2, sigma_v3, sigma_v4, sigma_v5]

        alpha = 1.0
        alpha_para = alpha * epsilon ** (-2/3)
        alpha_perp = epsilon * alpha_para

        sigma_v = {}
        modelvec = np.array([])

        for j in range(self.ndenbins):
            denbin = 'den{}'.format(j)
            sigma_v[denbin] = sigmalist[j]

            if self.model == 1:
                xi0, xi2, xi4 = self.model1_theory(fs8,
                                            sigma_v[denbin],
                                            alpha_perp,
                                            alpha_para,
                                            self.s_for_xi[denbin],
                                            self.mu_for_xi[denbin],
                                            denbin)

            if self.full_fit:
                modelvec = np.concatenate((modelvec, xi0, xi2))
            else:
                modelvec = np.concatenate((modelvec, xi2))

        chi2 = np.dot(np.dot((modelvec - self.datavec), self.icov), modelvec - self.datavec)
        loglike = -self.nmocks/2 * np.log(1 + chi2/(self.nmocks-1))
        return loglike

    def log_prior(self, theta):
        if self.model == 1:
            if self.ndenbins == 2:
                fs8, sigma_v1, sigma_v2, epsilon = theta

                if 0.1 < fs8 < 2.0 \
                and 10 < sigma_v1 < 700 \
                and 10 < sigma_v2 < 700 \
                and 0.8 < epsilon < 1.2:
                    return 0.0

            if self.ndenbins == 3:
                fs8, sigma_v1, sigma_v2, sigma_v3, epsilon = theta

                if 0.1 < fs8 < 2.0 \
                and 10 < sigma_v1 < 700 \
                and 10 < sigma_v2 < 700 \
                and 10 < sigma_v3 < 700 \
                and 0.8 < epsilon < 1.2:
                    return 0.0

            if self.ndenbins == 4:
                fs8, sigma_v1, sigma_v2, sigma_v3, sigma_v4, epsilon = theta

                if 0.1 < fs8 < 2.0 \
                and 10 < sigma_v1 < 700 \
                and 10 < sigma_v2 < 700 \
                and 10 < sigma_v3 < 700 \
                and 10 < sigma_v4 < 700 \
                and 0.8 < epsilon < 1.2:
                    return 0.0

            if self.ndenbins == 5:
                fs8, sigma_v1, sigma_v2, sigma_v3, sigma_v4, sigma_v5, epsilon = theta

                if 0.1 < fs8 < 2.0 \
                and 10 < sigma_v1 < 700 \
                and 10 < sigma_v2 < 700 \
                and 10 < sigma_v3 < 700 \
                and 10 < sigma_v4 < 700 \
                and 10 < sigma_v5 < 700 \
                and 0.8 < epsilon < 1.2:
                    return 0.0

        return -np.inf


    def model1_theory(self, fs8, sigma_v, alpha_perp, alpha_para, s, mu, denbin):
        '''
        Gaussian streaming model (Fisher 1995)
        '''
        beta = fs8 / (self.b * self.s8norm)
        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        hexadecapole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 100)
        r = self.r_for_xi[denbin]
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r[denbin](r)
        y2 = self.int_xi_r[denbin](r)
        y3 = self.sv[denbin](r, self.mu_for_v[denbin])

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=0)
        rescaled_int_xi_r = InterpolatedUnivariateSpline(x, y2, k=3, ext=0)
        rescaled_sv = RectBivariateSpline(x, self.mu_for_v[denbin], y3)
        sigma_v = alpha_para * sigma_v

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                rpar = true_spar

                # build the integration variable
                sy = 500 * self.iaH
                y = np.linspace(-5 * sy, 5 * sy, 200)

                rpary = rpar - y
                rr = np.sqrt(true_sperp ** 2 + rpary ** 2)
                mur = rpary / rr
                sy = sigma_v * rescaled_sv.ev(rr, mur) * self.iaH

                if self.vr_coupling == 'true':
                    vrmu = self.vr(rr) * mur * self.iaH
                elif self.vr_coupling == 'autocorr':
                    alpha = 0.5
                    int2_xi_r = rescaled_int_xi_r(rr) / (1 + rescaled_xi_r(rr))
                    vrmu = -2/3 * beta * rr * int2_xi_r * (1 + alpha * int2_xi_r) * mur
                else:
                    vrmu = -1/3 * beta * rr * rescaled_int_xi_r(rr) * mur

                los_pdf = norm.pdf(y, loc=vrmu, scale=sy)

                integrand = los_pdf * (1 + rescaled_xi_r(rr))

                xi_model[j] = simps(integrand, y) - 1


            # build interpolating function for xi_smu at true_mu
            mufunc = InterpolatedUnivariateSpline(true_mu[np.argsort(true_mu)],
                                                  xi_model[np.argsort(true_mu)],
                                                  k=3, ext=0)

            if true_mu.min() < 0:
                mumin = -1
                factor = 2
            else:
                mumin = 0
                factor = 1
        
            # get multipoles
            xaxis = np.linspace(mumin, 1, 1000)

            ell = 0
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            monopole[i] = simps(yaxis, xaxis)

            ell = 2
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            quadrupole[i] = simps(yaxis, xaxis)

            ell = 4
            lmu = eval_legendre(ell, xaxis)
            yaxis = mufunc(xaxis) * (2 * ell + 1) / factor * lmu
            hexadecapole[i] = simps(yaxis, xaxis)

        return monopole, quadrupole, hexadecapole


