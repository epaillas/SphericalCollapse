import numpy as np
import sys
import os
from utilities import Cosmology, Utilities
from scipy.integrate import quad, simps, odeint
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline, interp1d, interp2d
from scipy.optimize import fsolve, odeint
from scipy.signal import savgol_filter
from scipy.stats import norm
from scipy.special import eval_legendre, legendre

class SingleFit:
    def __init__(self,
                 xi_r_filename,
                 delta_r_filename,
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
                 Omega_m0=0.285,
                 s8=0.828,
                 eff_z=0.57,
                 vr_coupling='spherical'):

        self.xi_r_filename = xi_r_filename
        self.delta_r_filename = delta_r_filename
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
        self.Omega_m0 = Omega_m0
        self.s8 = s8
        self.cosmo = Cosmology(om_m=self.Omega_m0)
        self.nmocks = 299 # hardcoded for Minerva

        self.eff_z = eff_z
        self.dA = self.cosmo.get_angular_diameter_distance(self.eff_z)

        self.growth = self.cosmo.get_growth(self.eff_z)
        self.f = self.cosmo.get_f(self.eff_z)
        self.b = 2.01
        self.beta = self.f / self.b
        self.s8norm = self.s8 * self.growth 

        eofz = np.sqrt((self.Omega_m0 * (1 + self.eff_z) ** 3 + 1 - self.Omega_m0))
        self.iaH = (1 + self.eff_z) / (100. * eofz)

        # set this to true if you want to test the 
        # performance of the model using the 
        # measured radial velocities
        if self.vr_coupling == 'true':
            print('Using true radial velocity profile.')
        elif self.vr_coupling == 'spherical':
            print('Calculating peculiar velocities from spherical collapse.')
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

        # read void-matter correlation function
        data = np.genfromtxt(self.delta_r_filename)
        self.r_for_delta = data[:,0]
        delta_r = data[:,-2]

        Delta_r = np.zeros_like(self.r_for_delta)
        dr = np.diff(self.r_for_delta)[0]
        for i in range(len(Delta_r)):
            Delta_r[i] = 1./(self.r_for_delta[i]+dr/2)**3 * (np.sum(delta_r[:i+1]*((self.r_for_delta[:i+1]+dr/2)**3
                                                        - (self.r_for_delta[:i+1] - dr/2)**3)))
        self.Delta_r = InterpolatedUnivariateSpline(self.r_for_delta, Delta_r, k=3, ext=3)
        self.delta_r = InterpolatedUnivariateSpline(self.r_for_delta, delta_r, k=3, ext=3)

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
            sigma_v = self.sv_converge
            alpha = 1.0
            epsilon = 1.0
            alpha_para = alpha * epsilon ** (-2/3)
            alpha_perp = epsilon * alpha_para

            self.xi0_s, self.xi2_s, self.xi4_s = self.multipoles_theory(
                                                        self.Omega_m0,
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
        

    def multipoles_theory(self, Omega_m0, sigma_v, alpha_perp, alpha_para, s, mu):
        '''
        RSD model that relies on spherical collapse model 
        to map from densities to velocities.
        '''

        monopole = np.zeros(len(s))
        quadrupole = np.zeros(len(s))
        hexadecapole = np.zeros(len(s))
        true_mu = np.zeros(len(mu))
        xi_model = np.zeros(len(mu))

        if self.vr_coupling == 'spherical':
            # set up parameters for spherical collapse
            Omega_L0 = 1 - Omega_m0
            zi = 999
            zf = self.eff_z
            z = np.linspace(zi, zf, 100)
            a = 1/(1 + z)
            t = CosmologicalTime(zi, zf)

            # array of initial linear densities
            Delta_i = np.linspace(-0.01, 0.0025, 1000)

            # find solutions to spherical collapse ODE
            sol1 = []
            sol2 = []
            for dl in Delta_i:
                g0 = [1 - dl/3, -dl/3]
                sol = odeint(SphericalCollapse, g0, t, args=(Omega_m0, Omega_L0))
                y = sol[:,0]
                yprime = sol[:,1]
                sol1.append(y[-1]**-3 - 1)
                sol2.append(yprime[-1])

            # find the initial Deltas that match the late ones
            interp_Delta = InterpolatedUnivariateSpline(sol1, Delta_i, k=3, ext=0)
            matched_Delta_i = interp_Delta(self.Delta_r(self.r_for_delta))

            # find the peculiar velocities associated to late Deltas
            interp_vpec = InterpolatedUnivariateSpline(Delta_i, sol2, k=3, ext=0)
            matched_vpec = interp_vpec(matched_Delta_i)

            # transform peculiar velocities to desired units
            H = Hubble(a=a[-1], Omega_m0=Omega_m0, Omega_L0=Omega_L0)
            q = self.r_for_delta * a[-1] * (1 + self.Delta_r(self.r_for_delta))**(1/3) 
            vpec = matched_vpec * H * q
            dvpec = np.gradient(vpec, self.r_for_delta)
            self.vr = InterpolatedUnivariateSpline(self.r_for_delta, vpec, k=3, ext=0)
            self.dvr = InterpolatedUnivariateSpline(self.r_for_delta, dvpec, k=3, ext=0)

        # rescale input monopole functions to account for alpha values
        mus = np.linspace(0, 1., 80)
        r = self.r_for_delta
        rescaled_r = np.zeros_like(r)
        for i in range(len(r)):
            rescaled_r[i] = np.trapz((r[i] * alpha_para) * np.sqrt(1. + (1. - mus ** 2) *
                            (alpha_perp ** 2 / alpha_para ** 2 - 1)), mus)

        x = rescaled_r
        y1 = self.xi_r(r)
        y2 = self.vr(r)
        y3 = self.dvr(r)
        y4 = self.sv(r)

        # build rescaled interpolating functions using the relabelled separation vectors
        rescaled_xi_r = InterpolatedUnivariateSpline(x, y1, k=3, ext=0)
        rescaled_vr = InterpolatedUnivariateSpline(x, y2, k=3, ext=0)
        rescaled_dvr = InterpolatedUnivariateSpline(x, y3, k=3, ext=0)
        rescaled_sv = InterpolatedUnivariateSpline(x, y4, k=3, ext=0)
        sigma_v = alpha_para * sigma_v

        for i in range(len(s)):
            for j in range(len(mu)):
                true_sperp = s[i] * np.sqrt(1 - mu[j] ** 2) * alpha_perp
                true_spar = s[i] * mu[j] * alpha_para
                true_s = np.sqrt(true_spar ** 2. + true_sperp ** 2.)
                true_mu[j] = true_spar / true_s

                # solve Eq. 7 from arXiv 1712.07575
                def residual(rpar):
                    rperp = true_sperp
                    r = np.sqrt(rpar**2 + rperp**2)
                    mu = rpar / r
                    res = rpar - true_spar + rescaled_vr(r)*mu * self.iaH
                    return res

                rpar = fsolve(func=residual, x0=true_spar)[0]

                sy_central = sigma_v * rescaled_sv(np.sqrt(true_sperp**2 + rpar**2)) * self.iaH
                y = np.linspace(-5 * sy_central, 5 * sy_central, 200)

                rpary = rpar - y
                rr = np.sqrt(true_sperp ** 2 + rpary ** 2)
                sy = sigma_v * rescaled_sv(rr) * self.iaH

                integrand = (1 + rescaled_xi_r(rr)) * (1 + rescaled_vr(rr)/(rr/self.iaH) +\
                                                (rescaled_dvr(rr) - rescaled_vr(rr)/rr)*self.iaH * true_mu[j]**2)**(-1)

                integrand = integrand * np.exp(-(y**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)

                xi_model[j] = np.trapz(integrand, y) - 1


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


    def log_likelihood(self, theta):
        if self.model == 1:
            Omega_m0, sigma_v, epsilon = theta
        else:
            Omega_m0, epsilon = theta

        alpha = 1.0
        alpha_para = alpha * epsilon ** (-2/3)
        alpha_perp = epsilon * alpha_para

        if self.model == 1:
            xi0, xi2, xi4 = self.multipoles_theory(Omega_m0,
                                          sigma_v,
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
            Omega_m0, sigma_v, epsilon = theta
            if 0.1 < Omega_m0 < 2.0 and 10 < sigma_v < 700 and 0.8 < epsilon < 1.2:
                return 0.0
        else:
            Omega_m0, epsilon = theta
            if 0.1 < Omega_m0 < 2.0 and 0.8 < epsilon < 1.2:
                return 0.0
        
        return -np.inf


def SphericalCollapse(g, lna, Omega_m0, Omega_L0):
    '''
    Collapse of a spherical shell. Solution to the ODE

    y'' + (1/2 - 3/2 w om_l) y' + om_m/2 (y^{-3} - 1) y = 0

    Let h = y'
    h' + (1/2 - 3/2 w om_l) h + om_m/2 (y^{-3} - 1) y = 0
    '''
    om_m = Omega_m(lna, Omega_m0=Omega_m0, Omega_L0=Omega_L0)
    om_l = Omega_L(lna, Omega_m0=Omega_m0, Omega_L0=Omega_L0)
    y, h = g
    dgda = [h, -(1/2 + 3/2*om_l)*h - om_m/2*(y**(-3) - 1)*y]
    return dgda

def Omega_m(lna, Omega_m0, Omega_L0):
    a = np.exp(lna)
    om_m = Omega_m0 / (Omega_m0 + Omega_L0 * a**3)
    return om_m

def Omega_L(lna, Omega_m0, Omega_L0):
    a = np.exp(lna)
    om_l = Omega_L0 / (Omega_m0 * a**-3 + Omega_L0)
    return om_l

def CosmologicalTime(zi, zf):
    ai = np.log(1/(1 + zi))
    af = np.log(1/(1 + zf))
    t = np.linspace(ai, af, 10000)
    return t

def Hubble(a, Omega_m0, Omega_L0):
    return 100 * np.sqrt(Omega_m0 * a ** -3 + Omega_L0)


