import numpy as np
from streaming import SingleFit, JointFit
import os
import sys
import argparse
import emcee
from multiprocessing import Pool

def log_probability(theta):
        lp = model.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + model.log_likelihood(theta)

parser = argparse.ArgumentParser(description='')

parser.add_argument('--ncores', type=int)
parser.add_argument('--xi_smu', type=str)
parser.add_argument('--xi_r', type=str)
parser.add_argument('--sv_r', type=str)
parser.add_argument('--covmat', type=str)
parser.add_argument('--full_fit', type=int, default=1)
parser.add_argument('--smin', type=str)
parser.add_argument('--smax', type=str)
parser.add_argument('--model', type=int, default=1)
parser.add_argument('--const_sv', type=int, default=0)
parser.add_argument('--model_as_truth', type=int, default=0)
parser.add_argument('--backend_name', type=str)
parser.add_argument('--vr_coupling', type=str)

args = parser.parse_args()  

os.environ["OMP_NUM_THREADS"] = "1"

# figure out if single or joint fit
ndenbins = len(args.xi_r.split(','))

if ndenbins > 1:
    print('Joint fit with {} bins.'.format(ndenbins))

    if args.model == 1:
        model = JointFit(
                    xi_r_filename=args.xi_r,
                    sv_filename=args.sv_r,
                    xi_smu_filename=args.xi_smu,
                    covmat_filename=args.covmat,
                    full_fit=args.full_fit,
                    smin=args.smin,
                    smax=args.smax,
                    model=args.model,
                    const_sv=args.const_sv,
                    model_as_truth=args.model_as_truth,
                    vr_coupling=args.vr_coupling)

        nwalkers = args.ncores
        niter = 10000

        fs8 = 0.472
        epsilon = 1.0

        if ndenbins == 2:
            ndim = 4
            sigma_v1 = 360
            sigma_v2 = 360
            start_params = np.array([fs8, sigma_v1, sigma_v2, epsilon])
            scales = [1, 100, 100, 1]

        if ndenbins == 3:
            ndim = 5
            sigma_v1 = 360
            sigma_v2 = 360
            sigma_v3 = 360
            start_params = np.array([fs8, sigma_v1, sigma_v2, sigma_v3, epsilon])
            scales = [1, 100, 100, 100, 1]

        if ndenbins == 4:
            ndim = 6
            sigma_v1 = 360
            sigma_v2 = 360
            sigma_v3 = 360
            sigma_v4 = 360
            start_params = np.array([fs8, sigma_v1, sigma_v2, sigma_v3, sigma_v4, epsilon])
            scales = [1, 100, 100, 100, 100, 1]

        if ndenbins == 5:
            ndim = 7
            sigma_v1 = 360
            sigma_v2 = 360
            sigma_v3 = 360
            sigma_v4 = 360
            sigma_v5 = 360
            start_params = np.array([fs8, sigma_v1, sigma_v2, sigma_v3, sigma_v4, sigma_v5, epsilon])
            scales = [1, 100, 100, 100, 100, 100, 1]

        p0 = [start_params + 1e-2 * np.random.randn(ndim) * scales for i in range(nwalkers)]

        print('Running emcee with the following parameters:')
        print('nwalkers: ' + str(nwalkers))
        print('ndim: ' + str(ndim))
        print('niter: ' + str(niter))
        print('backend: ' + args.backend_name)
        print('Running in {} CPUs'.format(args.ncores))

        backend = emcee.backends.HDFBackend(args.backend_name)
        backend.reset(nwalkers, ndim)

        with Pool(processes=args.ncores) as pool:

            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            log_probability,
                                            backend=backend,
                                            pool=pool)
                                            
            sampler.run_mcmc(p0, niter, progress=True)

else:
    print('Single fit.')
    args.smin = float(args.smin)
    args.smax = float(args.smax)
    if args.model == 1:
        model = SingleFit(
                        xi_r_filename=args.xi_r,
                        sv_filename=args.sv_r,
                        xi_smu_filename=args.xi_smu,
                        covmat_filename=args.covmat,
                        full_fit=args.full_fit,
                        model=args.model,
                        const_sv=args.const_sv,
                        model_as_truth=args.model_as_truth,
                        smin=args.smin,
                        smax=args.smax,
                        vr_coupling=args.vr_coupling)

        ndim = 3
        nwalkers = args.ncores
        niter = 10000

        fs8 = 0.472
        sigma_v = 515
        epsilon = 1.0

        start_params = np.array([fs8, sigma_v, epsilon])
        scales = [1, 100, 1]

        p0 = [start_params + 1e-2 * np.random.randn(ndim) * scales for i in range(nwalkers)]

        print('Running emcee with the following parameters:')
        print('nwalkers: ' + str(nwalkers))
        print('ndim: ' + str(ndim))
        print('niter: ' + str(niter))
        print('backend: ' + args.backend_name)
        print('Running in {} CPUs'.format(args.ncores))

        backend = emcee.backends.HDFBackend(args.backend_name)
        backend.reset(nwalkers, ndim)

        with Pool(processes=args.ncores) as pool:

            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            log_probability,
                                            backend=backend,
                                            pool=pool)
            sampler.run_mcmc(p0, niter, progress=True)






