import time
import datetime
import likelihood
import plotting
from scipy.stats import norm
import os

os.environ["OMP_NUM_THREADS"] = "1"

def main():

    pipeline_variables_path = \
        '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim/set_config/set_variables.ini'

    now = datetime.datetime.now()
    # Perform likelihood analysis

    start_time = time.time()

    print('Performing likelihood analysis...')
    # print(now)

    # If using the analytic covariance matrix, generate from parameter values defined in variables config file
    covariance_matrix_type = 'numerical'     # Must be 'analytic' or 'numerical'

    # if covariance_matrix_type == 'analytic':
    #     likelihood.analytic_covariance.execute(pipeline_variables_path=pipeline_variables_path)

    # Need to add in fitting parameters and priors here instead of in sampler.py
    # Just define a tuple with the parameter and dist variable for the shape/width of the prior. The prior can itself
    # be a tuple (for a uniform prior) or e.g. scipy.stats.norm for a Gaussian prior
    priors = []

    # priors.append(("w0", (-1.5, -0.5))) # for a shape to the prior, this could be e.g ["w0", scipy.stats.norm(loc=2.0, scale=0.5)] as in nautilus documentation
    # priors.append(("wa", (-0.5, 0.5)))
    # priors.append(("Omega_m", (0.2, 0.4)))
    # priors.append(("h", (0.5, 0.8)))

    priors.append(("w0", (-1.25, -0.75)))  # for a shape to the prior, this could be e.g ["w0", scipy.stats.norm(loc=2.0, scale=0\
    priors.append(("wa", (-0.5, 0.5)))
    # priors.append(("Omega_m", (0.2, 0.4)))
    # priors.append(("h", (0.5, 0.8)))
    # priors.append(("Omega_b", (0.02, 0.08)))
    # priors.append(("n_s", (0.8, 1.2)))
    # priors.append(("sigma8", (0.75, 0.9)))

    # priors.append(("w0", (-1.15, -0.85)))
    # priors.append(("wa", (-0.3, 0.3)))
    # priors.append(("Omega_m", (0.27, 0.35)))
    # priors.append(("h", (0.55, 0.8)))
    # priors.append(("Omega_b", (0.02, 0.07)))
    # priors.append(("n_s", (0.9, 1.02)))
    # priors.append(("sigma8", (0.81, 0.87)))

    # priors.append(("_b1", (0.5, 3)))
    # priors.append(("_b2", (-2, 2)))
    # priors.append(("_bs", (-2, 2)))

    # priors.append(("_A1", (-8, 8)))
    # priors.append(("_A2", (-8, 8)))
    # priors.append(("_bTA", (-6, 6)))
    # priors.append(("_eta1", (-6, 6)))
    # priors.append(("_eta2", (-6, 6)))

    # priors.append(("_s0", (-5, 5)))
    # priors.append(("_s1", (-5, 5)))
    # priors.append(("_s2", (-5, 5)))
    # priors.append(("_s3", (-5, 5)))

    # priors.append(("_m_1", (-2, 2)))
    # priors.append(("_m_2", (-2, 2)))
    # priors.append(("_m_3", (-2, 2)))
    # priors.append(("_m_4", (-2, 2)))
    # priors.append(("_m_5", (-2, 2)))
    # priors.append(("_m_6", (-2, 2)))

    # priors.append(("_Dz_1", norm(loc=0, scale=0.01)))
    # priors.append(("_Dz_2", norm(loc=0, scale=0.01)))
    # priors.append(("_Dz_3", norm(loc=0, scale=0.01)))
    # priors.append(("_Dz_4", norm(loc=0, scale=0.01)))
    # priors.append(("_Dz_5", norm(loc=0, scale=0.01)))
    # priors.append(("_Dz_6", norm(loc=0, scale=0.01)))

    # For a constant global galaxy bias, we can have e.g.
    # priors.append(('_b1', (0,3)))   # for a constant global galaxy bias

    # Or we specify b1 as a constant that is different for each bin, e.g for a 3 bin analysis.
    # priors.append(('_b1_1', (0,3)))   # for a constant galaxy bias in bin 1
    # priors.append(('_b1_2', (0,3)))   # for a constant galaxy bias in bin 2
    # priors.append(('_b1_3', (0,3)))   # for a constant galaxy bias in bin 3
    # and in this case we need to have bi_marg=True in the sampler args below

    # Can also repeat this for m-bias marginalisation. If a global m-bias (independent of tomographic bin)
    # then we have to do
    # priors.append(('_m', (0,3)))   # for a constant global m-bias
    # Otherwise, we set the m-bias per tomographic bin, i.e
    # priors.append(('_m_1', (0,3)))   # for a constant m-bias in bin 1
    # priors.append(('_m_2', (0,3)))   # for a constant m-bias in bin 2
    # priors.append(('_m_3', (0,3)))   # for a constant m-bias in bin 3
    # and in this case we need to set mi_marg=True in the sampler args below

    # Can also repeat this for the A1 amplitude of IA TATT/NLA model. If a global A1 value (independent of tomographic
    # bin) then we have to do
    # priors.append(('_A1', dist=(0,3))   # for a constant global m-bias
    # Otherwise, we set the A1 value per tomographic bin, i.e
    # priors.append(('_A1_1', (0,3))   # for a constant A1 in bin 1
    # priors.append(('_A1_2', (0,3))   # for a constant A1 in bin 2
    # priors.append(('_A1_3', (0,3))   # for a constant A1 in bin 3
    # and in this case we need to set A1i_marg=True in the sampler args below

    # For marginalisation over shift paramaters for the n(z) we have to add a shift parameter per bin, i.e
    # priors.append(('_Dz_1', (0,3)))   # for a constant m-bias in bin 1
    # priors.append(('_Dz_2', (0,3)))   # for a constant m-bias in bin 2
    # priors.append(('_Dz_3', (0,3)))   # for a constant m-bias in bin 3
    # and we need to set Dzi_marg=True in the sampler args below

    likelihood.sampler_new.execute(
        pipeline_variables_path,
        covariance_matrix_type=covariance_matrix_type,
        priors=priors,
        checkpoint_filename='Cosmology_TEST.hdf5',
        bi_marg=False,
        mi_marg=False,
        Dzi_marg=False,
        A1i_marg=False
    )
    '''
    # For plotting
    sampler1 = likelihood.sampler.execute(
        pipeline_variables_path,
        covariance_matrix_type=covariance_matrix_type,
        priors=priors,
        checkpoint_filename='Cosmology_6x2pt_analytic.hdf5',
        bi_marg=False,
        mi_marg=False,
        Dzi_marg=False,
        A1i_marg=False
    )

    sampler2 = likelihood.sampler.execute(
        pipeline_variables_path,
        covariance_matrix_type=covariance_matrix_type,
        priors=priors,
        checkpoint_filename='Cosmology_6x2pt_numerical.hdf5',
        bi_marg=False,
        mi_marg=False,
        Dzi_marg=False,
        A1i_marg=False
    )

    sampler3 = likelihood.sampler.execute(
        pipeline_variables_path,
        covariance_matrix_type=covariance_matrix_type,
        priors=priors,
        checkpoint_filename='Cosmology_3x2pt_analytic.hdf5',
        bi_marg=False,
        mi_marg=False,
        Dzi_marg=False,
        A1i_marg=False
    )

    sampler4 = likelihood.sampler.execute(
        pipeline_variables_path,
        covariance_matrix_type=covariance_matrix_type,
        priors=priors,
        checkpoint_filename='Cosmology_3x2pt_numerical.hdf5',
        bi_marg=False,
        mi_marg=False,
        Dzi_marg=False,
        A1i_marg=False
    )

    plotting.plot_posteriors.nautilus_posterior_plotting(sampler1, sampler2, sampler3, sampler4)

    print('Done')
    # print(datetime.datetime.now())
    # print("--- %s seconds ---" % (time.time() - start_time))
    '''

if __name__ == '__main__':
    main()
