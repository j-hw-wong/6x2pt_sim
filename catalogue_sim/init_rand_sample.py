"""
Script to generate a random sample of N galaxies that follows a given p(z) distribution. Parameters are read in from
the 'set_variables_cat.ini' file, and the sample of galaxies is saved as 'Raw_Galaxy_Sample.hdf5' on disk.
"""

import os
import h5py
import configparser
import numpy as np
from random import choices


def pz_config(pipeline_variables_path):

    """
    Set up a config dictionary of cosmology/redshift parameters to generate a sample of galaxies

    Parameters
    ----------
    pipeline_variables_path : (str)
        Path to the 'set_variables_cat.ini' parameters file that exists within pipeline
        folder

    Returns
    -------
        Dictionary of config parameters for initialisation of random galaxies
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    # Constants for the galaxy n(z) probability distribution
    z0 = float(config['redshift_distribution']['Z0'])
    beta = float(config['redshift_distribution']['BETA'])

    # Set z-range to simulate over
    zmin = float(config['redshift_distribution']['ZMIN'])
    zmax = float(config['redshift_distribution']['ZMAX'])

    # Precision/step-size of z-range that is sampled over.
    dz = float(config['redshift_distribution']['DZ'])

    # No. galaxies to simulate
    sample_points = int(float(config['redshift_distribution']['NGAL']))

    photo_z_noise_mean = 0
    photo_z_noise_sigma = float(config['noise_cls']['SIGMA_PHOT'])

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])

    # Prepare config dictionary
    config_dict = {
        'z0': z0,
        'beta': beta,
        'zmin': zmin,
        'zmax': zmax,
        'dz': dz,
        'sample_points': sample_points,
        'save_dir': save_dir,
        'photo_z_noise_mean': photo_z_noise_mean,
        'photo_z_noise_sigma': photo_z_noise_sigma
    }

    return config_dict


def pz(z, z0, beta):

    """
    Takes as input an array of galaxy redshift values and generates a standard probability distribution

    Parameters
    ----------
    z : (array)
        Redshift values with which to return a probability distribution
    z0 : (float)
        Functional constant to normalise the redshift
    beta : (float)
        Exponential constant for redshift distribution

    Returns
    -------
        Array of the probability values at the given redshifts
    """

    return ((z/z0)**2)*np.exp(-1*((z/z0)**beta))


def generate_cat_err_sig(redshifts, lambda_1, lambda_2, sig):

    """
    Function to inject catastrophic photo-z errors into the redshift sample based on a confusion of two given
    wavelength lines and error distribution

    Parameters
    ----------
    redshifts : (arr)
        Array of galaxy redshifts to inject catastrophic photo-zs into
    lambda_1 : (float)
        Wavelength of first given spectral line
    lambda_2 : (float)
        Wavelength of second given spectral line
    sig : (float)
        Sigma spread describing the error distribution around where the pair confusion line is found

    Returns
    -------
        Array of galaxy redshifts with catastrophic photo-z errors
    """

    cat_z_mus = ((1+redshifts)*(lambda_1/lambda_2))-1
    return np.random.normal(cat_z_mus, sig*np.ones(len(cat_z_mus)), size=len(cat_z_mus))


def split_z_chunks(a, n):

    """
    Convenience function to split a chunk of galaxy redshifts into equal sub-chunks (used to distribute all pairs of
    photo-z confusion between)

    Parameters
    ----------
    a : (arr)
        Array of redshift values
    n : (int)
        Number of chunks to split data into

    Returns
    -------
        Array of n sub-samples that the original data array a has been split into
    """

    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]



def init_nz(config_dict):

    """
    Generate a random sample of N galaxies that follows the probability distribution pz and save to disk

    Parameters
    ----------
    config_dict : (dict)
        Dictionary of config parameters set up in pz_config


    Returns
    -------
        Saves sample of galaxies in hdf5 format
    """

    zmin = config_dict['zmin']
    zmax = config_dict['zmax']
    dz = config_dict['dz']

    z0 = config_dict['z0']
    beta = config_dict['beta']

    sample_points = config_dict['sample_points']

    save_dir = config_dict['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    photo_z_noise_mean = config_dict['photo_z_noise_mean']
    photo_z_noise_sigma = config_dict['photo_z_noise_sigma']

    z_sample = np.linspace(
        zmin,
        zmax,
        (round((zmax-zmin)/dz))+1
    )
    z_sample = z_sample.round(decimals=2)

    # Select n=sample_points number of galaxies randomly from p(z) distribution
    rnd_sample = choices(z_sample[0:-1], pz(z_sample[0:-1], z0=z0, beta=beta), k=sample_points)
    zs = np.round(np.asarray(rnd_sample), 2)

    phot_zs = []

    for k in range(len(z_sample)):
        id_arr = np.where(np.asarray(zs) == z_sample[k])
        zs_sub = zs[id_arr]
        zs_sub = zs_sub.round(decimals=2)

        photo_z_noise_sigmas = (1 + zs_sub) * photo_z_noise_sigma
        photo_z_noise_means = np.zeros(len(zs_sub)) + photo_z_noise_mean

        photo_z_noise = np.random.normal(photo_z_noise_means, photo_z_noise_sigmas, len(photo_z_noise_sigmas))
        obs_zs = zs_sub + photo_z_noise
        phot_zs = np.concatenate((phot_zs, obs_zs))

    true_zs = np.float32(zs)
    obs_gaussian_zs = np.float32(phot_zs)

    # Filename to save raw sample
    mock_cat_filename = 'Raw_Galaxy_Sample.hdf5'

    with h5py.File(save_dir + mock_cat_filename, 'w') as f:
        f.create_dataset("Redshift_z", data=obs_gaussian_zs)
        f.create_dataset("True_Redshift_z", data=true_zs)


def execute(pipeline_variables_path):

    """
    Generate the galaxy sample by reading in the pipeline variables file as environment variable, then setting up the
    config dictionary and initialising the n(z)
    """

    # pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    pz_config_dict = pz_config(pipeline_variables_path=pipeline_variables_path)
    init_nz(config_dict=pz_config_dict)


# if __name__ == '__main__':
#     main()
