"""
Create a tomographic n(z) based on measurement parameters supplied by 'set_variables_3x2pt_measurement.ini'. First, an
array of redshift boundary values for each bin is created and saved to disk, then the n(z) is measured using these
boundaries from the simulated catalogues.
"""

import os
import sys
import h5py
import configparser
import numpy as np
from collections import defaultdict


def nz_fromsim_config(pipeline_variables_path):

    """
    Set up a config dictionary to generate an n(z) distribution as measured from the simulations

    Parameters
    ----------
    pipeline_variables_path : (str)
        Path to location of pipeline variables file ('set_variables_3x2pt_measurement.ini')

    Returns
    -------
        Dictionary of n(z) parameters
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    zmin = float(config['redshift_distribution']['ZMIN'])
    zmax = float(config['redshift_distribution']['ZMAX'])

    # Precision/step-size of z-range that is sampled over.
    dz = float(config['redshift_distribution']['DZ'])

    nbins = int(float(config['redshift_distribution']['N_ZBIN']))
    bin_type = str(config['redshift_distribution']['ZBIN_TYPE'])

    nz_table_filename = str(config['redshift_distribution']['NZ_TABLE_NAME'])

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    realisations = int(float(config['simulation_setup']['REALISATIONS']))

    photo_z_noise_mean = 0
    photo_z_noise_sigma = float(config['noise_cls']['SIGMA_PHOT'])

    # cat_photo_z_frac = float(config['noise_cls']['CAT_PHOTO_Z_FRAC'])
    # cat_photo_z_sigma = float(config['noise_cls']['CAT_PHOTO_Z_SIGMA'])

    # Prepare config dictionary
    config_dict = {
        'zmin': zmin,
        'zmax': zmax,
        'dz': dz,
        'nbins': nbins,
        'bin_type': bin_type,
        'nz_table_filename': nz_table_filename,
        'save_dir': save_dir,
        'realisations': realisations,
        'photo_z_noise_mean': photo_z_noise_mean,
        'photo_z_noise_sigma': photo_z_noise_sigma
        # 'cat_photo_z_frac': cat_photo_z_frac,
        # 'cat_photo_z_sigma': cat_photo_z_sigma
    }

    return config_dict


def create_zbin_boundaries(config_dict):

    """
    Create a table of the redshift boundaries used for binning the galaxies in the simulated catalogues for the 3x2pt
    analysis, which is then saved to disk.

    Parameters
    ----------

    config_dict : (dict)
        Dictionary of pipeline and redshift distribution parameters used to generate the bin boundaries
        and overall n(z)

    Returns
    -------
        Array of the redshift bin boundaries evaluated for the given number of bins + binning configuration.
    """

    zmin = config_dict['zmin']
    zmax = config_dict['zmax']
    dz = config_dict['dz']
    nbins = config_dict['nbins']
    bin_type = config_dict['bin_type']
    save_dir = config_dict['save_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    z_boundaries_filename = 'z_boundaries.txt'

    if bin_type == 'EQUI_Z':
        z_boundaries_low = np.linspace(zmin, zmax, nbins + 1)
        z_boundaries_mid = z_boundaries_low + (((zmax - zmin) / nbins) / 2)
        z_boundaries_high = z_boundaries_mid + (((zmax - zmin) / nbins) / 2)

        z_boundaries = [z_boundaries_low, z_boundaries_mid, z_boundaries_high]
        np.savetxt(save_dir + z_boundaries_filename,
                   np.transpose(z_boundaries),
                   fmt=['%.2f', '%.2f', '%.2f'])

    elif bin_type == 'EQUI_POP':

        # Need to generate a rnd_sample from the measured n(z), i.e. n(z)*z for each z

        mock_cat_filename = 'Raw_Galaxy_Sample.hdf5'
        mock_cat = save_dir + mock_cat_filename
        with h5py.File(mock_cat, "r") as f:
            rnd_sample = f['Redshift_z'][()]

        # rnd_sample = np.load(save_dir)  # Placeholder for now!

        rnd_sample = np.round(rnd_sample, 2)
        rnd_sample = rnd_sample[rnd_sample<=(zmax-dz)]
        sorted_sample = np.sort(rnd_sample)
        split_sorted_sample = np.array_split(sorted_sample, nbins)
        z_boundaries_low = [zmin]
        z_boundaries_high = []
        for i in range(nbins):
            z_boundaries_low.append(split_sorted_sample[i][-1])
            z_boundaries_high.append(split_sorted_sample[i][-1])
        z_boundaries_high.append(z_boundaries_high[-1] + dz)
        z_boundaries_mid = []
        for i in range(len(z_boundaries_low)):
            z_boundaries_mid.append(round(np.mean([z_boundaries_low[i], z_boundaries_high[i]]), 2))

        z_boundaries = [z_boundaries_low, z_boundaries_mid, z_boundaries_high]
        np.savetxt(save_dir + z_boundaries_filename,
                   np.transpose(z_boundaries),
                   fmt=['%.2f', '%.2f', '%.2f'])

    elif bin_type == 'EQUI_D':
        # we need to go back to the directory of the simulation and into the cosmosis/distances file for the
        # comoving distance as a function of z. Then cut out the range that corresponds to the z_range of observation
        # then define equally spaced boundaries in d-space and take the corresponding z boundaries

        z_distances = np.loadtxt(save_dir + 'cosmosis/distances/z.txt')
        d_m = np.loadtxt(save_dir + 'cosmosis/distances/d_m.txt')
        zmin_id = np.where((z_distances == zmin))[0][0]
        zmax_id = np.where((z_distances == zmax))[0][0]

        d_m_observed = d_m[zmin_id:zmax_id+1]
        z_observed = z_distances[zmin_id:zmax_id+1]
        d_m_range = d_m_observed[-1]-d_m_observed[0]
        d_m_separation = d_m_range/nbins
        z_boundaries_low = [zmin]
        z_boundaries_high = []

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        for i in range(nbins):
            obs_id = find_nearest(d_m_observed, d_m_observed[0] + (d_m_separation*(i+1)))
            z_boundaries_low.append(z_observed[obs_id])
            z_boundaries_high.append(z_observed[obs_id])
        z_boundaries_high.append(z_boundaries_high[-1] + dz)

        z_boundaries_mid = []

        for i in range(len(z_boundaries_low)):
            z_boundaries_mid.append(round(np.mean([z_boundaries_low[i], z_boundaries_high[i]]), 2))

        z_boundaries = [z_boundaries_low, z_boundaries_mid, z_boundaries_high]

        np.savetxt(save_dir + z_boundaries_filename,
                   np.transpose(z_boundaries),
                   fmt=['%.2f', '%.2f', '%.2f'])

    else:
        print(bin_type)
        print("Bin Type Not Recognised! Must be 'EQUI_Z', 'EQUI_POP', or 'EQUI_D'")
        sys.exit()

    return np.asarray(z_boundaries)

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


def generate_nz(config_dict):

    """
    Execute the compilation of the final master catalogue. First load in galaxy pixel indices, assign the shear k, y1,
    y2 values based on the shear field maps at the same redshift slice, then inject both Gaussian and catastrophic
    photo-z errors.
    """

    # pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    # compile_cat_config_dict = compile_cat_config(pipeline_variables_path=pipeline_variables_path)

    save_dir = config_dict['save_dir']
    zmin = config_dict['zmin']
    zmax = config_dict['zmax']
    dz = config_dict['dz']
    nz_table_filename = config_dict['nz_table_filename']
    nbins = config_dict['nbins']
    photo_z_noise_mean = config_dict['photo_z_noise_mean']
    photo_z_noise_sigma = config_dict['photo_z_noise_sigma']

    # cat_photo_z_frac = config_dict['cat_photo_z_frac']
    # cat_photo_z_sigma = config_dict['cat_photo_z_sigma']

    dat = h5py.File(save_dir + 'Raw_Galaxy_Sample.hdf5')
    true_zs = np.array(dat.get('True_Redshift_z'))
    obs_gaussian_zs = np.array(dat.get('Redshift_z'))
    dat.close()

    ############
    # Here is some code from the full catalogue simulation which simulates the injection of catastrophic photo-z errors.
    # I am not sure if it makes any sense to inject these into the n(z) for the faster map-only sims (since this can't
    # really be introduced as a bias in the same way. Either way, I will keep the code here for now in case we can
    # figure out how it could be useful in the future
    ############
    #
    # # Inject some catastrophic photo-z errors
    #
    # final_phot_zs = np.copy(phot_zs)  # will inject catastrophic errors into this column
    # final_phot_zs = np.float32(final_phot_zs)
    #
    # if cat_photo_z_frac != 0:
    #     # Let's inject some catastrophic photo-zs
    #
    #     # Angstroms
    #     Ly_alpha = 1216
    #     Ly_break = 912
    #     Balmer = 3700
    #     D4000 = 4000
    #
    #     break_rfs = [Ly_alpha, Ly_alpha]  # , Ly_break, Ly_break] - could include more pair confusions
    #     break_catas = [Balmer, D4000]  # , Balmer, D4000] - could include more pair confusions
    #
    #     rand_zs_ids = choices(range(len(zs)), k=int(len(zs) * cat_photo_z_frac))
    #     rand_zs_ids = np.asarray(rand_zs_ids)
    #     rand_zs_ids_chunks = split_z_chunks(rand_zs_ids, 4)
    #
    #     final_phot_zs[rand_zs_ids_chunks[0]] = generate_cat_err_sig(zs[rand_zs_ids_chunks[0]], break_rfs[0],
    #                                                                 break_catas[0], cat_photo_z_sigma)
    #     final_phot_zs[rand_zs_ids_chunks[1]] = generate_cat_err_sig(zs[rand_zs_ids_chunks[1]], break_rfs[1],
    #                                                                 break_catas[1], cat_photo_z_sigma)
    #     final_phot_zs[rand_zs_ids_chunks[2]] = generate_cat_err_sig(zs[rand_zs_ids_chunks[2]], break_catas[0],
    #                                                                 break_rfs[0], cat_photo_z_sigma)
    #     final_phot_zs[rand_zs_ids_chunks[3]] = generate_cat_err_sig(zs[rand_zs_ids_chunks[3]], break_catas[1],
    #                                                                 break_rfs[1], cat_photo_z_sigma)
    #

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    z_boundaries_filename = 'z_boundaries.txt'
    z_boundaries = np.loadtxt(save_dir + z_boundaries_filename)

    z_boundaries_low = z_boundaries[:,0][0:-1]
    z_boundaries_mid = z_boundaries[:,1][0:-1]
    z_boundaries_high = z_boundaries[:,2][0:-1]

    z_boundaries_low = np.round(z_boundaries_low, 2)
    z_boundaries_mid = np.round(z_boundaries_mid, 2)
    z_boundaries_high = np.round(z_boundaries_high, 2)

    # Let's make the n(z) here - useful for making theoretical predictions and inference analysis
    sub_hist_bins = np.linspace(
        zmin,
        zmax + dz,
        (round((zmax + dz - zmin) / dz)) + 1
    )

    hists = defaultdict(list)
    for b in range(nbins):
        hists["BIN_{}".format(b + 1)] = []

    if dz == 0.1:
        obs_gaussian_zs = np.around(obs_gaussian_zs, decimals=1)
    elif dz == 0.01:
        obs_gaussian_zs = np.around(obs_gaussian_zs, decimals=2)

    for b in range(nbins):
        bin_pop = true_zs[np.where((obs_gaussian_zs >= z_boundaries_low[b]) & (obs_gaussian_zs < z_boundaries_high[b]))[0]]
        bin_hist = np.histogram(bin_pop, bins=int(np.rint((zmax + dz - zmin) / dz)), range=(zmin, zmax))[0]
        hists["BIN_{}".format(b + 1)].append(bin_hist)

    nz = []
    nz.append(sub_hist_bins[0:-1])

    for b in range(nbins):
        iter_hist_sample = hists["BIN_{}".format(b + 1)][0]
        nz.append(iter_hist_sample)

    final_cat_tab = np.asarray(nz)

    if zmin != 0:
        final_cat_tab = np.transpose(final_cat_tab)
        pad_vals = int((zmin-0)/dz)
        for i in range(pad_vals):
            z_pad = np.array([zmin-((i+1)*dz)])
            pad_arr = np.concatenate((z_pad, np.zeros(nbins)))
            final_cat_tab = np.vstack((pad_arr, final_cat_tab))

        final_cat_tab = np.transpose(final_cat_tab)
    '''
    final_cat_tab = np.transpose(final_cat_tab)
    zmax_pad = np.concatenate((np.array([zmax+dz]), np.zeros(nbins)))
    final_cat_tab = np.vstack((final_cat_tab, zmax_pad))
    final_cat_tab = np.transpose(final_cat_tab)
    '''

    np.savetxt(save_dir + nz_table_filename,
               np.transpose(final_cat_tab))

def execute(pipeline_variables_path):

    """
    Generate the n(z) measured from the simulated catalogues. First set up the config dictionary, then create the
    bin boundaries array for the chosen tomogaphy, then save n(z) to disk and plot.
    """

    # pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    # Create 'Observed Redshift'
    config_dict = nz_fromsim_config(pipeline_variables_path=pipeline_variables_path)
    create_zbin_boundaries(config_dict=config_dict)
    generate_nz(config_dict=config_dict)

# if __name__ == '__main__':
#     main()
