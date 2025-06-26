import os
import configparser
import numpy as np
import healpy as hp


def conversion_config(pipeline_variables_path):

    """
    Create a dictionary of parameters that will be useful to calculate measure Pseudo bandpowers (e.g. number of bins,
    realisations, mask etc.)

    Parameters
    ----------
    pipeline_variables_path (str):  Path to location of pipeline variables file ('set_variables_3x2pt_measurement.ini')

    Returns
    -------
    Dictionary of parameters used by this script to measure 3x2pt Pseudo bandpowers
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    nbins = int(config['redshift_distribution']['N_ZBIN'])
    nside = int(float(config['simulation_setup']['NSIDE']))
    mask_path = str(config['measurement_setup']['PATH_TO_MASK'])
    mask_path_cmb = str(config['measurement_setup']['PATH_TO_CMB_MASK'])

    input_lmin = int(float(config['simulation_setup']['INPUT_ELL_MIN']))
    input_lmax = int(float(config['simulation_setup']['INPUT_ELL_MAX']))

    # Prepare config dictionary
    config_dict = {
        'save_dir': save_dir,
        'nbins': nbins,
        'nside': nside,
        'mask_path': mask_path,
        'mask_path_cmb': mask_path_cmb,
        'input_lmin': input_lmin,
        'input_lmax': input_lmax
    }

    return config_dict


def execute(pipeline_variables_path):

    config_dict = conversion_config(pipeline_variables_path=pipeline_variables_path)

    save_dir = config_dict['save_dir']

    mask_path = config_dict['mask_path']
    mask_path_cmb = config_dict['mask_path_cmb']

    nbins = config_dict['nbins']

    ell_min = config_dict['input_lmin']
    ell_max = config_dict['input_lmax']

    fsky = np.mean(hp.read_map(mask_path))
    fsky_cmb = np.mean(hp.read_map(mask_path_cmb))

    galaxy_noise_dir = 'galaxy_cl/'
    galaxy_shear_noise_dir = 'galaxy_shear_cl/'
    shear_noise_dir = 'shear_cl/'
    galaxy_cmb_noise_dir = 'galaxy_cmbkappa_cl/'
    shear_cmb_noise_dir = 'shear_cmbkappa_cl/'
    cmb_noise_dir = 'cmbkappa_cl/'

    noise_cls_dirs = [
        galaxy_noise_dir, galaxy_shear_noise_dir, shear_noise_dir, galaxy_cmb_noise_dir, shear_cmb_noise_dir,
        cmb_noise_dir]

    for noise_dir in noise_cls_dirs:
        if not os.path.exists(f"{save_dir}NKA_noise_cls/{noise_dir}"):
            os.makedirs(f"{save_dir}NKA_noise_cls/{noise_dir}")

    cut_noise_cmb_cl = np.loadtxt(f"{save_dir}measured_noise_cls/{cmb_noise_dir}bin_1_1.txt")
    # assert all(item == cut_noise_cmb_cl[0] for item in cut_noise_cmb_cl)
    np.savetxt(
        f"{save_dir}NKA_noise_cls/{cmb_noise_dir}bin_1_1.txt",
        np.zeros(ell_max - ell_min + 1) + (cut_noise_cmb_cl[0]/fsky_cmb))

    for i in range(nbins):

        cut_noise_galaxy_cmb_cl = np.loadtxt(f"{save_dir}measured_noise_cls/{galaxy_cmb_noise_dir}bin_{i+1}_1.txt")
        assert all(item == cut_noise_galaxy_cmb_cl[0] for item in cut_noise_galaxy_cmb_cl)
        np.savetxt(
            f"{save_dir}NKA_noise_cls/{galaxy_cmb_noise_dir}bin_{i+1}_1.txt",
            np.zeros(ell_max - ell_min + 1) + (cut_noise_galaxy_cmb_cl[0] / (min(fsky, fsky_cmb))))

        cut_noise_shear_cmb_cl = np.loadtxt(f"{save_dir}measured_noise_cls/{shear_cmb_noise_dir}bin_{i+1}_1.txt")
        assert all(item == cut_noise_shear_cmb_cl[0] for item in cut_noise_shear_cmb_cl)
        np.savetxt(
            f"{save_dir}NKA_noise_cls/{shear_cmb_noise_dir}bin_{i+1}_1.txt",
            np.zeros(ell_max - ell_min + 1) + (cut_noise_shear_cmb_cl[0] / (min(fsky, fsky_cmb))))

        for j in range(nbins):
            # ggl
            cut_noise_ggl_cl = np.loadtxt(f"{save_dir}measured_noise_cls/{galaxy_shear_noise_dir}bin_{i+1}_{j+1}.txt")
            assert all(item == cut_noise_ggl_cl[0] for item in cut_noise_ggl_cl)
            np.savetxt(
                f"{save_dir}NKA_noise_cls/{galaxy_shear_noise_dir}bin_{i + 1}_{j + 1}.txt",
                np.zeros(ell_max - ell_min + 1) + cut_noise_ggl_cl[0]/fsky)

            if i >= j:
                # shear, clustering
                galaxy_cut_noise_cl = np.loadtxt(f"{save_dir}measured_noise_cls/{galaxy_noise_dir}bin_{i+1}_{j+1}.txt")
                assert all(item == galaxy_cut_noise_cl[0] for item in galaxy_cut_noise_cl)
                np.savetxt(
                    f"{save_dir}NKA_noise_cls/{galaxy_noise_dir}bin_{i+1}_{j+1}.txt",
                    np.zeros(ell_max - ell_min + 1) + (galaxy_cut_noise_cl[0] / fsky))

                shear_cut_noise_cl = np.loadtxt(f"{save_dir}measured_noise_cls/{shear_noise_dir}bin_{i+1}_{j+1}.txt")
                assert all(item == shear_cut_noise_cl[0] for item in shear_cut_noise_cl)
                np.savetxt(
                    f"{save_dir}NKA_noise_cls/{shear_noise_dir}bin_{i+1}_{j+1}.txt",
                    np.zeros(ell_max - ell_min + 1) + (shear_cut_noise_cl[0] / fsky))

