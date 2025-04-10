"""
Convert the 3x2pt data calculated from CosmoSIS into the correct file + field conventions required for the map
generation by Flask
"""

import os
import configparser
import numpy as np


def conversion_config(pipeline_variables_path):

    """
    Set up a config dictionary to execute the CosmoSIS-Flask file conversion based on pipeline parameters
    specified in a given input variables file

    Parameters
    ----------
    pipeline_variables_path (str):  Path to location of set_variables_cat.ini file

    Returns
    -------
    Dictionary of pipeline and file conversion parameters
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    nbins = int(float(config['create_nz']['N_ZBIN']))
    bins = np.arange(1, nbins + 1, 1)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])

    z_boundaries_filename = 'z_boundaries.txt'
    z_boundaries = np.loadtxt(save_dir + z_boundaries_filename)
    z_boundary_columns = np.transpose(z_boundaries)
    z_boundaries_low = z_boundary_columns[0][0:-1]
    z_boundaries_mid = z_boundary_columns[1][0:-1]
    z_boundaries_high = z_boundary_columns[2][0:-1]

    # Prepare config dictionary
    config_dict = {
        'nbins': nbins,
        'bins': bins,
        'save_dir': save_dir,
        'z_boundaries_low': z_boundaries_low,
        'z_boundaries_mid': z_boundaries_mid,
        'z_boundaries_high': z_boundaries_high
    }

    return config_dict


def open_data(data_file):

    """
    Convenience function to open data (in CosmoSIS format) and store into array

    Parameters
    ----------
    data_file (str):    Path to data file

    Returns
    -------
    Array of data stored on disk
    """

    data_arr = []

    with open(data_file) as f:
        for line in f:
            column = line.split()
            if not line.startswith('#'):
                data_i = float(column[0])
                data_arr.append(data_i)

    return data_arr


def conv_fields(config_dict):

    """
    Convert the 3x2pt data files output from CosmoSIS into the correct field + naming conventions for Flask

    Parameters
    ----------
    config_dict (dict): Dictionary of pipeline and field parameters for the 3x2pt simulation
    """

    save_dir = config_dict['save_dir']
    nbins = config_dict['nbins']
    bins = config_dict['bins']
    z_boundaries_low = config_dict['z_boundaries_low']
    z_boundaries_mid = config_dict['z_boundaries_mid']
    z_boundaries_high = config_dict['z_boundaries_high']

    flask_data_dir = save_dir + 'flask/data/'
    fiducial_data_dir = save_dir + 'fiducial_cosmology/'

    if not os.path.exists(flask_data_dir):
        os.makedirs(flask_data_dir)

    cmb_kk_txt_file = fiducial_data_dir + 'cmbkappa_cl/bin_1_1.txt'
    cmb_kk_cl = open_data(cmb_kk_txt_file)

    cmb_kk_ell_file = fiducial_data_dir + 'cmbkappa_cl/ell.txt'
    cmb_kk_ell = open_data(cmb_kk_ell_file)

    cmb_kk_file_name = '/Cl-f1z{}f1z{}.dat'.format(nbins+1, nbins+1)
    cmb_kk__save_file_name = flask_data_dir + cmb_kk_file_name
    np.savetxt(cmb_kk__save_file_name, np.transpose([cmb_kk_ell, cmb_kk_cl]), fmt='%.18f')

    for i in bins:

        ell_file = fiducial_data_dir + 'shear_cmbkappa_cl/ell.txt'
        ell = open_data(ell_file)

        shear_cmbkappa_txt_file = fiducial_data_dir + 'shear_cmbkappa_cl/bin_{}_1.txt'.format(i)
        shear_cmbkappa_cl = open_data(shear_cmbkappa_txt_file)

        shear_cmbkappa_file_name = '/Cl-f1z{}f1z{}.dat'.format(nbins+1, i)
        shear_cmbkappa_save_file_name = flask_data_dir + shear_cmbkappa_file_name
        np.savetxt(shear_cmbkappa_save_file_name, np.transpose([ell, shear_cmbkappa_cl]), fmt='%.18f')

        galaxy_cmbkappa_txt_file = fiducial_data_dir + 'galaxy_cmbkappa_cl/bin_{}_1.txt'.format(i)
        galaxy_cmbkappa_cl = open_data(galaxy_cmbkappa_txt_file)

        galaxy_cmbkappa_file_name = '/Cl-f2z{}f1z{}.dat'.format(i, nbins+1)
        galaxy_cmbkappa_save_file_name = flask_data_dir + galaxy_cmbkappa_file_name
        np.savetxt(galaxy_cmbkappa_save_file_name, np.transpose([ell, galaxy_cmbkappa_cl]), fmt='%.18f')

        for j in bins:

            ell_file = fiducial_data_dir + 'galaxy_shear_cl/ell.txt'
            ell = open_data(ell_file)

            gal_shear_txt_file = fiducial_data_dir + 'galaxy_shear_cl/bin_{}_{}.txt'.format(i, j)
            gal_shear_cl = open_data(gal_shear_txt_file)

            gal_shear_file_name = '/Cl-f2z{}f1z{}.dat'.format(i, j)
            gal_shear_save_file_name = flask_data_dir + gal_shear_file_name
            np.savetxt(gal_shear_save_file_name, np.transpose([ell, gal_shear_cl]), fmt='%.18f')

            if i >= j:
                ell_file = fiducial_data_dir + 'shear_cl/ell.txt'
                ell = open_data(ell_file)

                shear_txt_file = fiducial_data_dir + 'shear_cl/bin_{}_{}.txt'.format(i, j)
                shear_cl = open_data(shear_txt_file)

                shear_file_name = '/Cl-f1z{}f1z{}.dat'.format(i, j)
                shear_save_file_name = flask_data_dir + shear_file_name
                np.savetxt(shear_save_file_name, np.transpose([ell, shear_cl]), fmt='%.18f')

                gal_txt_file = fiducial_data_dir + 'galaxy_cl/bin_{}_{}.txt'.format(i, j)
                gal_cl = open_data(gal_txt_file)

                gal_file_name = '/Cl-f2z{}f2z{}.dat'.format(i, j)
                gal_save_file_name = flask_data_dir + gal_file_name
                np.savetxt(gal_save_file_name, np.transpose([ell, gal_cl]), fmt='%.18f')

    gal_field = 1
    wl_field = 2

    field_nos = np.zeros(nbins)

    gal_field_nos = field_nos + 2
    wl_field_nos = field_nos + 1

    z_bin_number = bins
    mean = np.zeros(nbins)
    shift = np.zeros(nbins)
    shift = shift + 1

    field_type = np.zeros(nbins)
    gal_field_type = field_type + gal_field
    wl_field_type = field_type + wl_field

    field_info_gal = [
        gal_field_nos,
        z_bin_number,
        mean,
        shift,
        gal_field_type,
        z_boundaries_low,
        z_boundaries_high
    ]

    field_info_wl = [
        wl_field_nos,
        z_bin_number,
        mean,
        shift,
        wl_field_type,
        z_boundaries_low,
        z_boundaries_high
    ]

    field_info_cmb_cl = [
        [1],                      # Weak lensing (for CMB) index number
        [nbins+1],                # Bin number - essentially treat as additional lensing bin
        [0],                      # Mean 0 (not using lognormal fields)
        [0],                      # Shift 0 (not using lognormal fields)
        [2],                      # Weak lensing (for CMB) field type - designation for FLASK
        [z_boundaries_low[0]],    # Minimum redshift observed
        [1100]                    # Redshift of last scattering surface for CMB
    ]

    field_info_3x2pt = np.concatenate((field_info_wl, field_info_gal), axis=1)
    field_info_6x2pt = np.concatenate((field_info_wl, field_info_cmb_cl, field_info_gal), axis=1)

    field_info_3x2pt_filename = 'field_info_3x2pt.dat'
    field_info_6x2pt_filename = 'field_info_6x2pt.dat'
    field_info_wl_filename = 'field_info_wl.dat'
    field_info_gal_filename = 'field_info_gal.dat'

    np.savetxt(
        flask_data_dir + field_info_3x2pt_filename,
        np.transpose(field_info_3x2pt),
        fmt=['%6i', '%6i', '%10.4f', '%10.4f', '%6i', '%10.4f', '%10.4f'],
        header='Field number, z bin number, mean, shift, field type, zmin, zmax\nTypes: 1-galaxies 2-lensing\n'
    )

    np.savetxt(
        flask_data_dir + field_info_6x2pt_filename,
        np.transpose(field_info_6x2pt),
        fmt=['%6i', '%6i', '%10.4f', '%10.4f', '%6i', '%10.4f', '%10.4f'],
        header='Field number, z bin number, mean, shift, field type, zmin, zmax\nTypes: 1-galaxies 2-lensing\n'
    )

    np.savetxt(
        flask_data_dir + field_info_wl_filename,
        np.transpose(field_info_wl),
        fmt=['%6i', '%6i', '%10.4f', '%10.4f', '%6i', '%10.4f', '%10.4f'],
        header='Field number, z bin number, mean, shift, field type, zmin, zmax\nTypes: 1-galaxies 2-lensing\n'
    )

    np.savetxt(
        flask_data_dir + field_info_gal_filename,
        np.transpose(field_info_gal),
        fmt=['%6i', '%6i', '%10.4f', '%10.4f', '%6i', '%10.4f', '%10.4f'],
        header='Field number, z bin number, mean, shift, field type, zmin, zmax\nTypes: 1-galaxies 2-lensing\n'
    )


def execute(pipeline_variables_path):

    """
    Generate and save the Flask 3x2pt field data files by reading in the pipeline variables file as environment
    variable, then setting up the config dictionary and converting the CosmoSIS field information saved on disk
    """

    conversion_config_dict = conversion_config(pipeline_variables_path=pipeline_variables_path)
    conv_fields(config_dict=conversion_config_dict)

