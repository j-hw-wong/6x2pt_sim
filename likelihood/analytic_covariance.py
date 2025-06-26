import os
import sys
import configparser
import likelihood.convert_noise as convert_noise
from likelihood.gaussian_pcl_covariance import get_1x2pt_cov_pcl, get_3x2pt_cov_pcl, get_6x2pt_cov_pcl


def execute(pipeline_variables_path):

    # pipeline_variables_path = \
    #     '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim/set_config/set_variables.ini'

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    # realisations = int(config['simulation_setup']['REALISATIONS'])

    nside = int(config['simulation_setup']['NSIDE'])
    nbins = int(config['redshift_distribution']['N_ZBIN'])
    # npix = hp.nside2npix(nside)

    # These lmin, lmax out could be changed - lrange that is measured out from Healpix maps
    # raw_pcl_lmin_out = 0
    # raw_pcl_lmax_out = int(float(config['simulation_setup']['INPUT_ELL_MAX']))
    #
    # sigma_phot = float(config['photo_z']['SIGMA_PHOT'])
    # sigma_e = float(config['shape_noise']['SIGMA_SHEAR'])

    lss_mask_path = str(config['measurement_setup']['PATH_TO_MASK'])
    cmb_mask_path = str(config['measurement_setup']['PATH_TO_CMB_MASK'])
    # nz_dat = np.loadtxt(save_dir + str(config['redshift_distribution']['NZ_TABLE_NAME']))

    input_lmin = int(float(config['simulation_setup']['INPUT_ELL_MIN']))
    input_lmax = int(float(config['simulation_setup']['INPUT_ELL_MAX']))

    output_lmin_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_EE']))
    output_lmax_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_EE']))

    output_lmin_galaxy_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_NE']))
    output_lmax_galaxy_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_NE']))

    output_lmin_galaxy = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_NN']))
    output_lmax_galaxy = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_NN']))

    output_lmin_cmbkk = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_CMBKK']))
    output_lmax_cmbkk = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_CMBKK']))

    output_lmin_cmbkk_galaxy = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_CMBKK_N']))
    output_lmax_cmbkk_galaxy = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_CMBKK_N']))

    output_lmin_cmbkk_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_CMBKK_E']))
    output_lmax_cmbkk_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_CMBKK_E']))

    n_bandpowers = int(float(config['measurement_setup']['N_BANDPOWERS']))
    bandpower_spacing = str(config['measurement_setup']['BANDPOWER_SPACING'])

    obs_spec = str(config['measurement_setup']['OBS_TYPE'])
    obs_field = str(config['measurement_setup']['FIELD'])

    # Need to convert partial sky noise Cls to full sky equivalent for use in the improved NKA
    convert_noise.execute(pipeline_variables_path=pipeline_variables_path)

    analytic_cov_dir = save_dir + 'analytic_covariance/'

    if not os.path.exists(analytic_cov_dir):
        os.makedirs(analytic_cov_dir)

    if obs_spec == '1X2PT':
        if obs_field == 'E':
            output_lmax = output_lmax_shear
            output_lmin = output_lmin_shear
            mask_path = lss_mask_path

        elif obs_field == 'N':
            output_lmax = output_lmax_galaxy
            output_lmin = output_lmin_galaxy
            mask_path = lss_mask_path

        elif obs_field == 'EK':
            output_lmax = output_lmax_cmbkk_shear
            output_lmin = output_lmin_cmbkk_shear
            mask_path = lss_mask_path

        elif obs_field == 'NK':
            output_lmax = output_lmax_cmbkk_galaxy
            output_lmin = output_lmin_cmbkk_galaxy
            mask_path = lss_mask_path

        elif obs_field == 'K':
            output_lmax = output_lmax_cmbkk
            output_lmin = output_lmin_cmbkk
            mask_path = cmb_mask_path

        else:
            sys.exit()

        get_1x2pt_cov_pcl(
            n_zbin=nbins,
            signal_path=save_dir + 'fiducial_cosmology',
            noise_path=save_dir + 'NKA_noise_cls',
            field=obs_field,
            lmax_in=input_lmax,
            lmin_in=input_lmin,
            lmax_out=output_lmax,
            lmin_out=output_lmin,
            noise_lmin=input_lmin,
            mask_path=mask_path,
            nside=nside,
            save_block_filemask=analytic_cov_dir + 'cov_spec1_{spec1_idx}_spec2_{spec2_idx}.npz',
            n_bp=n_bandpowers,
            bandpower_spacing=bandpower_spacing,
            save_cov_filemask=analytic_cov_dir + 'cov_{n_bp}bp.npz')

    elif obs_spec == '3X2PT':

        get_3x2pt_cov_pcl(
            n_zbin=nbins,
            signal_path=save_dir + 'fiducial_cosmology/',
            noise_path=save_dir + 'NKA_noise_cls/',
            lmax_in=input_lmax,
            lmin_in=input_lmin,
            lmax_out_nn=output_lmax_galaxy,
            lmin_out_nn=output_lmin_galaxy,
            lmax_out_ne=output_lmax_galaxy_shear,
            lmin_out_ne=output_lmin_galaxy_shear,
            lmax_out_ee=output_lmax_shear,
            lmin_out_ee=output_lmin_shear,
            noise_lmin=input_lmin,
            mask_path=lss_mask_path,
            nside=nside,
            save_filemask=analytic_cov_dir + 'cov_spec1_{spec1_idx}_spec2_{spec2_idx}.npz',
            n_bp=n_bandpowers,
            bandpower_spacing=bandpower_spacing,
            cov_filemask=analytic_cov_dir + 'cov_{n_bp}bp.npz')

    elif obs_spec == '6X2PT':

        get_6x2pt_cov_pcl(
            n_zbin=nbins,
            signal_path=save_dir + 'fiducial_cosmology/',
            noise_path=save_dir + 'NKA_noise_cls/',
            lmax_in=input_lmax,
            lmin_in=input_lmin,
            lmax_out_nn=output_lmax_galaxy,
            lmin_out_nn=output_lmin_galaxy,
            lmax_out_ne=output_lmax_galaxy_shear,
            lmin_out_ne=output_lmin_galaxy_shear,
            lmax_out_ee=output_lmax_shear,
            lmin_out_ee=output_lmin_shear,
            lmax_out_kk=output_lmax_cmbkk,
            lmin_out_kk=output_lmin_cmbkk,
            lmax_out_kn=output_lmax_cmbkk_galaxy,
            lmin_out_kn=output_lmin_cmbkk_galaxy,
            lmax_out_ke=output_lmax_cmbkk_shear,
            lmin_out_ke=output_lmin_cmbkk_shear,
            noise_lmin=input_lmin,
            mask_path=lss_mask_path,
            cmb_mask_path=cmb_mask_path,
            nside=nside,
            save_filemask=analytic_cov_dir + 'cov_spec1_{spec1_idx}_spec2_{spec2_idx}.npz',
            n_bp=n_bandpowers,
            bandpower_spacing=bandpower_spacing,
            cov_filemask=analytic_cov_dir + 'cov_{n_bp}bp.npz')

