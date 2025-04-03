import os
import sys
import configparser
import numpy as np
import healpy as hp
import pymaster as nmt
from collections import defaultdict

def measure_pcls_config(pipeline_variables_path):
    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    # save_dir = str(config['measurement_setup']['MEASUREMENT_SAVE_DIR'])
    realisations = int(config['simulation_setup']['REALISATIONS'])

    nside = int(config['simulation_setup']['NSIDE'])
    nbins = int(config['redshift_distribution']['N_ZBIN'])
    npix = hp.nside2npix(nside)

    # These lmin, lmax out could be changed - lrange that is measured out from Healpix maps
    raw_pcl_lmin_out = 0
    raw_pcl_lmax_out = int(float(config['simulation_setup']['INPUT_ELL_MAX']))

    sigma_phot = float(config['noise_cls']['SIGMA_PHOT'])
    sigma_e = float(config['noise_cls']['SIGMA_SHEAR'])

    mask_path = str(config['measurement_setup']['PATH_TO_MASK'])
    nz_dat = np.loadtxt(save_dir + str(config['redshift_distribution']['NZ_TABLE_NAME']))

    # Prepare config dictionary
    config_dict = {
        'nside': nside,
        'nbins': nbins,
        'raw_pcl_lmin_out': raw_pcl_lmin_out,
        'raw_pcl_lmax_out': raw_pcl_lmax_out,
        'save_dir': save_dir,
        'realisations': realisations,
        'sigma_phot': sigma_phot,
        'sigma_e': sigma_e,
        'mask_path': mask_path,
        'nz_dat': nz_dat
    }

    return config_dict


def maps(config_dict, iter_no):

    save_dir = config_dict['save_dir']
    nbins = config_dict['nbins']
    nside = config_dict['nside']
    raw_pcl_lmin_out = config_dict['raw_pcl_lmin_out']
    raw_pcl_lmax_out = config_dict['raw_pcl_lmax_out']
    sigma_e = config_dict['sigma_e']
    nz_dat = config_dict['nz_dat']
    npix = hp.nside2npix(nside)
    pi = np.pi

    mask = hp.read_map(config_dict['mask_path'])
    ell_arr = np.arange(raw_pcl_lmin_out, raw_pcl_lmax_out + 1, 1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cut_sky_map_dicts = defaultdict(list)

    sky_coverage = hp.nside2pixarea(nside, degrees=True) * (npix) * ((pi / 180) ** 2)
    w_survey = np.count_nonzero(mask == 1) / npix

    unobserved_ids = np.where((mask == 0))[0]
    observed_ids = np.where((mask == 1))[0]

    for b in range(nbins):

        poisson_cls_theory_dir = save_dir + 'raw_noise_cls/galaxy_cl/iter_{}/'.format(iter_no)
        if not os.path.exists(poisson_cls_theory_dir):
            os.makedirs(poisson_cls_theory_dir)

        shape_noise_cls_dir = save_dir + 'raw_noise_cls/shear_cl/iter_{}/'.format(iter_no)
        if not os.path.exists(shape_noise_cls_dir):
            os.makedirs(shape_noise_cls_dir)

        dens_map = hp.read_map(save_dir + 'flask/output/iter_{}/map-f2z{}.fits'.format(iter_no, b+1))
        k_map = hp.read_map(save_dir + 'flask/output/iter_{}/map-f1z{}.fits'.format(iter_no, b+1))
        y1_map = hp.read_map(save_dir +
                             'flask/output/iter_{}/kappa-gamma-f1z{}.fits'.format(iter_no, b+1), field=1)
        y2_map = hp.read_map(save_dir +
                             '/flask/output/iter_{}/kappa-gamma-f1z{}.fits'.format(iter_no, b+1), field=2)

        k_map[unobserved_ids] = hp.UNSEEN
        y1_map[unobserved_ids] = hp.UNSEEN
        y2_map[unobserved_ids] = hp.UNSEEN

        nz_bin = nz_dat[:, b+1]

        poisson_noise_cls = np.zeros(raw_pcl_lmax_out+1)

        sky_coverage = hp.nside2pixarea(nside, degrees=True) * (npix - len(observed_ids)) * ((pi / 180) ** 2)
        poisson_noise_cls_allsky = poisson_noise_cls + (1 * sky_coverage / sum(nz_bin))
        poisson_noise_cls = poisson_noise_cls + (w_survey * sky_coverage / sum(nz_bin))

        poisson_noise_map = hp.synfast(cls=poisson_noise_cls_allsky, nside=nside, lmax=raw_pcl_lmax_out, pol=False)
        dens_map += poisson_noise_map

        shape_noise_cls = np.zeros(raw_pcl_lmax_out+1)
        she_nl = ((((sigma_e / np.sqrt(2)) ** 2) / (sum(nz_bin) / sky_coverage)) * w_survey)
        she_nl_all_sky = ((((sigma_e / np.sqrt(2)) ** 2) / (sum(nz_bin) / sky_coverage)))
        shape_noise_cls += she_nl

        dens_map[unobserved_ids] = hp.UNSEEN
        shape_noise_map = hp.synfast(cls=np.zeros(raw_pcl_lmax_out+1)+she_nl_all_sky, nside=nside, lmax=raw_pcl_lmax_out, pol=False)

        y1_map += shape_noise_map
        y2_map += shape_noise_map

        y1_map[unobserved_ids] = hp.UNSEEN
        y2_map[unobserved_ids] = hp.UNSEEN

        np.savetxt(poisson_cls_theory_dir + 'bin_{}_{}.txt'.format(b + 1, b + 1),
                   np.transpose(poisson_noise_cls))

        np.savetxt(poisson_cls_theory_dir + 'ell.txt', np.transpose(ell_arr))

        np.savetxt(shape_noise_cls_dir + 'bin_{}_{}.txt'.format(b + 1, b + 1),
                   np.transpose(shape_noise_cls))

        np.savetxt(shape_noise_cls_dir + 'ell.txt', np.transpose(ell_arr))

        cut_sky_map_dicts["BIN_{}".format(b + 1)] = []
        cut_sky_map_dicts["BIN_{}".format(b + 1)].append(
            nmt.NmtField(mask=mask, maps=[k_map], spin=0, lmax_sht=raw_pcl_lmax_out))
        cut_sky_map_dicts["BIN_{}".format(b + 1)].append(
            nmt.NmtField(mask=mask, maps=[y1_map, y2_map], spin=2, lmax_sht=raw_pcl_lmax_out))
        cut_sky_map_dicts["BIN_{}".format(b + 1)].append(
            nmt.NmtField(mask=mask, maps=[dens_map], spin=0, lmax_sht=raw_pcl_lmax_out))

    return cut_sky_map_dicts


def measure_00_pcls(lmin_out, lmax_out, cut_maps_dic, spectra_type, bin_i, bin_j, measured_pcl_save_dir):
    accepted_spectra_types = {'TT', 'gal_gal'}
    if spectra_type not in accepted_spectra_types:
        print('Warning! Field Type Not Recognised - Exiting...')
        sys.exit()

    nmt_fields = {
        'TT': [cut_maps_dic["BIN_{}".format(bin_i)][0],
               cut_maps_dic["BIN_{}".format(bin_j)][0],
               ],

        'gal_gal': [cut_maps_dic["BIN_{}".format(bin_i)][-1],
                    cut_maps_dic["BIN_{}".format(bin_j)][-1],
                    ],
    }

    pcl_coupled = nmt.compute_coupled_cell(nmt_fields[spectra_type][0], nmt_fields[spectra_type][1])

    measured_pcl = pcl_coupled[0][lmin_out:lmax_out + 1]

    np.savetxt(measured_pcl_save_dir + 'bin_{}_{}.txt'.format(bin_i, bin_j),
               np.transpose(measured_pcl))


def measure_02_pcls(lmin_out, lmax_out, cut_maps_dic, spectra_type, bin_i, bin_j, measured_pcl_save_dir):
    accepted_spectra_types = {'TE', 'TB', 'gal_E', 'gal_B'}
    if spectra_type not in accepted_spectra_types:
        print('Warning! Field Type Not Recognised - Exiting...')
        sys.exit()

    nmt_fields = {
        'TE': [cut_maps_dic["BIN_{}".format(bin_i)][0],
               cut_maps_dic["BIN_{}".format(bin_j)][1],
               ],

        'TB': [cut_maps_dic["BIN_{}".format(bin_i)][0],
               cut_maps_dic["BIN_{}".format(bin_j)][1],
               ],

        'gal_E': [cut_maps_dic["BIN_{}".format(bin_i)][-1],
                  cut_maps_dic["BIN_{}".format(bin_j)][1],
                  ],

        'gal_B': [cut_maps_dic["BIN_{}".format(bin_i)][-1],
                  cut_maps_dic["BIN_{}".format(bin_j)][1],
                  ]
    }

    pcl_coupled = nmt.compute_coupled_cell(nmt_fields[spectra_type][0], nmt_fields[spectra_type][1])
    measured_pcls = defaultdict(list)

    if spectra_type == 'TE' or 'TB':
        measured_pcls['TE'] = (pcl_coupled[0][lmin_out:lmax_out + 1])
        measured_pcls['TB'] = (pcl_coupled[1][lmin_out:lmax_out + 1])

    if spectra_type == 'gal_E' or 'gal_B':
        measured_pcls['gal_E'] = (pcl_coupled[0][lmin_out:lmax_out + 1])
        measured_pcls['gal_B'] = (pcl_coupled[1][lmin_out:lmax_out + 1])

    np.savetxt(measured_pcl_save_dir + 'bin_{}_{}.txt'.format(bin_i, bin_j),
               np.transpose(measured_pcls[spectra_type]))


def measure_22_pcls(lmin_out, lmax_out, cut_maps_dic, spectra_type, bin_i, bin_j, measured_pcl_save_dir):
    accepted_spectra_types = {'EE', 'EB', 'BE', 'BB'}
    if spectra_type not in accepted_spectra_types:
        # print(spectra_type)
        print('Warning! Field Type Not Recognised - Exiting...')
        sys.exit()

    pcl_coupled = nmt.compute_coupled_cell(cut_maps_dic["BIN_{}".format(bin_i)][1],
                                           cut_maps_dic["BIN_{}".format(bin_j)][1])

    measured_pcl_components = {
        'EE': (pcl_coupled[0][lmin_out:lmax_out + 1]),
        'EB': (pcl_coupled[1][lmin_out:lmax_out + 1]),
        'BE': (pcl_coupled[2][lmin_out:lmax_out + 1]),
        'BB': (pcl_coupled[3][lmin_out:lmax_out + 1])
    }

    np.savetxt(measured_pcl_save_dir + 'bin_{}_{}.txt'.format(bin_i, bin_j),
               np.transpose(measured_pcl_components[spectra_type]))


def execute_pcl_measurement(config_dict, iter_no, cut_sky_map_dicts):
    # In main do a for loop over realisations

    save_dir = config_dict['save_dir']
    nbins = config_dict['nbins']
    raw_pcl_lmin_out = config_dict['raw_pcl_lmin_out']
    raw_pcl_lmax_out = config_dict['raw_pcl_lmax_out']

    ell_arr = np.arange(raw_pcl_lmin_out, raw_pcl_lmax_out + 1, 1)

    realisation = iter_no

    recov_cat_cls_dir = save_dir + 'raw_3x2pt_cls/'

    gal_dir = recov_cat_cls_dir + 'galaxy_cl/iter_{}/'.format(realisation)
    gal_shear_dir = recov_cat_cls_dir + 'galaxy_shear_cl/iter_{}/'.format(realisation)

    shear_dir = recov_cat_cls_dir + 'shear_cl/'
    k_dir = shear_dir + 'Cl_TT/iter_{}/'.format(realisation)
    y1_dir = shear_dir + 'Cl_EE/iter_{}/'.format(realisation)
    y1y2_dir = shear_dir + 'Cl_EB/iter_{}/'.format(realisation)
    y2y1_dir = shear_dir + 'Cl_BE/iter_{}/'.format(realisation)
    y2_dir = shear_dir + 'Cl_BB/iter_{}/'.format(realisation)

    for path in [recov_cat_cls_dir, gal_dir, gal_shear_dir, shear_dir, k_dir, y1_dir, y1y2_dir, y2y1_dir, y2_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

        np.savetxt(path + 'ell.txt',
                   np.transpose(ell_arr))

    # Let's measure some Pseudo-Cls
    for bin_i in range(nbins):
        for bin_j in range(nbins):

            # Galaxy-Galaxy Lensing
            measure_02_pcls(
                lmin_out=raw_pcl_lmin_out,
                lmax_out=raw_pcl_lmax_out,
                cut_maps_dic=cut_sky_map_dicts,
                spectra_type='gal_E',
                bin_i=bin_i + 1,
                bin_j=bin_j + 1,
                measured_pcl_save_dir=gal_shear_dir
            )

            if bin_i >= bin_j:
                # Weak Lensing Convergence
                measure_00_pcls(
                    lmin_out=raw_pcl_lmin_out,
                    lmax_out=raw_pcl_lmax_out,
                    cut_maps_dic=cut_sky_map_dicts,
                    spectra_type='TT',
                    bin_i=bin_i + 1,
                    bin_j=bin_j + 1,
                    measured_pcl_save_dir=k_dir
                )

                # Galaxy Clustering
                measure_00_pcls(
                    lmin_out=raw_pcl_lmin_out,
                    lmax_out=raw_pcl_lmax_out,
                    cut_maps_dic=cut_sky_map_dicts,
                    spectra_type='gal_gal',
                    bin_i=bin_i + 1,
                    bin_j=bin_j + 1,
                    measured_pcl_save_dir=gal_dir)

                # Weak Lensing E-Mode
                measure_22_pcls(
                    lmin_out=raw_pcl_lmin_out,
                    lmax_out=raw_pcl_lmax_out,
                    cut_maps_dic=cut_sky_map_dicts,
                    spectra_type='EE',
                    bin_i=bin_i + 1,
                    bin_j=bin_j + 1,
                    measured_pcl_save_dir=y1_dir
                )

                # Weak Lensing EB Cross
                measure_22_pcls(
                    lmin_out=raw_pcl_lmin_out,
                    lmax_out=raw_pcl_lmax_out,
                    cut_maps_dic=cut_sky_map_dicts,
                    spectra_type='EB', bin_i=bin_i + 1,
                    bin_j=bin_j + 1,
                    measured_pcl_save_dir=y1y2_dir
                )

                # Weak Lensing BE Cross
                measure_22_pcls(
                    lmin_out=raw_pcl_lmin_out,
                    lmax_out=raw_pcl_lmax_out,
                    cut_maps_dic=cut_sky_map_dicts,
                    spectra_type='BE',
                    bin_i=bin_i + 1,
                    bin_j=bin_j + 1,
                    measured_pcl_save_dir=y2y1_dir
                )

                # Weak Lensing B-Mode
                measure_22_pcls(
                    lmin_out=raw_pcl_lmin_out,
                    lmax_out=raw_pcl_lmax_out,
                    cut_maps_dic=cut_sky_map_dicts,
                    spectra_type='BB',
                    bin_i=bin_i + 1,
                    bin_j=bin_j + 1,
                    measured_pcl_save_dir=y2_dir
                )

    # Fill in the remaining theory Cls - off diagonal spectra will be zero:

    gal_shear_noise_cls_dir = save_dir + 'raw_noise_cls/galaxy_shear_cl/iter_%s/' % realisation
    if not os.path.exists(gal_shear_noise_cls_dir):
        os.makedirs(gal_shear_noise_cls_dir)

    np.savetxt(save_dir + 'raw_noise_cls/galaxy_cl/ell.txt',
               np.transpose(ell_arr))

    np.savetxt(save_dir + 'raw_noise_cls/shear_cl/ell.txt',
               np.transpose(ell_arr))

    np.savetxt(save_dir + 'raw_noise_cls/galaxy_shear_cl/ell.txt',
               np.transpose(ell_arr))

    for i in range(nbins):
        for j in range(nbins):
            null_noise_cls = np.zeros(len(ell_arr))
            np.savetxt(
                gal_shear_noise_cls_dir + 'bin_%s_%s.txt' % (i + 1, j + 1),
                np.transpose(null_noise_cls)
            )

            np.savetxt(gal_shear_noise_cls_dir + 'ell.txt', np.transpose(ell_arr))

            if i > j:
                poisson_cls_theory_dir = save_dir + 'raw_noise_cls/galaxy_cl/iter_{}/'.format(realisation)
                np.savetxt(
                    poisson_cls_theory_dir + 'bin_%s_%s.txt' % (i + 1, j + 1),
                    np.transpose(null_noise_cls)
                )

                shape_noise_cls_dir = save_dir + 'raw_noise_cls/shear_cl/iter_{}/'.format(realisation)
                np.savetxt(
                    shape_noise_cls_dir + 'bin_%s_%s.txt' % (i + 1, j + 1),
                    np.transpose(null_noise_cls)
                )


def execute(pipeline_variables_path, realisation):
    # pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    # realisation = os.environ['ITER_NO']

    config_dict = measure_pcls_config(pipeline_variables_path=pipeline_variables_path)
    cut_maps = maps(config_dict=config_dict, iter_no=realisation)
    execute_pcl_measurement(config_dict=config_dict, iter_no=realisation, cut_sky_map_dicts=cut_maps)


# if __name__ == '__main__':
#     main()