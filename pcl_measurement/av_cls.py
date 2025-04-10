import configparser
import numpy as np
from scipy.stats import sem


def av_cls_config(pipeline_variables_path):
    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    nside = int(config['simulation_setup']['NSIDE'])
    realisations = int(config['simulation_setup']['realisations'])
    pcl_lmin_out = 0
    pcl_lmax_out = int(float(config['simulation_setup']['INPUT_ELL_MAX']))

    nbins = int(config['redshift_distribution']['N_ZBIN'])

    # Prepare config dictionary
    config_dict = {
        'nside': nside,
        'nbins': nbins,
        'pcl_lmin_out': pcl_lmin_out,
        'pcl_lmax_out': pcl_lmax_out,
        'save_dir': save_dir,
        'realisations': realisations,
    }

    return config_dict


def calc_av_cls(cl_dir, ell_min, ell_max, bin_i, bin_j, realisations, err=True):
    cls = []
    ell = np.arange(ell_min, ell_max + 1)

    for x in range(realisations):
        cl_file = cl_dir + 'iter_{}/bin_{}_{}.txt'.format(x + 1, bin_i, bin_j)
        cls.append(np.loadtxt(cl_file))

    cls = np.asarray(cls)

    cls_av = np.mean(cls, axis=0)

    np.savetxt(cl_dir + 'bin_{}_{}.txt'.format(bin_i, bin_j),
               np.transpose(cls_av))

    if err:
        cls_err = sem(cls, axis=0)

        np.savetxt(cl_dir + 'bin_{}_{}_err.txt'.format(bin_i, bin_j),
                   np.transpose(cls_err))

    np.savetxt(cl_dir + 'ell.txt', np.transpose(ell))


def execute(pipeline_variables_path):
    # pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']
    config_dict = av_cls_config(pipeline_variables_path=pipeline_variables_path)

    save_dir = config_dict['save_dir']
    nbins = config_dict['nbins']

    pcl_lmin_out = config_dict['pcl_lmin_out']
    pcl_lmax_out = config_dict['pcl_lmax_out']

    realisations = config_dict['realisations']

    noise_cls_dir = save_dir + 'raw_noise_cls/'
    measured_cls_dir = save_dir + 'raw_6x2pt_cls/'

    # kcmb auto power
    calc_av_cls(cl_dir=measured_cls_dir + 'cmbkappa_cl/',
                ell_min=pcl_lmin_out,
                ell_max=pcl_lmax_out,
                bin_i=1,
                bin_j=1,
                realisations=realisations)

    calc_av_cls(cl_dir=noise_cls_dir + 'cmbkappa_cl/',
                ell_min=pcl_lmin_out,
                ell_max=pcl_lmax_out,
                bin_i=1,
                bin_j=1,
                realisations=realisations,
                err=False)

    for i in range(nbins):

        # kcmb_d
        calc_av_cls(cl_dir=measured_cls_dir + 'galaxy_cmbkappa_cl/',
                    ell_min=pcl_lmin_out,
                    ell_max=pcl_lmax_out,
                    bin_i=i + 1,
                    bin_j=1,
                    realisations=realisations)

        calc_av_cls(cl_dir=noise_cls_dir + 'galaxy_cmbkappa_cl/',
                    ell_min=pcl_lmin_out,
                    ell_max=pcl_lmax_out,
                    bin_i=i + 1,
                    bin_j=1,
                    realisations=realisations,
                    err=False)

        # kcmb_y (E mode)
        calc_av_cls(cl_dir=measured_cls_dir + 'shear_cmbkappa_cl/kCMB_E/',
                    ell_min=pcl_lmin_out,
                    ell_max=pcl_lmax_out,
                    bin_i=i + 1,
                    bin_j=1,
                    realisations=realisations)

        # kcmb_y (B mode)
        calc_av_cls(cl_dir=measured_cls_dir + 'shear_cmbkappa_cl/kCMB_B/',
                    ell_min=pcl_lmin_out,
                    ell_max=pcl_lmax_out,
                    bin_i=i + 1,
                    bin_j=1,
                    realisations=realisations)

        calc_av_cls(cl_dir=noise_cls_dir + 'shear_cmbkappa_cl/',
                    ell_min=pcl_lmin_out,
                    ell_max=pcl_lmax_out,
                    bin_i=i + 1,
                    bin_j=1,
                    realisations=realisations,
                    err=False)

        for j in range(nbins):

            calc_av_cls(cl_dir=noise_cls_dir + 'galaxy_shear_cl/',
                        ell_min=pcl_lmin_out,
                        ell_max=pcl_lmax_out,
                        bin_i=i + 1,
                        bin_j=j + 1,
                        realisations=realisations,
                        err=False)

            calc_av_cls(cl_dir=measured_cls_dir + 'galaxy_shear_cl/',
                        ell_min=pcl_lmin_out,
                        ell_max=pcl_lmax_out,
                        bin_i=i + 1,
                        bin_j=j + 1,
                        realisations=realisations)

            if i >= j:
                calc_av_cls(noise_cls_dir + 'galaxy_cl/',
                            ell_min=pcl_lmin_out,
                            ell_max=pcl_lmax_out,
                            bin_i=i + 1,
                            bin_j=j + 1,
                            realisations=realisations,
                            err=False)

                calc_av_cls(measured_cls_dir + 'galaxy_cl/',
                            ell_min=pcl_lmin_out,
                            ell_max=pcl_lmax_out,
                            bin_i=i + 1,
                            bin_j=j + 1,
                            realisations=realisations)

                calc_av_cls(noise_cls_dir + 'shear_cl/',
                            ell_min=pcl_lmin_out,
                            ell_max=pcl_lmax_out,
                            bin_i=i + 1,
                            bin_j=j + 1,
                            realisations=realisations,
                            err=False)

    cl_shear_types = ['Cl_TT', 'Cl_EE', 'Cl_EB', 'Cl_BE', 'Cl_BB']

    for shear_type in cl_shear_types:
        for i in range(nbins):
            for j in range(nbins):
                if i >= j:
                    calc_av_cls(measured_cls_dir + 'shear_cl/' + shear_type + '/',
                                ell_min=pcl_lmin_out,
                                ell_max=pcl_lmax_out,
                                bin_i=i + 1,
                                bin_j=j + 1,
                                realisations=realisations)


# if __name__ == '__main__':
#     main()
