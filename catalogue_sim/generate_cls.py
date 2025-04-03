import os
import configparser
import numpy as np
import pyccl as ccl


def execute(pipeline_variables_path):

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    nz_filename = str(config['redshift_distribution']['NZ_TABLE_NAME'])
    n_zbin = int(float(config['redshift_distribution']['N_ZBIN']))
    ell_min = int(float(config['simulation_setup']['INPUT_ELL_MIN']))
    ell_max = int(float(config['simulation_setup']['INPUT_ELL_MAX']))

    # Read in cosmology parameters
    Omega_c = float(config['cosmology']['Omega_c'])
    Omega_b = float(config['cosmology']['Omega_b'])
    h = float(config['cosmology']['h'])
    A_s = float(config['cosmology']['A_s'])
    # sigma8 = float(config['cosmology']['sigma8'])
    n_s = float(config['cosmology']['n_s'])
    Omega_k = float(config['cosmology']['Omega_k'])
    w0 = float(config['cosmology']['w0'])
    wa = float(config['cosmology']['wa'])

    # Fiducial cosmology. These parameters could be read in from the config file
    fid_cosmo = ccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        h=h,
        A_s=A_s,
        n_s=n_s,
        Omega_k=Omega_k,
        w0=w0,
        wa=wa
    )

    # Read in some n(z) table data
    nz_dat = np.loadtxt(f"{save_dir}{nz_filename}")

    ells = np.arange(ell_min, ell_max + 1, 1)

    cl_dirs = ['galaxy_cl/', 'shear_cl/', 'galaxy_shear_cl/', 'cmbkappa_cl/', 'galaxy_cmbkappa_cl/', 'shear_cmbkappa_cl/']
    for cl_dir in cl_dirs:
        cl_save_dir = f"{save_dir}fiducial_cosmology/{cl_dir}"
        if not os.path.exists(cl_save_dir):
            os.makedirs(cl_save_dir)
        np.savetxt(f"{cl_save_dir}ell.txt", ells)
    assert n_zbin == (nz_dat.shape[1]) - 1
    z = nz_dat[:,0]

    # Galaxy bias
    b = np.ones_like(z)
    # bz = 0.95/ccl.growth_factor(fid_cosmo,1./(1+z))

    # Intrinsic alignment amplitude
    A_IA = 0.6*np.ones_like(z)

    # Magnification bias
    sz = np.ones_like(z)

    # CMB lensing
    k_CMB = ccl.CMBLensingTracer(fid_cosmo, z_source=1100)

    cl_kCMB = ccl.angular_cl(fid_cosmo, k_CMB, k_CMB, ells)
    np.savetxt(save_dir + 'fiducial_cosmology/cmbkappa_cl/bin_1_1.txt', cl_kCMB)

    for i in range(n_zbin):

        # Bin i number
        bin_i = i + 1

        # Galaxy clustering bin i
        g_i = ccl.NumberCountsTracer(fid_cosmo, has_rsd=False, dndz=(z, nz_dat[:, bin_i]), bias=(z, b))

        # Cosmic shear with intrinsic alignments bin j
        y_i = ccl.WeakLensingTracer(fid_cosmo, dndz=(z, nz_dat[:, bin_i]), has_shear=True, ia_bias=(z, A_IA))

        # Galaxy clustering - CMB kappa Cl cross-correlation
        cl_g_kCMB = ccl.angular_cl(fid_cosmo, g_i, k_CMB, ells)
        np.savetxt(save_dir + 'fiducial_cosmology/galaxy_cmbkappa_cl/bin_{}_1.txt'.format(bin_i), cl_g_kCMB)

        # Cosmic shear - CMB kappa Cl cross-correlation
        cl_y_kCMB = ccl.angular_cl(fid_cosmo, y_i, k_CMB, ells)
        np.savetxt(save_dir + 'fiducial_cosmology/shear_cmbkappa_cl/bin_{}_1.txt'.format(bin_i), cl_y_kCMB)

        for j in range(n_zbin):

            # Bin j number
            bin_j = j+1

            # Galaxy clustering bin j
            g_j = ccl.NumberCountsTracer(fid_cosmo, has_rsd=False, dndz=(z, nz_dat[:,bin_j]), bias=(z, b))

            # Cosmic shear with intrinsic alignments bin j
            y_j = ccl.WeakLensingTracer(fid_cosmo, dndz=(z, nz_dat[:,bin_j]), has_shear=True, ia_bias=(z, A_IA))

            # Tomographic angular clustering Cl
            cl_gg = ccl.angular_cl(fid_cosmo, g_i, g_j, ells)
            np.savetxt(save_dir + 'fiducial_cosmology/galaxy_cl/bin_{}_{}.txt'.format(bin_i, bin_j), cl_gg)

            # Tomographic galaxy-galaxy lensing Cl
            cl_gy = ccl.angular_cl(fid_cosmo, g_i, y_j, ells)
            np.savetxt(save_dir + 'fiducial_cosmology/galaxy_shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j), cl_gy)

            # Tomographic cosmic shear Cl
            cl_yy = ccl.angular_cl(fid_cosmo, y_i, y_j, ells)
            np.savetxt(save_dir + 'fiducial_cosmology/shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j), cl_yy)


