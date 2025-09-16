import os
import configparser
import sys

import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
import pyccl.ccllib as ccllib


def setup_config_dict(pipeline_variables_path):

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    nz_filename = str(config['redshift_distribution']['NZ_TABLE_NAME'])
    n_zbin = int(float(config['redshift_distribution']['N_ZBIN']))
    ell_min = int(float(config['simulation_setup']['INPUT_ELL_MIN']))
    ell_max = int(float(config['simulation_setup']['INPUT_ELL_MAX']))

    setup_dict = {
        'save_dir': save_dir,
        'nz_filename': nz_filename,
        'n_zbin': n_zbin,
        'ell_min': ell_min,
        'ell_max': ell_max
    }

    return setup_dict


def CCL_cosmo(pipeline_variables_path):
    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    # Read in cosmology parameters
    Omega_c = float(config['cosmology']['Omega_c'])
    Omega_b = float(config['cosmology']['Omega_b'])
    h = float(config['cosmology']['h'])
    n_s = float(config['cosmology']['n_s'])
    Omega_k = float(config['cosmology']['Omega_k'])
    w0 = float(config['cosmology']['w0'])
    wa = float(config['cosmology']['wa'])
    try:
        A_s = float(config['cosmology']['A_s'])
        # Fiducial cosmology. These parameters could be read in from the config file
        cosmo = ccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            A_s=A_s,
            n_s=n_s,
            Omega_k=Omega_k,
            w0=w0,
            wa=wa,
            extra_parameters={'camb': {'dark_energy_model': 'ppf'}}
        )

    except KeyError:
        sigma8 = float(config['cosmology']['sigma8'])
        # Fiducial cosmology. These parameters could be read in from the config file
        cosmo = ccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            sigma8=sigma8,
            n_s=n_s,
            Omega_k=Omega_k,
            w0=w0,
            wa=wa,
            extra_parameters={'camb': {'dark_energy_model': 'ppf'}}
        )

    return cosmo


def setup_systematics_dict(pipeline_variables_path):

    # bi_marg and mi_marg must be lists of values per tomographic bin, e.g [b_bin1, b_bin2, b_bin3],
    # [m_bin1, m_bin2, m_bin3]. Or can be [b] or [m] if setting a global linear galaxy bias or m-bias

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    nz_filename = str(config['redshift_distribution']['NZ_TABLE_NAME'])
    n_zbin = int(float(config['redshift_distribution']['N_ZBIN']))
    # ell_min = int(float(config['simulation_setup']['INPUT_ELL_MIN']))
    # ell_max = int(float(config['simulation_setup']['INPUT_ELL_MAX']))

    nz_dat = np.loadtxt(f"{save_dir}{nz_filename}")
    assert n_zbin == (nz_dat.shape[1]) - 1
    z = nz_dat[:,0]

    ## Set up a nonlinear galaxy bias model.

    nz_boundaries = np.loadtxt(f"{save_dir}z_boundaries.txt")

    # Galaxy bias parameters
    # bi = str(config['galaxy_bias']['bi'])

    b1_config = config['galaxy_bias']['b1']
    if ',' in b1_config:
        # assert bi == 'True'
        b1_dat = [float(i) for i in b1_config.split(',')]
        assert len(b1_dat) == n_zbin, "Insufficient galaxy bias parameters to match number of tomographic redshift bins..."
        b_1 = np.ones(len(z))
        for i in range(n_zbin-1):
            ids = np.where((z.round(decimals=2) >= nz_boundaries[:,0][i].round(decimals=2)) & (z.round(decimals=2) < nz_boundaries[:,2][i].round(decimals=2)))[0]
            b_1[ids] = b1_dat[i]
        ids_last = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][-2].round(decimals=2)) & (
                    z.round(decimals=2) <= nz_boundaries[:, 2][-2].round(decimals=2)))[0]
        b_1[ids_last] = b1_dat[-1]

    else:
        b1_dat = float(b1_config)
        b_1 = b1_dat * np.ones(len(z))

    b2_config = config['galaxy_bias']['b2']
    if ',' in b2_config:
        b2_dat = [float(i) for i in b2_config.split(',')]
        assert len(b2_dat) == n_zbin, "Insufficient galaxy bias parameters to match number of tomographic redshift bins..."
        b_2 = np.ones(len(z))
        for i in range(n_zbin-1):
            ids = np.where((z.round(decimals=2) >= nz_boundaries[:,0][i].round(decimals=2)) & (z.round(decimals=2) < nz_boundaries[:,2][i].round(decimals=2)))[0]
            b_2[ids] = b2_dat[i]
        ids_last = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][-2].round(decimals=2)) & (
                    z.round(decimals=2) <= nz_boundaries[:, 2][-2].round(decimals=2)))[0]
        b_2[ids_last] = b2_dat[-1]

    else:
        b2_dat = float(b2_config)
        b_2 = b2_dat * np.ones(len(z))

    bs_config = config['galaxy_bias']['bs']
    if ',' in bs_config:
        bs_dat = [float(i) for i in bs_config.split(',')]
        assert len(bs_dat) == n_zbin, "Insufficient galaxy bias parameters to match number of tomographic redshift bins..."
        b_s = np.ones(len(z))
        for i in range(n_zbin-1):
            ids = np.where((z.round(decimals=2) >= nz_boundaries[:,0][i].round(decimals=2)) & (z.round(decimals=2) < nz_boundaries[:,2][i].round(decimals=2)))[0]
            b_s[ids] = bs_dat[i]
        ids_last = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][-2].round(decimals=2)) & (
                    z.round(decimals=2) <= nz_boundaries[:, 2][-2].round(decimals=2)))[0]
        b_s[ids_last] = bs_dat[-1]
    else:
        bs_dat = float(bs_config)
        b_s = bs_dat * np.ones(len(z))

    ## Set up some code to implement TATT model for IAs.

    # Intrinsic alignment parameters
    A1_config = config['intrinsic_alignment']['A1']
    if ',' in A1_config:
        A1_dat = [float(i) for i in A1_config.split(',')]
        assert len(A1_dat) == n_zbin, "Insufficient IA parameters to match number of tomographic redshift bins..."
        A_1 = np.ones(len(z))
        for i in range(n_zbin-1):
            ids = np.where((z.round(decimals=2) >= nz_boundaries[:,0][i].round(decimals=2)) & (z.round(decimals=2) < nz_boundaries[:,2][i].round(decimals=2)))[0]
            A_1[ids] = A1_dat[i]
        ids_last = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][-2].round(decimals=2)) & (
                    z.round(decimals=2) <= nz_boundaries[:, 2][-2].round(decimals=2)))[0]
        A_1[ids_last] = A1_dat[-1]
    else:
        A1_dat = float(A1_config)
        A_1 = A1_dat * np.ones(len(z))

    A2_config = config['intrinsic_alignment']['A2']
    if ',' in A2_config:
        A2_dat = [float(i) for i in A2_config.split(',')]
        assert len(A2_dat) == n_zbin, "Insufficient IA parameters to match number of tomographic redshift bins..."
        A_2 = np.ones(len(z))
        for i in range(n_zbin-1):
            ids = np.where((z.round(decimals=2) >= nz_boundaries[:,0][i].round(decimals=2)) & (z.round(decimals=2) < nz_boundaries[:,2][i].round(decimals=2)))[0]
            A_2[ids] = A2_dat[i]
        ids_last = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][-2].round(decimals=2)) & (
                    z.round(decimals=2) <= nz_boundaries[:, 2][-2].round(decimals=2)))[0]
        A_2[ids_last] = A2_dat[-1]
    else:
        A2_dat = float(A2_config)
        A_2 = A2_dat * np.ones(len(z))

    bTA_config = config['intrinsic_alignment']['bTA']
    if ',' in bTA_config:
        bTA_dat = [float(i) for i in bTA_config.split(',')]
        assert len(bTA_dat) == n_zbin, "Insufficient IA parameters to match number of tomographic redshift bins..."
        bTA = np.ones(len(z))
        for i in range(n_zbin-1):
            ids = np.where((z.round(decimals=2) >= nz_boundaries[:,0][i].round(decimals=2)) & (z.round(decimals=2) < nz_boundaries[:,2][i].round(decimals=2)))[0]
            bTA[ids] = bTA_dat[i]
        ids_last = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][-2].round(decimals=2)) & (
                    z.round(decimals=2) <= nz_boundaries[:, 2][-2].round(decimals=2)))[0]
        bTA[ids_last] = bTA_dat[-1]
    else:
        bTA_dat = float(bTA_config)
        bTA = bTA_dat * np.ones(len(z))

    eta1 = float(config['intrinsic_alignment']['eta1'])
    eta2 = float(config['intrinsic_alignment']['eta2'])

    z0 = float(config['redshift_distribution']['Z0'])

    # Growth factor
    # gz = ccl.growth_factor(cosmo, 1/(1+z))

    # Shear m bias

    mi_config = config['shear_m_bias']['mi']
    if ',' in mi_config:
        mi_dat = [float(i) for i in mi_config.split(',')]
        assert len(mi_dat) == n_zbin, "Insufficient shear m-bias parameters to match number of tomographic redshift bins..."

    else:
        mi_dat = [float(mi_config)] * n_zbin

    # Magnification bias
    # sz = np.ones_like(z)

    s0 = float(config['magnification_bias']['s0'])
    s1 = float(config['magnification_bias']['s1'])
    s2 = float(config['magnification_bias']['s2'])
    s3 = float(config['magnification_bias']['s3'])

    # print(sz)
    # sz = np.ones_like(z)

    # Redshift shift
    # Dzi = np.array([])

    systematics = {
        'b1': b_1,
        'b2': b_2,
        'bs': b_s,
        'A1': A_1,
        'A2': A_2,
        'bTA': bTA,
        'eta1': eta1,
        'eta2': eta2,
        'z0': z0,
        # 'c_1': c_1,
        # 'c_delta': c_delta,
        # 'c_2': c_2,
        'mi': np.asarray(mi_dat),
        # 'Dzi': Dzi,
        's0': s0,
        's1': s1,
        's2': s2,
        's3': s3,
        # 'sz': sz
    }

    return systematics


def setup_6x2pt_cls(save_dir, nz_filename, n_zbin, ell_min, ell_max, fid_cosmo, systematics_dict, mode='dict', Dzi_dat=None):

    if mode not in {'save', 'dict'}:
        print("Incorrect mode - must be 'save' or 'dict'")
        sys.exit()

    # Read in cosmology parameters
    # Omega_c = float(config['cosmology']['Omega_c'])
    # Omega_b = float(config['cosmology']['Omega_b'])
    # h = float(config['cosmology']['h'])
    # A_s = float(config['cosmology']['A_s'])
    # # sigma8 = float(config['cosmology']['sigma8'])
    # n_s = float(config['cosmology']['n_s'])
    # Omega_k = float(config['cosmology']['Omega_k'])
    # w0 = float(config['cosmology']['w0'])
    # wa = float(config['cosmology']['wa'])

    # # Fiducial cosmology. These parameters could be read in from the config file
    # fid_cosmo = ccl.Cosmology(
    #     Omega_c=Omega_c,
    #     Omega_b=Omega_b,
    #     h=h,
    #     A_s=A_s,
    #     n_s=n_s,
    #     Omega_k=Omega_k,
    #     w0=w0,
    #     wa=wa,
    #     extra_parameters={'camb': {'dark_energy_model': 'ppf'}}
    #
    # )

    # Read in some n(z) table data
    nz_dat = np.loadtxt(f"{save_dir}{nz_filename}")
    # nz_dat = np.loadtxt(f"{save_dir}nz_model.txt")
    assert n_zbin == (nz_dat.shape[1]) - 1
    z = nz_dat[:,0]

    if Dzi_dat is not None:
        assert len(Dzi_dat) == n_zbin

    ells = np.arange(ell_min, ell_max + 1, 1)

    cl_dirs = ['galaxy_cl/', 'shear_cl/', 'galaxy_shear_cl/', 'cmbkappa_cl/', 'galaxy_cmbkappa_cl/', 'shear_cmbkappa_cl/']

    if mode == 'save':
        cls_dict = None
        for cl_dir in cl_dirs:
            cl_save_dir = f"{save_dir}fiducial_cosmology/{cl_dir}"
            if not os.path.exists(cl_save_dir):
                os.makedirs(cl_save_dir)
            np.savetxt(f"{cl_save_dir}ell.txt", ells)

    else:
        assert mode == 'dict'
        # Now we have to make the Cls, PCls, bandpowers, then combined data vector
        cls_dict = {'galaxy_cl': {'ell':ells},
                    'shear_cl': {'ell':ells},
                    'galaxy_shear_cl': {'ell':ells},
                    'cmbkappa_cl': {'ell':ells},
                    'galaxy_cmbkappa_cl': {'ell':ells},
                    'shear_cmbkappa_cl': {'ell':ells},
                    'null_spectra': {'ell':ells}
                    }


    '''
    # Simple galaxy bias model
    b = np.ones_like(z)
    # bz = 0.95/ccl.growth_factor(fid_cosmo,1./(1+z))

    # Intrinsic alignment amplitude (for 'simple' NLA model)
    A_IA = 1*np.ones_like(z)
    
    # Amplitude params
    # a_1 = 2.5
    # a_2 = 0         # set to zero for standard NLA-z
    # a_1delta = 0    # set to zero for standard NLA-z

    b_1 = 1.0   # could be a function of z
    b_2 = 0   # Set to zero for standard first order bias parameterisation. Could also be a function of z
    b_s = 0   # Set to zero for standard first order bias parameterisation. Could be a function of z
    '''

    b_1 = systematics_dict['b1']
    b_2 = systematics_dict['b2']
    b_s = systematics_dict['bs']
    A_1 = systematics_dict['A1']
    A_2 = systematics_dict['A2']
    bTA = systematics_dict['bTA']
    eta1 = systematics_dict['eta1']
    eta2 = systematics_dict['eta2']
    z0 = systematics_dict['z0']
    # c_1 = systematics_dict['c_1']
    # c_delta = systematics_dict['c_delta']
    # c_2 = systematics_dict['c_2']
    mi_dat = systematics_dict['mi']
    s0 = systematics_dict['s0']
    s1 = systematics_dict['s1']
    s2 = systematics_dict['s2']
    s3 = systematics_dict['s3']
    # sz = systematics_dict['sz']

    # Convert IA params
    # c_1, c_delta, c_2 = pt.translate_IA_norm(fid_cosmo, z=z, a1=A_1, a1delta=A_1d, a2=A_2, Om_m2_for_c2=False)

    # print('Cosmology setup')
    # test_start_time = time.time()

    Om_m = fid_cosmo['Omega_m']
    rho_crit = ccllib.cvar.constants.RHO_CRITICAL
    gz = ccl.growth_factor(fid_cosmo, 1/(1+z))
    Cbar = 5e-14

    c_1 = -1 * A_1 * Cbar * rho_crit * (Om_m / gz) * (((1+z)/(1+z0))**eta1)
    c_2 = 5 * A_2 * Cbar * rho_crit * (Om_m / (gz**2)) * (((1+z)/(1+z0))**eta2)
    c_delta = bTA * c_1

    # Magnification_bias
    sz = s0 + (s1*z) + (s2*(z**2)) + (s3*(z**3))

    ptt_g = pt.PTNumberCountsTracer(b1=(z, b_1), b2=(z, b_2), bs=(z, b_s))

    ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(z, c_1), c2=(z, c_2), cdelta=(z, c_delta))
    ptt_m = pt.PTMatterTracer()

    ptc = pt.PTCalculator(with_NC=True, with_IA=True)   # leave other specific argument values to their default
    # ptc.update_ingredients(fid_cosmo)

    # Calculate PT power spectra. Please see CCLX example notebook -
    # https://github.com/LSSTDESC/CCLX/blob/master/PerturbationTheoryPk.ipynb

    # Galaxies x galaxies.
    pk_gg = pt.get_pt_pk2d(fid_cosmo, ptt_g, ptc=ptc)

    # Galaxies x matter
    pk_gm = pt.get_pt_pk2d(fid_cosmo, ptt_g, tracer2=ptt_m, ptc=ptc)

    # Galaxies x IAs
    pk_gi = pt.get_pt_pk2d(fid_cosmo, ptt_g, tracer2=ptt_i, ptc=ptc)

    # IAs x IAs
    pk_ii = pt.get_pt_pk2d(fid_cosmo, ptt_i, tracer2=ptt_i, ptc=ptc)
    # pk_ii_bb = pt.get_pt_pk2d(fid_cosmo, ptt_i, tracer2=ptt_i, return_ia_bb=True, ptc=ptc)

    # IAs x matter
    pk_im = pt.get_pt_pk2d(fid_cosmo, ptt_i, tracer2=ptt_m, ptc=ptc)

    # Matter x matter
    pk_mm = pt.get_pt_pk2d(fid_cosmo, ptt_m, tracer2=ptt_m, ptc=ptc)

    # CMB lensing
    k_CMB = ccl.CMBLensingTracer(fid_cosmo, z_source=1100)
    cl_kCMB = ccl.angular_cl(fid_cosmo, k_CMB, k_CMB, ells, p_of_k_a=pk_mm)

    if mode == 'save':
        np.savetxt(save_dir + 'fiducial_cosmology/cmbkappa_cl/bin_1_1.txt', cl_kCMB)

    else:
        assert mode == 'dict'
        cls_dict['cmbkappa_cl']['bin_1_1'] = cl_kCMB

    # CMB lensing
    # k_CMB = ccl.CMBLensingTracer(fid_cosmo, z_source=1100)

    # cl_kCMB = ccl.angular_cl(fid_cosmo, k_CMB, k_CMB, ells)

    # print(time.time()-test_start_time)

    for i in range(n_zbin):

        # Bin i number
        bin_i = i + 1

        if Dzi_dat is None:
            Dzi = 0
        else:
            Dzi = Dzi_dat[i]

        new_z_i = z+Dzi
        keep_ids_i = np.where((new_z_i>=0))[0]

        g_i = ccl.NumberCountsTracer(fid_cosmo, has_rsd=False, dndz=(new_z_i[keep_ids_i], nz_dat[:, bin_i][keep_ids_i]), bias=(z, np.ones_like(z)), mag_bias=(z,sz)) # set a simple linear bias of 1 since we deal with the bias at P(k) level
        y_i = ccl.WeakLensingTracer(fid_cosmo, dndz=(new_z_i[keep_ids_i], nz_dat[:, bin_i][keep_ids_i])) # don't add in IA here? Will have to sort out manually?

        y_i_ia = ccl.WeakLensingTracer(fid_cosmo, dndz=(new_z_i[keep_ids_i], nz_dat[:, bin_i][keep_ids_i]), has_shear=False, ia_bias=(z, np.ones_like(z)), use_A_ia=False) # set a simple IA bias of 1 since we deal with the bias at P(k) level

        cl_g_kCMB = ccl.angular_cl(fid_cosmo, g_i, k_CMB, ells, p_of_k_a=pk_gm)
        if mode == 'save':
            np.savetxt(save_dir + 'fiducial_cosmology/galaxy_cmbkappa_cl/bin_{}_1.txt'.format(bin_i), cl_g_kCMB)
        else:
            cls_dict['galaxy_cmbkappa_cl']['bin_{}_1'.format(bin_i)] = cl_g_kCMB

        cl_y_kCMB = ccl.angular_cl(fid_cosmo, y_i, k_CMB, ells, p_of_k_a=pk_mm)
        cl_I_kCMB = ccl.angular_cl(fid_cosmo, y_i_ia, k_CMB, ells, p_of_k_a=pk_im)

        if mode == 'save':
            np.savetxt(save_dir + 'fiducial_cosmology/shear_cmbkappa_cl/bin_{}_1.txt'.format(bin_i), (1+mi_dat[i])*(cl_y_kCMB+cl_I_kCMB))
        else:
            cls_dict['shear_cmbkappa_cl']['bin_{}_1'.format(bin_i)] = (1+mi_dat[i])*(cl_y_kCMB+cl_I_kCMB)

        # # Galaxy clustering bin i
        # g_i = ccl.NumberCountsTracer(fid_cosmo, has_rsd=False, dndz=(z, nz_dat[:, bin_i]), bias=(z, b))

        # # Cosmic shear with intrinsic alignments bin j
        # # y_i = ccl.WeakLensingTracer(fid_cosmo, dndz=(z, nz_dat[:, bin_i]), has_shear=True, ia_bias=(z, A_IA))
        # y_i = ccl.WeakLensingTracer(fid_cosmo, dndz=(z, nz_dat[:, bin_i]), has_shear=True, ia_bias=None)

        # # Galaxy clustering - CMB kappa Cl cross-correlation
        # cl_g_kCMB = ccl.angular_cl(fid_cosmo, g_i, k_CMB, ells)
        # np.savetxt(save_dir + 'fiducial_cosmology/galaxy_cmbkappa_cl/bin_{}_1.txt'.format(bin_i), cl_g_kCMB)

        # # Cosmic shear - CMB kappa Cl cross-correlation
        # cl_y_kCMB = ccl.angular_cl(fid_cosmo, y_i, k_CMB, ells)
        # np.savetxt(save_dir + 'fiducial_cosmology/shear_cmbkappa_cl/bin_{}_1.txt'.format(bin_i), cl_y_kCMB)

        for j in range(n_zbin):

            # Bin j number
            bin_j = j+1

            if Dzi_dat is None:
                Dzj = 0
            else:
                Dzj = Dzi_dat[j]

            new_z_j = z + Dzj
            keep_ids_j = np.where((new_z_j >= 0))[0]

            g_j = ccl.NumberCountsTracer(fid_cosmo, has_rsd=False, dndz=(new_z_j[keep_ids_j], nz_dat[:, bin_j][keep_ids_j]),
                                         bias=(z, np.ones_like(z)),
                                         mag_bias=(z,sz))  # set a simple linear bias of 1 since we deal with the bias at P(k) level

            y_j = ccl.WeakLensingTracer(fid_cosmo, dndz=(new_z_j[keep_ids_j], nz_dat[:, bin_j][keep_ids_j]))  # don't add in IA here? Will have to sort out manually?

            y_j_ia = ccl.WeakLensingTracer(fid_cosmo, dndz=(new_z_j[keep_ids_j], nz_dat[:, bin_j][keep_ids_j]), has_shear=False,
                                           ia_bias=(z, np.ones_like(z)), use_A_ia=False)  # set a simple IA bias of 1 since we deal with the bias at P(k) level

            cl_gy = ccl.angular_cl(fid_cosmo, g_i, y_j, ells, p_of_k_a=pk_gm)
            cl_gy_ia = ccl.angular_cl(fid_cosmo, g_i, y_j_ia, ells, p_of_k_a=pk_gi)

            if mode == 'save':
                np.savetxt(save_dir + 'fiducial_cosmology/galaxy_shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j), (1+mi_dat[j])*(cl_gy+cl_gy_ia))
            else:
                cls_dict['galaxy_shear_cl']['bin_{}_{}'.format(bin_i, bin_j)] = (1+mi_dat[j])*(cl_gy+cl_gy_ia)


            # # Galaxy clustering bin j
            # g_j = ccl.NumberCountsTracer(fid_cosmo, has_rsd=False, dndz=(z, nz_dat[:,bin_j]), bias=(z, b))

            # # Cosmic shear with intrinsic alignments bin j
            # # y_j = ccl.WeakLensingTracer(fid_cosmo, dndz=(z, nz_dat[:,bin_j]), has_shear=True, ia_bias=(z, A_IA))
            # y_j = ccl.WeakLensingTracer(fid_cosmo, dndz=(z, nz_dat[:,bin_j]), has_shear=True, ia_bias=None)

            # # Tomographic galaxy-galaxy lensing Cl
            # cl_gy = ccl.angular_cl(fid_cosmo, g_i, y_j, ells)
            # np.savetxt(save_dir + 'fiducial_cosmology/galaxy_shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j), cl_gy)

            if i>=j:

                cl_gg = ccl.angular_cl(fid_cosmo, g_i, g_j, ells, p_of_k_a=pk_gg)
                if mode == 'save':
                    np.savetxt(save_dir + 'fiducial_cosmology/galaxy_cl/bin_{}_{}.txt'.format(bin_i, bin_j), cl_gg)
                else:
                    cls_dict['galaxy_cl']['bin_{}_{}'.format(bin_i, bin_j)] = cl_gg

                cl_yy = ccl.angular_cl(fid_cosmo, y_i, y_j, ells, p_of_k_a=pk_mm)
                cl_yi = ccl.angular_cl(fid_cosmo, y_i, y_j_ia, ells, p_of_k_a=pk_im)
                cl_iy = ccl.angular_cl(fid_cosmo, y_i_ia, y_j, ells, p_of_k_a=pk_im)
                cl_ii = ccl.angular_cl(fid_cosmo, y_i_ia, y_j_ia, ells, p_of_k_a=pk_ii)

                if mode == 'save':
                    np.savetxt(save_dir + 'fiducial_cosmology/shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j),
                               (1+mi_dat[i])*(1+mi_dat[j])*(cl_yy+cl_yi+cl_iy+cl_ii))

                else:
                    cls_dict['shear_cl']['bin_{}_{}'.format(bin_i, bin_j)] = (1+mi_dat[i])*(1+mi_dat[j])*(cl_yy+cl_yi+cl_iy+cl_ii)

                # # Tomographic angular clustering Cl
                # cl_gg = ccl.angular_cl(fid_cosmo, g_i, g_j, ells)
                # np.savetxt(save_dir + 'fiducial_cosmology/galaxy_cl/bin_{}_{}.txt'.format(bin_i, bin_j), cl_gg)
                #
                # # Tomographic cosmic shear Cl
                # cl_yy = ccl.angular_cl(fid_cosmo, y_i, y_j, ells)
                # np.savetxt(save_dir + 'fiducial_cosmology/shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j), cl_yy)

    return cls_dict


def execute(pipeline_variables_path):

    config_dict = setup_config_dict(pipeline_variables_path=pipeline_variables_path)

    save_dir = config_dict['save_dir']
    nz_filename = config_dict['nz_filename']
    n_zbin = config_dict['n_zbin']
    ell_min = config_dict['ell_min']
    ell_max = config_dict['ell_max']

    fid_cosmo = CCL_cosmo(pipeline_variables_path=pipeline_variables_path)
    systematics_dict = setup_systematics_dict(pipeline_variables_path=pipeline_variables_path)
    setup_6x2pt_cls(
        save_dir=save_dir,
        nz_filename=nz_filename,
        n_zbin=n_zbin,
        ell_min=ell_min,
        ell_max=ell_max,
        fid_cosmo=fid_cosmo,
        systematics_dict=systematics_dict,
        mode='save')
