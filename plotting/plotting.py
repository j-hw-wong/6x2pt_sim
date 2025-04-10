import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from collections import defaultdict

simulation_save_dir = '/raid/scratch/wongj/mywork/3x2pt/TEST_COV_AREAS_FINAL_superclass_data/j/'

no_realisations = 1
nbins = 3
lmax = 250

def open_dat(fname):
    dat_arr = []
    with open(fname) as f:
        for line in f:
            column = line.split()
            if not line.startswith('#'):
                dat_i = float(column[0])
                dat_arr.append(dat_i)
    dat_arr = np.asarray(dat_arr)
    return dat_arr


def process_cls(save_dir, type, bin_i=None, bin_j=None):

    if type not in ['kk', 'ky', 'kd', 'yy', 'dd', 'dy']:
        print('Warning! XCorr Type Not Recognised - Exiting...')
        sys.exit()

    theory_cls = []
    measured_cls = []

    if type == 'kk':

        label = "$\~{C}_\ell^{\kappa_{\mathrm{CMB}}\kappa_{\mathrm{CMB}}}$"
        # if n == 0:
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/cmbkappa_cl/PCl_Bandpowers_kCMB_kCMB_bin_1_1.txt'))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/cmbkappa_bp/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/cmbkappa_bp/bin_1_1.txt'))

    elif type == 'ky':
        label = "$\~{C}_\ell^{\kappa_{\mathrm{CMB}}\gamma}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cmbkappa_cl/PCl_Bandpowers_kCMB_E_bin_{}_1.txt'.format(bin_i)))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_cmbkappa_bp/kCMB_E/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_cmbkappa_bp/kCMB_E/bin_{}_1.txt'.format(bin_i)))

    elif type == 'kd':
        label = "$\~{C}_\ell^{\kappa_{\mathrm{CMB}}\delta_{\mathrm{g}}}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cmbkappa_cl/PCl_Bandpowers_kCMB_gal_bin_{}_1.txt'.format(bin_i)))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_cmbkappa_bp/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_cmbkappa_bp/bin_{}_1.txt'.format(bin_i)))

    elif type == 'yy':
        label = "$\~{C}_\ell^{\gamma\gamma}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'.format(bin_i, bin_j)))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_bp/Cl_EE/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_bp/Cl_EE/bin_{}_{}.txt'.format(bin_i, bin_j)))

    elif type == 'dd':
        label = "$\~{C}_\ell^{\delta_{\mathrm{g}}\delta_{\mathrm{g}}}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'.format(bin_i, bin_j)))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_bp/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_bp/bin_{}_{}.txt'.format(bin_i, bin_j)))

    elif type == 'dy':
        label = "$\~{C}_\ell^{\delta_{\mathrm{g}}\gamma}$"
        # theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/ell_bp.txt'))
        # theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'.format(bin_i, bin_j)))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/ell.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j)))

        # measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_shear_bp/ell.txt'))
        # measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_shear_bp/bin_{}_{}.txt'.format(bin_i, bin_j)))
        measured_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j)))

    # measured_cls_av = np.mean(np.array(measured_cls[1:]), axis=0)

    return label, theory_cls, measured_cls


def plot_kCMB(save_dir):

    kCMB_label, kCMB_theory, kCMB_measured = process_cls(save_dir=simulation_save_dir,
                                                         type='kk')

    # CMB convergence
    f, a = plt.subplots()
    a.plot(kCMB_theory[0], kCMB_theory[1],color='0',alpha=0.5)
    a.plot(kCMB_measured[0], kCMB_measured[1],marker='x',ls='None')
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel("$\\ell$", fontsize=15)
    a.set_ylabel(kCMB_label, fontsize=15)
    plt.tight_layout()
    plt.savefig(save_dir + 'cmbkappa.png')
    plt.show()


# kCMB_label, kCMB_theory, kCMB_measured = process_cls(, type='kk')
# plot_kCMB(save_dir=simulation_save_dir)


def plot_tom_xcorr(save_dir, nbins, type):

    fig = plt.figure(figsize=(10, 10))
    sz = 1.0 / (nbins + 2)

    for i in range(nbins):
        for j in range(nbins):
            if ((type == 'yy' or type == 'dd') and i >= j) or (type == 'dy') or ((type == 'ky' or type == 'kd') and i == j):

                labelstr, theory_cl, measured_cl = process_cls(save_dir=save_dir, type=type, bin_i=i+1, bin_j=j+1)

                rect = ((i+1)*sz,(j+1)*sz,sz,sz)
                ax = fig.add_axes(rect)

                plt.plot(theory_cl[0], theory_cl[1],zorder=1,color='0',alpha=0.5)
                # plt.plot(measured_cl[0], measured_cl[1],zorder=2,marker='x',linestyle='None')
                # plt.xscale('log')
                # plt.yscale('log')

                if j == 0:
                    plt.xlabel("$\\ell$", fontsize=15)

                if i == j:
                    # labelstr = str("$C_\ell^{\delta_{g}\delta_{g}}$")
                    plt.ylabel(labelstr, fontsize=15)
                    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                if j != 0:
                    plt.gca().xaxis.set_ticklabels([])

                if i != j:
                    plt.gca().yaxis.set_ticklabels([])

                ax.minorticks_on()

                ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                               bottom=True, labelleft=True, labelbottom=True, direction='in')
                ax.tick_params(length=2.5, which='minor')
                ax.tick_params(length=5.5, which='major')
                ax.tick_params(labelsize=12.5)

                plt.text(0.125, 0.75, "("r'$z_{%d}$' ", "r'$z_{%d}$'")" % (i+1, j+1), fontsize=15, color='black',
                         transform=ax.transAxes)

    plt.savefig(save_dir + '{}.png'.format(type))
    plt.show()


plot_tom_xcorr(save_dir=simulation_save_dir,nbins=nbins, type='dy')
