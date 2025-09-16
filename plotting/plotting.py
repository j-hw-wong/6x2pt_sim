import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'

colors = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']

simulation_save_dir = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/ff/'

no_realisations = 1
nbins = 6

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

    label = None

    if type == 'kk':

        label = "$\~{C}_\ell^{\,\kappa_{\mathrm{CMB}}\kappa_{\mathrm{CMB}}}$"
        # if n == 0:
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/cmbkappa_cl/PCl_Bandpowers_kCMB_kCMB_bin_1_1.txt'))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/cmbkappa_bp/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/cmbkappa_bp/bin_1_1.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/cmbkappa_bp/bin_1_1_err.txt'))

    elif type == 'ky':
        label = "$\~{C}_\ell^{\kappa_{\mathrm{CMB}}\gamma}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cmbkappa_cl/PCl_Bandpowers_kCMB_E_bin_{}_1.txt'.format(bin_i)))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_cmbkappa_bp/kCMB_E/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_cmbkappa_bp/kCMB_E/bin_{}_1.txt'.format(bin_i)))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_cmbkappa_bp/kCMB_E/bin_{}_1_err.txt'.format(bin_i)))

    elif type == 'kd':
        label = "$\~{C}_\ell^{\kappa_{\mathrm{CMB}}\delta_{\mathrm{g}}}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cmbkappa_cl/PCl_Bandpowers_kCMB_gal_bin_{}_1.txt'.format(bin_i)))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_cmbkappa_bp/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_cmbkappa_bp/bin_{}_1.txt'.format(bin_i)))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_cmbkappa_bp/bin_{}_1_err.txt'.format(bin_i)))

    elif type == 'yy':
        label = "$\~{C}_\ell^{\gamma\gamma}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'.format(bin_i, bin_j)))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_bp/Cl_EE/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_bp/Cl_EE/bin_{}_{}.txt'.format(bin_i, bin_j)))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/shear_bp/Cl_EE/bin_{}_{}_err.txt'.format(bin_i, bin_j)))

    elif type == 'dd':
        label = "$\~{C}_\ell^{\delta_{\mathrm{g}}\delta_{\mathrm{g}}}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'.format(bin_i, bin_j)))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_bp/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_bp/bin_{}_{}.txt'.format(bin_i, bin_j)))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_bp/bin_{}_{}_err.txt'.format(bin_i, bin_j)))

    elif type == 'dy':
        label = "$\~{C}_\ell^{\delta_{\mathrm{g}}\gamma}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'.format(bin_i, bin_j)))
        # theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/ell.txt'))
        # theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j)))

        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_shear_bp/ell.txt'))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_shear_bp/bin_{}_{}.txt'.format(bin_i, bin_j)))
        measured_cls.append(open_dat(save_dir + 'measured_6x2pt_bps/galaxy_shear_bp/bin_{}_{}_err.txt'.format(bin_i, bin_j)))
        # measured_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/ell.txt'))
        # measured_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j)))

    # measured_cls_av = np.mean(np.array(measured_cls[1:]), axis=0)

    return label, theory_cls, measured_cls


def plot_kCMB(save_dir):

    kCMB_label, kCMB_theory, kCMB_measured = process_cls(save_dir=simulation_save_dir,
                                                         type='kk')

    # CMB convergence
    f, a = plt.subplots()
    a.plot(kCMB_theory[0], kCMB_theory[1],color='0', linewidth=0.8)
    # a.plot(kCMB_measured[0], kCMB_measured[1],marker='x',ls='None')
    a.errorbar(kCMB_measured[0], kCMB_measured[1],xerr=None,yerr=kCMB_measured[2], marker='x',markersize=8, ls='None',color=colors[2])
    # a.plot(kCMB_measured[0], (kCMB_measured[1]- kCMB_theory[1])/ kCMB_theory[1],marker='x',ls='None')

    a.set_xlabel("$\\ell$", fontsize=20)
    a.set_ylabel(kCMB_label, fontsize=20)

    # a.set_xscale('log')
    a.set_yscale('log')

    a.tick_params(axis='both', labelsize=17.5)

    a.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                  bottom=True, labelleft=True, labelbottom=True, direction='in')
    a.tick_params(length=3, which='minor')
    a.tick_params(length=7.5, which='major')

    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(save_dir + 'cmbkappa.png')
    plt.show()


def plot_tom_xcorr(save_dir, nbins, type):

    fig = plt.figure(figsize=(10, 10))
    sz = 1.0 / (nbins + 2)

    for j in range(nbins):

        x_lows = []
        x_highs = []

        y_lows = []
        y_highs = []

        axes = []

        for i in range(nbins):
            if ((type == 'yy' or type == 'dd') and i >= j) or (type == 'dy') or ((type == 'ky' or type == 'kd') and i == j):
                print(i, j)

                labelstr, theory_cl, measured_cl = process_cls(save_dir=save_dir, type=type, bin_i=i+1, bin_j=j+1)

                rect = ((i+1)*sz,(j+1)*sz,sz,sz)
                ax = fig.add_axes(rect)

                plt.plot(theory_cl[0], theory_cl[1],zorder=1,color='0', linewidth=0.8)
                # plt.plot(measured_cl[0], measured_cl[1],zorder=2,marker='x',linestyle='None')
                plt.errorbar(measured_cl[0], measured_cl[1],xerr=None,yerr=measured_cl[2],zorder=2,marker='x',markersize=6, ls='None',color=colors[2])

                # print(theory_cl[1])
                # plt.plot(theory_cl[0], (measured_cl[1] - theory_cl[1])/theory_cl[1],zorder=1,color='0',alpha=0.5)

                plt.xscale('log')
                # plt.yscale('log')

                if j == 0:
                    plt.xlabel("$\\ell$", fontsize=17.5)

                if i == 1:
                    # labelstr = str("$C_\ell^{\delta_{g}\delta_{g}}$")
                    plt.ylabel(labelstr, fontsize=17.5)
                    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                if j != 0:
                    plt.gca().xaxis.set_ticklabels([])

                if i != j:
                    plt.gca().yaxis.set_ticklabels([])

                ax.minorticks_on()

                ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                               bottom=True, labelleft=True, labelbottom=True, direction='in')
                ax.tick_params(length=3, which='minor')
                ax.tick_params(length=7.5, which='major')

                plt.minorticks_on()
                ax.tick_params(labelsize=12.5)

                plt.text(0.125, 0.75, "("r'$z_{%d}$' ", "r'$z_{%d}$'")" % (i+1, j+1), fontsize=12.5, color='black',
                         transform=ax.transAxes)
                axes.append(ax)

                left, right = ax.get_xlim()
                x_lows.append(left)
                x_highs.append(right)

                lower, upper = ax.get_ylim()
                y_lows.append(lower)
                y_highs.append(upper)

        for a in axes:
            a.set_xlim(min(x_lows),max(x_highs))
            a.set_ylim(abs(min(y_lows)),1.1*max(y_highs))

    plt.savefig(save_dir + '{}.png'.format(type))
    plt.show()

# plot_kCMB(save_dir=simulation_save_dir)

plot_tom_xcorr(save_dir=simulation_save_dir, nbins=nbins, type='kd')
