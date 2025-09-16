import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'

colors = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']

measurement_save_dir = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/j/'
simulation_save_dir = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/j/'
# simulation_save_dir2 = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/n3/'
# simulation_save_dir3 = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/n3b/'
simulation_save_dir2 = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/bbias/'
simulation_save_dir3 = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/bias_bestfit/'

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


def process_cls(save_dir, save_dir2, save_dir3, type, bin_i=None, bin_j=None):

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
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/cmbkappa_cl/PCl_Bandpowers_kCMB_kCMB_bin_1_1.txt'))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/cmbkappa_cl/PCl_Bandpowers_kCMB_kCMB_bin_1_1.txt'))

        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/cmbkappa_bp/ell.txt'))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/cmbkappa_bp/bin_1_1.txt'))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/cmbkappa_bp/bin_1_1_err.txt'))

    elif type == 'ky':
        label = "$\~{C}_\ell^{\kappa_{\mathrm{CMB}}\gamma}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cmbkappa_cl/PCl_Bandpowers_kCMB_E_bin_{}_1.txt'.format(bin_i)))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/shear_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/shear_cmbkappa_cl/PCl_Bandpowers_kCMB_E_bin_{}_1.txt'.format(bin_i)))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/shear_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/shear_cmbkappa_cl/PCl_Bandpowers_kCMB_E_bin_{}_1.txt'.format(bin_i)))

        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/shear_cmbkappa_bp/kCMB_E/ell.txt'))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/shear_cmbkappa_bp/kCMB_E/bin_{}_1.txt'.format(bin_i)))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/shear_cmbkappa_bp/kCMB_E/bin_{}_1_err.txt'.format(bin_i)))

    elif type == 'kd':
        label = "$\~{C}_\ell^{\kappa_{\mathrm{CMB}}\delta_{\mathrm{g}}}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cmbkappa_cl/PCl_Bandpowers_kCMB_gal_bin_{}_1.txt'.format(bin_i)))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/galaxy_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/galaxy_cmbkappa_cl/PCl_Bandpowers_kCMB_gal_bin_{}_1.txt'.format(bin_i)))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/galaxy_cmbkappa_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/galaxy_cmbkappa_cl/PCl_Bandpowers_kCMB_gal_bin_{}_1.txt'.format(bin_i)))

        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/galaxy_cmbkappa_bp/ell.txt'))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/galaxy_cmbkappa_bp/bin_{}_1.txt'.format(bin_i)))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/galaxy_cmbkappa_bp/bin_{}_1_err.txt'.format(bin_i)))

    elif type == 'yy':
        label = "$\~{C}_\ell^{\gamma\gamma}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'.format(bin_i, bin_j)))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/shear_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'.format(bin_i, bin_j)))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/shear_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'.format(bin_i, bin_j)))

        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/shear_bp/Cl_EE/ell.txt'))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/shear_bp/Cl_EE/bin_{}_{}.txt'.format(bin_i, bin_j)))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/shear_bp/Cl_EE/bin_{}_{}_err.txt'.format(bin_i, bin_j)))

    elif type == 'dd':
        label = "$\~{C}_\ell^{\delta_{\mathrm{g}}\delta_{\mathrm{g}}}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'.format(bin_i, bin_j)))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/galaxy_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'.format(bin_i, bin_j)))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/galaxy_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'.format(bin_i, bin_j)))

        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/galaxy_bp/ell.txt'))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/galaxy_bp/bin_{}_{}.txt'.format(bin_i, bin_j)))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/galaxy_bp/bin_{}_{}_err.txt'.format(bin_i, bin_j)))

    elif type == 'dy':
        label = "$\~{C}_\ell^{\delta_{\mathrm{g}}\gamma}$"
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'.format(bin_i, bin_j)))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/galaxy_shear_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir2 + 'fiducial_cosmology/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'.format(bin_i, bin_j)))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/galaxy_shear_cl/ell_bp.txt'))
        theory_cls.append(open_dat(save_dir3 + 'fiducial_cosmology/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'.format(bin_i, bin_j)))

        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/galaxy_shear_bp/ell.txt'))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/galaxy_shear_bp/bin_{}_{}.txt'.format(bin_i, bin_j)))
        measured_cls.append(open_dat(measurement_save_dir + 'measured_6x2pt_bps/galaxy_shear_bp/bin_{}_{}_err.txt'.format(bin_i, bin_j)))
        # measured_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/ell.txt'))
        # measured_cls.append(open_dat(save_dir + 'fiducial_cosmology/galaxy_shear_cl/bin_{}_{}.txt'.format(bin_i, bin_j)))

    # measured_cls_av = np.mean(np.array(measured_cls[1:]), axis=0)

    return label, theory_cls, measured_cls


def plot_kCMB(save_dir):

    kCMB_label, kCMB_theory, kCMB_measured = process_cls(save_dir=simulation_save_dir,
                                                         save_dir2=simulation_save_dir2,
                                                         save_dir3=simulation_save_dir3,
                                                         type='kk')

    # CMB convergence
    f, a = plt.subplots()
    a.plot(kCMB_theory[0], kCMB_theory[1],color='0', linewidth=0.8)
    a.plot(kCMB_theory[2], kCMB_theory[3],color='0', linewidth=0.8,ls='--')
    a.plot(kCMB_theory[4], kCMB_theory[5],color='0', linewidth=0.8,ls=':')
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
    # plt.savefig(save_dir + 'cmbkappa.png')
    line1 = Line2D([0], [0], label='Fiducial Model (Nonlinear Bias)', color='0', linestyle='-')
    line2 = Line2D([0], [0], label='Fiducial + Constant Bias per Bin', color='0', linestyle='--')
    line3 = Line2D([0], [0], label='Best Fit', color='0', linestyle=':')

    plt.legend(handles=[line1,line2,line3],loc='upper left',fontsize=13.5)
    plt.savefig(save_dir + '{}.png'.format('kk'))
    plt.show()


def plot_tom_xcorr(save_dir, save_dir2, save_dir3, nbins, type):

    fig = plt.figure(figsize=(10, 10))
    sz = 1.0 / (nbins + 2)

    for j in range(nbins):

        x_lows = []
        x_highs = []

        y_lows = []
        y_highs = []

        axes = []

        for i in range(nbins):
            if ((type == 'yy' or type == 'dd') and i >= j) or (type == 'dy') or ((type == 'ky' or type == 'kd') and j==0):

                # print(i, j)

                labelstr, theory_cl, measured_cl = process_cls(save_dir=save_dir, save_dir2=save_dir2, save_dir3=save_dir3, type=type, bin_i=i+1, bin_j=j+1)
                if type == 'ky' or type == 'kd':
                    rect = ((i + 1) * sz + (i*0.0175), (1) * sz, sz, sz)
                else:
                    rect = ((i+1)*sz,(j+1)*sz,sz,sz)
                ax = fig.add_axes(rect)

                plt.plot(theory_cl[0], theory_cl[1],zorder=1,color='0', linewidth=0.8)
                plt.plot(theory_cl[2], theory_cl[3],zorder=1,color='0', linewidth=0.8,ls='--')
                plt.plot(theory_cl[4], theory_cl[5],zorder=1,color='0', linewidth=0.8,ls=':')
                # plt.plot(measured_cl[0], measured_cl[1],zorder=2,marker='x',linestyle='None')
                plt.errorbar(measured_cl[0], measured_cl[1],xerr=None,yerr=measured_cl[2],zorder=2,marker='x',markersize=6, ls='None',color=colors[2])

                # print(theory_cl[1])
                # plt.plot(theory_cl[0], (measured_cl[1] - theory_cl[1])/theory_cl[1],zorder=1,color='0',alpha=0.5)

                plt.xscale('log')
                # plt.yscale('log')

                ax.minorticks_on()

                ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
                               bottom=True, labelleft=True, labelbottom=True, direction='in')
                ax.tick_params(length=3, which='minor')
                ax.tick_params(length=7.5, which='major')

                plt.minorticks_on()
                ax.tick_params(which='major',labelsize=12.5)
                ax.tick_params(which='minor',labelleft=False)
                ax.tick_params(which='minor',labelbottom=False)

                plt.text(0.125, 0.75, "("r'$z_{%d}$' ", "r'$z_{%d}$'")" % (i+1, j+1), fontsize=12.5, color='black',
                         transform=ax.transAxes)
                axes.append(ax)

                left, right = ax.get_xlim()
                x_lows.append(left)
                x_highs.append(right)

                lower, upper = ax.get_ylim()
                y_lows.append(lower)
                y_highs.append(upper)

                plt.gca().xaxis.set_major_formatter(ScalarFormatter())
                plt.gca().xaxis.set_minor_formatter(ScalarFormatter())
                # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

                if type == 'yy' or type == 'dd':
                    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                    plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    if j == 0:
                        plt.xlabel("$\\ell$", fontsize=17.5)

                    if i == j:
                        # labelstr = str("$C_\ell^{\delta_{g}\delta_{g}}$")
                        plt.ylabel(labelstr, fontsize=17.5)
                        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                    if j != 0:
                        plt.gca().xaxis.set_ticklabels([])

                    if i != j:
                        plt.gca().yaxis.set_ticklabels([])

                    if type == 'yy':
                        ax.set_xticks([0,500,1000,1500])
                    else:
                        ax.set_xticks([0,200,400,600,800,1000])

                if type == 'ky' or type == 'kd':
                    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                    plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    plt.xlabel("$\\ell$", fontsize=17.5)

                    if i == 0 and j ==0:
                        plt.ylabel(labelstr, fontsize=17.5)
                        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                    if i != 0:
                        plt.gca().yaxis.set_ticklabels([])

                    ax.set_xticks([0,200,400,600,800,1000])

                if type == 'dy':
                    # ax = plt.gca()

                    def MyFormatter(x, lim):
                        if x == 0:
                            return 0
                        return '{0:.2f}e{1:.0f}'.format(
                            np.sign(x) * 10 ** (-np.floor(np.log10(abs(x))) + np.log10(abs(x))),
                            np.floor(np.log10(abs(x))))
                        # The first argument of the format gives the first significant digits of the number with the sign preserved and brought to a range between [1-10), The next argument gives the  numbers integer exponent of 10
                        # Both the first and second arguments are formatted to display only 2 decimal places due to the lack of space.

                    majorFormatter = FuncFormatter(MyFormatter)
                    ax.get_yaxis().set_major_formatter(majorFormatter)

                    if j == 0:
                        plt.xlabel("$\\ell$", fontsize=17.5)

                    if i == 0:
                        # labelstr = str("$C_\ell^{\delta_{g}\delta_{g}}$")
                        plt.ylabel(labelstr, fontsize=17.5)
                        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                    if j != 0:
                        plt.gca().xaxis.set_ticklabels([])

                    if i != 0:
                        plt.gca().yaxis.set_ticklabels([])

                    # ax.get_yaxis().get_offset_text().set_position((-0.175, 0))
                    ax.set_xticks([0,200,400,600,800,1000])

        min_factors = [-5,-15,-45,-15,-5,-1.5]
        for a in axes:
            a.set_xlim(min(x_lows),max(x_highs))
            a.set_ylim(min_factors[j]*abs(min(y_lows)),1.1*max(y_highs))

    line1 = Line2D([0], [0], label='Fiducial Model (Nonlinear Bias)', color='0', linestyle='-')
    line2 = Line2D([0], [0], label='Fiducial + Constant Bias per Bin', color='0', linestyle='--')
    line3 = Line2D([0], [0], label='Best Fit', color='0', linestyle=':')

    if type == 'yy' or type == 'dd' or type == 'dy':
        legend_loc = 'upper center'
    elif type == 'ky' or type == 'kd':
        legend_loc = 'center'
    else:
        legend_loc = None
    fig.legend(handles=[line1,line2,line3],loc=legend_loc,fontsize=13.5)
    plt.savefig(save_dir + '{}.png'.format(type),dpi=200)
    plt.show()

# plot_kCMB(save_dir=simulation_save_dir)

plot_tom_xcorr(save_dir=simulation_save_dir, save_dir2=simulation_save_dir2, save_dir3=simulation_save_dir3, nbins=nbins, type='ky')
