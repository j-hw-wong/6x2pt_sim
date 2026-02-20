import corner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from plotting import spider_plot
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.serif'] = 'cm'


def nautilus_posterior_plotting(sampler, sampler2, sampler3, sampler4):

    """
    Plot posterior corner plot

    Parameters
    ----------
    sampler (nautilus sampler):     Instance of nautilus sampler for given set up
    sampler2 (nautilus sampler):    Instance of nautilus sampler for given set up
    sampler3 (nautilus sampler):    Instance of nautilus sampler for given set up
    sampler4 (nautilus sampler):    Instance of nautilus sampler for given set up

    Returns
    -------
    Corner plot of parameters
    """

    points, log_w, log_l = sampler.posterior()
    # print(points, log_w, log_l)

    # max_like_id = log_l.argmax()
    # print(max_like_id)
    # w0_m, wa_m, Omega_c_m, h_m,  = points[max_like_id]
    # print(w0_m, wa_m, Omega_c_m, h_m)
    # corner.corner(
    #     points, weights=np.exp(log_w), bins=30, labels=prior.keys
    # )
    # plt.show()

    points = points[:, 0:7]

    # max_like_id = log_l.argmax()
    # print(max_like_id)
    # w0_m, wa_m, Omega_c_m, h_m,  = points[max_like_id]
    # print(w0_m, wa_m, Omega_c_m, h_m)

    one_sigma_1d = 0.683

    q_lower = 1 / 2 - one_sigma_1d / 2
    q_upper = 1 / 2 + one_sigma_1d / 2
    colours = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    colours = ['#1b9e77','#d95f02','#7570b3','#e7298a']
    colours = ['royalblue', 'darkred', '#417B5A']
    # colour_3x2pt_analytic = 'xkcd:faded red'
    colour_3x2pt_analytic = 'darkred'
    colour_3x2pt_numerical = 'xkcd:midnight purple'
    colour_6x2pt_analytic = 'xkcd:soft blue'
    colour_6x2pt_numerical = 'xkcd:very dark green'
    # colour_6x2pt_numerical = 'xkcd:dusty green'
    # colour_6x2pt_numerical = 'xkcd:dull teal'
    # colours = ['xkcd:lightish blue', 'darkred', 'xkcd:midnight purple','xkcd:midnight purple']
    # colours = ['xkcd:lightish blue', '#FF7070', 'xkcd:midnight purple','xkcd:midnight purple']

    hist_bins = [20, 20, 100, 30, 40, 100, 100]

    figure = corner.corner(
        points,
        weights=np.exp(log_w),
        bins=hist_bins,  # ,5000,300],
        plot_density=False,
        fill_contours=True,
        # color='royalblue',
        color=colour_6x2pt_analytic,
        data_kwargs={'color': '0.45', 'ms': '0'},
        label_kwargs={'fontsize': '20'},
        hist_kwargs={'linewidth':1.75},
        labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$'],
        labelpad=0.025,
        levels=(0.683, 0.955),
        smooth=1.5,
        smooth1d=True,
        # title_quantiles=[q_lower, 0.5, q_upper],
        # show_titles=True,
        # title_fmt='.4f'
    )

    points2, log_w2, log_l2 = sampler2.posterior()
    points2 = points2[:,0:7]

    corner.corner(
        points2,
        weights=np.exp(log_w2),
        bins=hist_bins,
        plot_density=False,
        # fill_contours=True,
        no_fill_contours=True,
        color=colour_6x2pt_numerical,
        # color='xkcd:very dark green',
        # contour_kwargs={'linestyles':'--','linewidths':2.5,},
        contour_kwargs={'linestyles':'-'},
        data_kwargs={'color': '0.45', 'ms': '0'},
        label_kwargs={'fontsize': '20'},
        labelpad=0.025,
        levels=(0.683, 0.955),
        smooth=1.5,
        smooth1d=True,
        fig=figure,
        # hist_kwargs={'linestyle':'--','linewidth':2.5,'dashes':(8, 6)},
        hist_kwargs={'linestyle':'-','linewidth':1.75},
        hist2d_kwargs={'contour_kwargs':{'linestyles': '--'}}
    )

    points3, log_w3, log_l3 = sampler3.posterior()
    points3 = points3[:, 0:7]

    corner.corner(
        points3,
        weights=np.exp(log_w3),
        bins=hist_bins,
        plot_density=False,
        # no_fill_contours=False,
        fill_contours=True,
        color=colour_3x2pt_analytic,
        # contour_kwargs={'linestyles': '--', 'linewidths': 1.5},
        data_kwargs={'color': '0.45', 'ms': '0'},
        label_kwargs={'fontsize': '20'},
        labelpad=0.025,
        levels=(0.683, 0.955),
        smooth=1.5,
        smooth1d=True,
        fig=figure,
        hist_kwargs={'linestyle': '-', 'linewidth': 1.75},
        # hist2d_kwargs={'contour_kwargs': {'linestyles': '-'}}
    )

    points4, log_w4, log_l4 = sampler4.posterior()
    points4 = points4[:, 0:7]

    corner.corner(
        points4,
        weights=np.exp(log_w4),
        bins=hist_bins,
        plot_density=False,
        no_fill_contours=True,
        color=colour_3x2pt_numerical,
        contour_kwargs={'linestyles': ':', 'linewidths': 2.5},
        data_kwargs={'color': '0.45', 'ms': '0'},
        label_kwargs={'fontsize': '20'},
        labelpad=0.025,
        levels=(0.683, 0.955),
        smooth=1.5,
        smooth1d=True,
        fig=figure,
        # hist_kwargs={'linestyle': ':', 'linewidth': 3.},
        hist_kwargs={'linestyle': ':', 'linewidth': 3.,'dashes': (1, 1.75)},
        hist2d_kwargs={'contour_kwargs': {'linestyles': '-'}}
    )

    ndim = len(points[:,][0:7])

    # # Format the quantile display.
    # fmt = "{{0:{0}}}".format(title_fmt).format
    # title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    # title = title.format(fmt(q_mid), fmt(q_m), fmt(q_p))

    # Print some best fit values with their 1 sigma errors
    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid

        print(np.array([q_mid, q_p, q_m]))

    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points2[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w2)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid

        print(np.array([q_mid, q_p, q_m]))

    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points3[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w3)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid

        print(np.array([q_mid, q_p, q_m]))

    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points4[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w4)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid

        print(np.array([q_mid, q_p, q_m]))

    conf_bound = 0.999999999999999

    # Set the x and y ranges on the posterior plots. We can do this automatically by cutting at a given confidence
    # bound/interval (conf_bound)
    xranges = []

    for i in range(ndim):

        q_lo, q_mid, q_hi = corner.quantile(
            points[:, i], [1 / 2 - conf_bound / 2, 0.5, 1 / 2 + conf_bound / 2], weights=np.exp(log_w)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid
        xranges.append([q_lo, q_hi])

    # Or set the x ranges manually
    # xranges = [
    #     [-1.25, -0.75],
    #     [-0.5, 0.5],
    #     [0.27, 0.39],
    #     [0.5, 0.8],
    #     [0.02, 0.075],
    #     [0.88, 1.05],
    #     [0.8, 0.875]]

    yranges = xranges[1:]

    for ax in figure.get_axes():
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='both', direction='in', labelsize=20)

    axes = np.array(figure.axes).reshape((ndim, ndim))

    fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048] # for cosmo

    for i in range(ndim):
        ax = axes[i, i]
        ax.set_xlim(xranges[i][0], xranges[i][1])
        ax.axvline(fid_vals[i], color='0.5', linestyle=':')

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.set_xlim(xranges[xi][0], xranges[xi][1])
            ax.set_ylim(yranges[yi - 1][0], yranges[yi - 1][1])
            ax.axvline(fid_vals[xi], color='0.5', linestyle=':')
            ax.axhline(fid_vals[yi], color='0.5', linestyle=':')

    patch1 = mpatches.Patch(color=colour_6x2pt_analytic, label='Stage IV-like 6x2pt (6 Bin)\nAnalytic Covariance Matrix')
    patch2 = mpatches.Patch(color=colour_3x2pt_analytic, label='Stage IV-like 3x2pt (6 Bin)\nAnalytic Covariance Matrix')
    line1 = Line2D([0], [0], color=colour_6x2pt_numerical, lw=4, ls='--', label='Stage IV-like 6x2pt (6 Bin)\nNumerical Covariance Matrix')
    line2 = Line2D([0], [0], color=colour_3x2pt_numerical, lw=4, ls=':', label='Stage IV-like 3x2pt (6 Bin)\nNumerical Covariance Matrix')
    figure.legend(handles=[patch1, line1, patch2, line2], loc='center right', fontsize=20)  # legend can be centre right for big plots
    save_fig_dir = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/b/inference_chains/P4.png'

    # if os.path.exists(save_fig_dir):
    #     print('WARNING! File exists, did not overwrite')
    # else:
    #     plt.savefig(save_fig_dir,dpi=200)

    plt.show()

    # spider_plot.spiderplot(spider_plot.spider_data4(sampler1=sampler2, sampler2=sampler, sampler3=sampler2, sampler4=sampler))


