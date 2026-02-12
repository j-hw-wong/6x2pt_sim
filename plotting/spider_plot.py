import matplotlib.pyplot as plt
import numpy as np
import corner

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def spider_data(sampler1, sampler2):

    points, log_w, log_l = sampler1.posterior()
    points_comparison, log_w_comparison, log_l_comparison = sampler2.posterior()

    # ndim = 12
    ndim = 7

    one_sigma_1d = 0.683
    # one_sigma_1d = 0.955

    q_lower = 1 / 2 - one_sigma_1d / 2
    q_upper = 1 / 2 + one_sigma_1d / 2

    sigmas = []
    sigmas_comparison = []

    sigmas3 = []
    sigmas4 = []

    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid
        sigmas.append((q_m+q_p)/2)

    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points_comparison[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w_comparison)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid
        sigmas_comparison.append((q_m+q_p)/2)

    print(np.asarray(sigmas_comparison) / np.asarray(sigmas))
    print(np.round(np.asarray(sigmas_comparison)/np.asarray(sigmas), decimals=2))

    data = [
        # [r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{c}$', r'$n_{s}$', r'$\sigma_{8}$', r'$A_{1}$', r'$A_{2}$', r'$b_{TA}$', r'$\eta_{1}$', r'$\eta_{2}$'],
        [r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{c}$', r'$n_{s}$', r'$\sigma_{8}$', r'$b_{1}$', r'$b_{2}$', r'$b_{s}$'],
        ('6x2pt / 3x2pt', [
            np.round(np.asarray(sigmas_comparison)/np.asarray(sigmas), decimals=2)
        ])
    ]

    return data


def spider_data4(sampler1, sampler2, sampler3, sampler4):

    points, log_w, log_l = sampler1.posterior()
    points_comparison, log_w_comparison, log_l_comparison = sampler2.posterior()

    points3, log_w3, log_l3 = sampler3.posterior()
    points4, log_w4, log_l4 = sampler4.posterior()

    ndim = 7

    one_sigma_1d = 0.683
    # one_sigma_1d = 0.955

    q_lower = 1 / 2 - one_sigma_1d / 2
    q_upper = 1 / 2 + one_sigma_1d / 2

    sigmas = []
    sigmas_comparison = []

    sigmas3 = []
    sigmas4 = []

    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid
        sigmas.append((q_m+q_p)/2)

    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points_comparison[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w_comparison)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid
        sigmas_comparison.append((q_m+q_p)/2)

    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points3[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w3)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid
        sigmas3.append((q_m+q_p)/2)

    for i in range(ndim):
        q_lo, q_mid, q_hi = corner.quantile(
            points4[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w4)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid
        sigmas4.append((q_m+q_p)/2)

    print(np.asarray(sigmas_comparison) / np.asarray(sigmas))
    print(np.asarray(sigmas4) / np.asarray(sigmas3))


    # spider_data = np.asarray(sigmas_comparison)/np.asarray(sigmas)

    # print(spider_data)

    print(np.round(np.asarray(sigmas_comparison)/np.asarray(sigmas), decimals=2))
    print(np.round(np.asarray(sigmas4)/np.asarray(sigmas3), decimals=2))

    data = [
        [r'$w_{0}$', r'$w_{a}$ ', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$'], #, r'$A_{1}$', r'$A_{2}$', r'$b_{TA}$', r'$\eta_{1}$', r'$\eta_{2}$'],
        ('6x2pt / 3x2pt', [
            np.round(np.asarray(sigmas_comparison)/np.asarray(sigmas), decimals=5),
            np.round(np.asarray(sigmas4)/np.asarray(sigmas3), decimals=5)
        ])
    ]

    return data


def spiderplot(data):
    # N = 12
    N = 7
    theta = radar_factory(N, frame='polygon')

    spoke_labels = data.pop(0)

    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['r','b', 'g', 'm', 'y']
    # colors = ['#800074','#298C8C']
    markers = ['o','^']
    # Plot the four cases from the example data on separate Axes
    # for ax, (title, case_data) in zip(axs.flat, data):

    title, case_data = data[0]
    ax.plot(theta, np.ones(N), color='0',ls='--', label='_nolegend_')

    ax.set_title('3 Tomographic Bins\n(Low-'+r'$z$'+')', weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')
    for d, color, marker in zip(case_data, colors, markers):
        ax.plot(theta, d, color=color,marker=marker,markersize=7.5)
        ax.fill(theta, d, facecolor=color, alpha=0.075, label='_nolegend_')

    ax.set_varlabels(spoke_labels)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], va='center')
    ax.set_ylim(0, 1.15)
    # add legend relative to top-left plot
    labels = ['Numerical\nCov. Matrix','Analytic\nCov. Matrix']
    # labels = ['3 Tomographic Bins','6 Tomographic Bins']
    # labels = ('Ratio of 6x2pt')
    # legend = ax.legend(labels, loc=(0.75, 1.),
    legend = ax.legend(labels, loc=(0.725, .925),
                              labelspacing=0.1)
    plt.tight_layout()
    # plt.savefig('/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/b/inference_chains/Spider_3bin.png',dpi=200)
    plt.show()
