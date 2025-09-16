import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from collections import defaultdict
import cmocean
import matplotlib
import cmasher as cmr
# import colormaps as cmaps
import colorcet as cc
import colorbrewer as cb

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'

simulation_save_dir = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/b/'

nbins = 6

nz_dat = np.loadtxt(simulation_save_dir + 'Galaxy_nz.txt')

# Choose colormap
# cmap_names = ['viridis_r', 'plasma_r', 'ocean_r', 'cividis_r']
# cmaps = [matplotlib.colormaps.get_cmap(cmap_name) for cmap_name in cmap_names]

# cmap = cmocean.cm.dense_r
cmap = cmr.amethyst
# cmap = matplotlib.cm.inferno

# Pick N numbers from 0 to 1
gradient = np.linspace(0.2, 0.8, 6)
colours = [matplotlib.colors.to_hex(cmap(i)) for i in gradient]

# colours = ['#332288','#44AA99', '#88CCEE','#DDCC77','#CC6677','#882255']

zs = nz_dat[:,0]

fig, ax = plt.subplots()
for i in range(nbins):
    ax.plot(zs, nz_dat[:,i+1],color=colours[i])

ax.set_xlabel(r'$z$', fontsize=20)
ax.set_ylabel(r'$N(z)$', fontsize=20)

# a.set_xscale('log')
# ax.set_yscale('log')

ax.tick_params(axis='both', labelsize=17.5)

ax.tick_params(which='both', axis='both', right=True, top=True, labelright=False, labeltop=False, left=True,
              bottom=True, labelleft=True, labelbottom=True, direction='in')
ax.tick_params(length=3, which='minor')
ax.tick_params(length=7.5, which='major')
ax.yaxis.offsetText.set_fontsize(15)

plt.minorticks_on()
plt.tight_layout()

# plt.savefig(simulation_save_dir + 'nz.png',dpi=200)

plt.show()