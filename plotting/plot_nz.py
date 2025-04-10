import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from collections import defaultdict

simulation_save_dir = '/raid/scratch/wongj/mywork/3x2pt/TEST_COV_AREAS_FINAL_superclass_data/pcl/'

nbins = 3

nz_dat = np.loadtxt(simulation_save_dir + 'Galaxy_nz.txt')

zs = nz_dat[:,0]

fig, ax = plt.subplots()
for i in range(nbins):
    ax.plot(zs, nz_dat[:,i+1])
plt.show()