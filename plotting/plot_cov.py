import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

# save_dir = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/pcl/'
save_dir = '/home/wongj/'
n_zbin = 8

n_bps = 10

obs_type = '3x2pt'
obs_field = 'E'

if obs_type == '6x2pt':

	n_field = 2 * n_zbin
	n_spec = int((n_zbin*((n_zbin+1))) + (n_zbin**2) + (2*n_zbin) + 1)
	fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]

	spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

	for i in range(n_zbin):
		spectra.append('E{}K1'.format(i+1))
		spectra.append('N{}K1'.format(i+1))

	spectra.append('K1K1')

elif obs_type == '3x2pt':

	n_field = 2 * n_zbin
	n_spec = int((n_zbin*((n_zbin+1))) + (n_zbin**2))
	fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]
	spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

elif obs_type == '1x2pt':

	n_field = n_zbin
	if obs_field == 'E':
		n_spec = int((n_zbin*((n_zbin+1)))/2)
		fields = [f'E{z}' for z in range(1, n_zbin + 1)]
		spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

	elif obs_field == 'N':
		n_spec = int((n_zbin*((n_zbin+1)))/2)
		fields = [f'N{z}' for z in range(1, n_zbin + 1)]
		spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

	elif obs_field == 'K':
		n_spec = 1
		spectra = ['K1K1']

	elif obs_field == 'EK':
		n_spec = n_zbin
		spectra = [f'E{z}K1' for z in range(1, n_zbin + 1)]

	elif obs_field == 'NK':
		n_spec = n_zbin
		spectra = [f'N{z}K1' for z in range(1, n_zbin + 1)]


#elif obs_type == '1x2pt':

cov_a = np.load(save_dir + 'cov_10bp.npz')['cov']
cov_n = np.load(save_dir + 'cov_10bp_num.npz')['cov']

#delta = cov_400 - cov_200

min_val = np.amin(np.abs(cov_n[cov_n>0]))

ticks = np.linspace(n_bps/2,(n_spec*n_bps)-(n_bps/2), n_spec)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

ax1.matshow((cov_a), norm=SymLogNorm(linthresh=1e-18), vmin=min_val)
ax2.matshow((cov_n), norm=SymLogNorm(linthresh=1e-18), vmin=min_val)
# ax2.matshow((cov_n-cov_a)*100/cov_a, norm=LogNorm(), vmin=min_val)

for ax in [ax1, ax2]:

	ax.set_xticks(ticks=ticks)
	ax.set_xticklabels(spectra, rotation=45, fontsize=8)
	ax.set_yticks(ticks=ticks)
	ax.set_yticklabels(spectra, fontsize=8)

# plt.savefig(save_dir + 'covmat_comparison_E_new.png')
plt.show()

# plt.plot(np.arange(len(spectra)*10),np.diag(cov_a))
# plt.plot(np.arange(len(spectra)*10),np.diag(cov_n))
# plt.plot(np.arange(len(spectra)*10),(np.diag(cov_n)-np.diag(cov_a))/np.diag(cov_a))
# plt.plot(np.arange(len(spectra)*10),(np.diag(cov_n)-np.diag(cov_a))/np.diag(cov_a))
print(np.mean((np.diag(cov_n)-np.diag(cov_a))/np.diag(cov_a)))
plt.show()