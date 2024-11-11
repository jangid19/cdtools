import cdtools
from cdtools.tools import plotting as p
from matplotlib import pyplot as plt
import numpy as np
import h5py

# prop_dist_list = [30, 35, 40, 45, 50, 55, 60]

scan = 198907
dist = 12.5
results_dir = '/home/rjangid/GitHub/cdtools_results/NIST17_results/'

# print(results_dir+f'Scan_{scan}_results.h5')
# print(results_dir+f'Scan_{scan}_dis_{dist}_results.h5')

# f = h5py.File(results_dir+f'Scan_{scan}_results.h5','r')
# print(list(f))


# data = np.array(f['entry_1/data_1/data'][:200])#[sl])

# We load all three reconstructions
full = cdtools.tools.data.h5_to_nested_dict(
    results_dir+f'Scan_{scan}_dis_{dist}_results.h5')

print(list(full))

# This defines the region of recovered object to use for the analysis.
pad = 400
window = np.s_[pad:-pad, pad:-pad]

# This brings all three reconstructions to a common basis, correcting for
# possible global phase offsets, position shifts, and phase ramps. It also
# calculates a Fourier ring correlation and spectral signal-to-noise ratio
# estimate from the two half reconstructions.
# results = cdtools.tools.analysis.standardize_reconstruction_set(
#     # half_1,
#     # half_2,
#     full,
#     window=window,
#     nbins=40, # The number of bins to use for the FRC calculation
# )

# We plot the normalized object images
p.plot_amplitude(full['obj'][window], basis=full['obj_basis'])
p.plot_phase(full['obj'][window], basis=full['obj_basis'])

p.plot_amplitude(full['probe'], basis=full['probe_basis'])
p.plot_phase(full['probe'], basis=full['probe_basis'])

# p.plot_real(full['probe'], basis=full['probe_basis'])
# p.plot_imag(full['probe'], basis=full['probe_basis'])

plt.show()
