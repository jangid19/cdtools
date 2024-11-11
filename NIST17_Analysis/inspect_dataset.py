import cdtools
from matplotlib import pyplot as plt

scan = 198907

# First, we load an example dataset from a .cxi file
filename = '/home/rjangid/20240209_NNO_Wedged_Levitan/Processed_CXIs/%s_p.cxi' % str(scan)
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# And we take a look at the data
dataset.inspect()
plt.show()
