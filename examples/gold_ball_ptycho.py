import cdtools
from matplotlib import pyplot as plt
from scipy import io

# First, we load an example dataset from a .cxi file
filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# Next, we create a ptychography model from the dataset
# Note that we explicitly ask for two incoherent probe modes
model = cdtools.models.FancyPtycho.from_dataset(dataset, n_modes=2)

# Let's do this reconstruction on the GPU, shall we? 
model.to(device='cuda')
dataset.get_as(device='cuda')

# Now, we run a short reconstruction from the dataset
for loss in model.Adam_optimize(10, dataset, batch_size=50, schedule=True):
    # And we liveplot the updates to the model as they happen
    print(model.report())
    model.inspect(dataset)

# This orthogonalizes the incoherent probe modes
model.tidy_probes()

# And we save out the results as a .mat file
io.savemat('example_reconstructions/gold_balls.mat',
           model.save_results(dataset))

# Finally, we plot the results
model.inspect(dataset)
model.compare(dataset)
plt.show()
