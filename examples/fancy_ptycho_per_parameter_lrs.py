import cdtools
import torch as t
from matplotlib import pyplot as plt

filename = 'example_data/lab_ptycho_data.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

model = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=3, # Use 3 incoherently mixing probe modes
    oversampling=2, # Simulate the probe on a 2xlarger real-space array
    probe_support_radius=120, # Force the probe to 0 outside a radius of 120 pix
    propagation_distance=5e-3, # Propagate the initial probe guess by 5 mm
    units='mm', # Set the units for the live plots
    obj_view_crop=-50, # Expands the field of view in the object plot by 50 pix
)

if t.cuda.is_available():
    model.to(device='cuda')
    dataset.get_as(device='cuda')

# Here, we tune the learning rates of individual parameters. The default
# learning rate factor is 1. Any learning rate factor set here will multiply
# the learning rate for each recon.optimize loop. The dictionary can be passed
# to the reconstructor object at creation time, as done here. It can also be
# updated later with the call to recon.optimize(..., lr_factors=lr_factors).
lr_factors = {
    'translation_offsets' : 1.2,
    'weights' : 0.2,
    'background' : 0.3,
}

recon = cdtools.reconstructors.AdamReconstructor(
    model, dataset, lr_factors=lr_factors)

# For example, background will get a lr of 0.03 * 0.3 (lr * lr_factor).
for loss in recon.optimize(50, lr=0.03, batch_size=10, lr_factors=lr_factors, verbose=True):
    print(model.report())
    model.inspect(min_interval=10)

# And here background will get a lr of 0.005 * 0.3 (lr * lr_factor).
for loss in recon.optimize(50, lr=0.005, batch_size=50):
    print(model.report())
    model.inspect(min_interval=10)

model.tidy_probes()

model.inspect(replot_all=True)
model.compare(dataset)
plt.show()
