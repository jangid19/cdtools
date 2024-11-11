import cdtools
from matplotlib import pyplot as plt
# from scipy import io
import torch as t
# import numpy as np

probe_guess_file = '/home/rjangid/GitHub/cdtools_results/NIST17_results/Scan_196486_results.h5'

scan = 198907

prop_dist_list = [10, 12.5]

filename = '/home/rjangid/20240209_NNO_Wedged_Levitan/Processed_CXIs/%s_p.cxi' % str(scan)
results_dir = '/home/rjangid/GitHub/cdtools_results/NIST17_results/'

dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

for idx, dist in enumerate(prop_dist_list):
    print(f'Simulating propagation distance = {dist} um')
    # dataset.inspect()
    
    # FancyPtycho is the workhorse model
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        # probe_guess = probe['probe'], 
        # oversampling=2,
        translation_scale=0.25,
        n_modes=6, # Use 3 incoherently mixing probe modes
        propagation_distance=dist*10**-6,
    )

    # probe = cdtools.tools.data.h5_to_nested_dict(probe_guess_file)
    # model.probe.data = cdtools.tools.initializers.SHARP_style_probe(t.as_tensor(np.array(probe['probe'])) / model.probe_norm, propagation_distance=130e-6)
    # model.probe.data = probe / model.probe_norm

    # probe = cdtools.tools.initializers.SHARP_style_probe(dataset)
    # model.probe.data = probe / model.probe_norm

    device = 'cuda'
    model.to(device=device)
    dataset.get_as(device=device)

    # dataset.inspect()
    
    model.translation_offsets.data += 0.7 * t.randn_like(model.translation_offsets)

    # The learning rate parameter sets the alpha for Adam.
    # The beta parameters are (0.9, 0.999) by default
    # The batch size sets the minibatch size
    for loss in model.Adam_optimize(10, dataset, lr=0.03, batch_size=50):
        print(model.report())
        # Plotting is expensive, so we only do it every tenth epoch
        if model.epoch % 5 == 0:
            model.inspect(dataset)

    for loss in model.Adam_optimize(10, dataset, lr=0.025, batch_size=50):
        print(model.report())
        # Plotting is expensive, so we only do it every tenth epoch
        if model.epoch % 5 == 0:
            model.inspect(dataset)

    for loss in model.Adam_optimize(20, dataset, lr=0.02, batch_size=50):
        print(model.report())
        # Plotting is expensive, so we only do it every tenth epoch
        if model.epoch % 5 == 0:
            model.inspect(dataset)

    for loss in model.Adam_optimize(30, dataset, lr=0.01, batch_size=50):
        print(model.report())
        # Plotting is expensive, so we only do it every tenth epoch
        if model.epoch % 5 == 0:
            model.inspect(dataset)

    # It's common to chain several different reconstruction loops. Here, we
    # started with an aggressive refinement to find the probe, and now we
    # polish the reconstruction with a lower learning rate and larger minibatch
    for loss in model.Adam_optimize(60, dataset,  lr=0.005, batch_size=50):
        print(model.report())
        # Plotting is expensive, so we only do it every tenth epoch
        if model.epoch % 5 == 0:
            model.inspect(dataset)

    # This orthogonalizes the recovered probe modes
    model.tidy_probes()

    # This saves the final result
    model.save_to_h5(results_dir+f'Scan_{scan}_dis_{dist}_results.h5', dataset)
    plt.close('all')

# model.inspect(dataset)
# model.compare(dataset)

# plt.show()