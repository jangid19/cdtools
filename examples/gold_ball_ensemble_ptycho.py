from __future__ import division, print_function, absolute_import

import CDTools
import numpy as np
import pickle
from matplotlib import pyplot as plt

filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'

dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(filename)

results = []

for idx in range(25):
    print('Starting Reconstruction', idx)
    
    model = CDTools.models.FancyPtycho.from_dataset(dataset,n_modes=3,randomize_ang=0.1*np.pi)

    # default is CPU with 32-bit floats
    model.to(device='cuda')
    dataset.get_as(device='cuda')
    
    for i, loss in enumerate(model.Adam_optimize(30, dataset, batch_size=100)):
        print(i,loss)
        # Here we see how to liveplot the results - this call will create
        # or update a readout of the various parameters being reconstructed

    results.append(model.save_results(dataset))
    

with open('example_reconstructions/gold_balls_ensemble.pickle', 'wb') as f:
    pickle.dump(results,f)
    
model.inspect(dataset)
model.compare(dataset)
plt.show()
