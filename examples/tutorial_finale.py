from tutorial_basic_ptycho_dataset import BasicPtychoDataset
from tutorial_simple_ptycho import SimplePtycho
from h5py import File
from matplotlib import pyplot as plt

filename = '/home/rjangid/GitHub/cdtools/examples/example_data/lab_ptycho_data.cxi'
with File(filename, 'r') as f:
    dataset = BasicPtychoDataset.from_cxi(f)    

dataset.inspect()

model = SimplePtycho.from_dataset(dataset)

model.to(device='cuda')
dataset.get_as(device='cuda')

for loss in model.Adam_optimize(100, dataset):
    model.inspect(dataset)
    print(model.report())

model.inspect(dataset)
model.compare(dataset)
plt.show()
