"""This module contains the base CDIModel class for CDI Models.

The subclasses of the main CDIModel class are required to define their
own implementations of the following functions:

Loading and Saving
------------------
from_dataset
    Creates a CDIModel from an appropriate CDataset
simulate_to_dataset
    Creates a CDataset from the simulation defined in the model
save_results
    Saves out a dictionary with the recovered parameters


Simulation
----------
interaction
    Simulates exit waves from experimental parameters
forward_propagator
    The propagator from the experiment plane to the detector plane
backward_propagator
    Optional, the propagator from the detector plane to the experiment plane
measurement
    Simulates the detector readout from a detector plane wavefront
loss
    the loss function to report and use for automatic differentiation

"""

import torch as t
from torch.utils import data as torchdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker
import numpy as np
import threading
import queue
import time
from .complex_adam import MyAdam
from .complex_lbfgs import MyLBFGS

__all__ = ['CDIModel']


class CDIModel(t.nn.Module):
    """This base model defines all the functions that must be exposed for a valid CDIModel subclass

    Most of the functions only raise a NotImplementedError at this level and
    must be explicitly defined by any subclass - these are noted explocitly
    in the module-level intro. The work of defining the various subclasses
    boils down to creating an appropriate implementation for this set of
    functions.
    """

    def __init__(self):
        super(CDIModel,self).__init__()

        self.loss_train = []
        self.iteration_count = 0

    def from_dataset(self, dataset):
        raise NotImplementedError()


    def interaction(self, *args):
        raise NotImplementedError()


    def forward_propagator(self, exit_wave):
        raise NotImplementedError()


    def backward_propagator(self, detector_wave):
        raise NotImplementedError()


    def measurement(self, detector_wave):
        raise NotImplementedError()


    def forward(self, *args):
        """The complete forward model

        This model relies on composing the interaction, forward propagator,
        and measurement functions which are required to be defined by all
        subclasses. It therefore should not be redefined by the subclasses.

        The arguments to this function, for any given subclass, will be
        the same as the arguments to the interaction function.
        """
        return self.measurement(self.forward_propagator(self.interaction(*args)))

    def loss(self, sim_data, real_data):
        raise NotImplementedError()

    
    def store_detector_geometry(self, detector_geometry):
        if 'distance' in detector_geometry:
            self.register_buffer('det_distance',
                                 t.as_tensor(detector_geometry['distance']))
        if 'basis' in detector_geometry:
            self.register_buffer('det_basis',
                                 t.as_tensor(detector_geometry['basis']))
        if 'corner' in detector_geometry:
            self.register_buffer('det_corner',
                                 t.as_tensor(detector_geometry['corner']))

    def get_detector_geometry(self):
        detector_geometry = {}
        if hasattr(self, 'det_distance'):
            detector_geometry['distance'] = self.det_distance
        if hasattr(self, 'det_basis'):
            detector_geometry['basis'] = self.det_basis
        if hasattr(self, 'det_corner'):
            detector_geometry['corner'] = self.det_corner
        return detector_geometry    
    
    def simulate_to_dataset(self, args_list):
        raise NotImplementedError()

    def save_results(self):
        """A convenience function to get the state dict as numpy arrays

        This function exists for two reasons, even though it is just a thin
        wrapper on top of t.module.state_dict(). First, because the model
        parameters for Automatic Differentiation ptychography and
        related CDI methods *are* the results, it's nice to explicitly
        recognize the role of extracting the state_dict as saving the
        results of the reconstruction

        Second, because display, further processing, long-term storage,
        etc. are often done with dictionaries of numpy files, it's useful
        to have a convenience function which does that conversion
        automatically.
        """
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}

    def AD_optimize(self, iterations, data_loader,  optimizer,\
                    scheduler=None, regularization_factor=None, thread=True,
                    calculation_width=10):
        """Runs a round of reconstruction using the provided optimizer

        This is the basic automatic differentiation reconstruction tool
        which all the other, algorithm-specific tools, use.

        Like all the other optimization routines, it is defined as a
        generator function which yields the average loss each epoch.

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        dataset : CDataset
            The dataset to reconstruct against
        optimizer : torch.optim.Optimizer
            The optimizer to run the reconstruction with
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Optional, a learning rate scheduler to use
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation
        calculation_width : int
            Default 10, how many translations to pass through at once for each round of gradient accumulation
        """
        # First, calculate the normalization
        normalization = 0
        for inputs, patterns in data_loader:
            normalization += t.sum(patterns).cpu().numpy()

        def run_iteration(stop_event=None):
            loss = 0
            N = 0
            t0 = time.time()
            for inputs, patterns in data_loader:
                N += 1
                def closure():
                    optimizer.zero_grad()

                    input_chunks = [[inp[i:i + calculation_width]
                                     for inp in inputs]
                                    for i in range(0, len(inputs[0]),
                                                   calculation_width)]
                    pattern_chunks = [patterns[i:i + calculation_width]
                                      for i in range(0, len(inputs[0]),
                                                     calculation_width)]

                    total_loss = 0
                    for inp, pats in zip(input_chunks, pattern_chunks):
                        # This is just used to allow graceful exit when
                        # threading
                        if stop_event is not None and stop_event.is_set():
                            exit()

                        sim_patterns = self.forward(*inp)

                        if hasattr(self, 'mask'):
                            loss = self.loss(pats,sim_patterns, mask=self.mask)
                        else:
                            loss = self.loss(pats,sim_patterns)

                        loss.backward()

                        total_loss += loss.detach()

                    if regularization_factor is not None \
                       and hasattr(self, 'regularizer'):
                        loss = self.regularizer(regularization_factor)
                        loss.backward()
                    return total_loss

                loss += optimizer.step(closure).detach().cpu().numpy()

            loss /= normalization
            if scheduler is not None:
                scheduler.step(loss)

            self.loss_train.append(loss)
            self.latest_iteration_time = time.time() - t0
            return loss

        if thread:
            result_queue = queue.Queue()
            stop_event = threading.Event()
            def target():
                try:
                    result_queue.put(run_iteration(stop_event))
                except Exception as e:
                    # If something bad happens, put the exception into the
                    # result queue
                    result_queue.put(e)

        for it in range(iterations):
            if thread:
                calc = threading.Thread(target=target, name='calculator', daemon=True)
                try:
                    calc.start()
                    while calc.is_alive():
                        if hasattr(self, 'figs'):
                            self.figs[0].canvas.start_event_loop(0.01)
                        else:
                            calc.join()

                except KeyboardInterrupt as e:
                    stop_event.set()
                    print('\nAsking execution thread to stop cleanly - please be patient.')
                    calc.join()
                    raise e

                res = result_queue.get()

                # If something went wrong in the thead, we'll get an exception
                if isinstance(res, Exception):
                    raise res

                yield res

            else:
                yield run_iteration()


    def Adam_optimize(self, iterations, dataset, batch_size=15, lr=0.005,
                      schedule=False, amsgrad=False, subset=None,
                      regularization_factor=None, thread=True,
                      calculation_width=10):
        """Runs a round of reconstruction using the Adam optimizer

        This is generally accepted to be the most robust algorithm for use
        with ptychography. Like all the other optimization routines,
        it is defined as a generator function, which yields the average
        loss each epoch.

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        dataset : CDataset
            The dataset to reconstruct against
        batch_size : int
            Optional, the size of the minibatches to use
        lr : float
            Optional, The learning rate (alpha) to use
        schedule : float
            Optional, whether to use the ReduceLROnPlateau scheduler
        subset : list(int) or int
            Optional, a pattern index or list of pattern indices to use
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation
        calculation_width : int
            Default 1, how many translations to pass through at once for each round of gradient accumulation

        """

        if subset is not None:
            # if just one pattern, turn into a list for convenience
            if type(subset) == type(1):
                subset = [subset]
            dataset = torchdata.Subset(dataset, subset)

        # Make a dataloader
        data_loader = torchdata.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True)

        # Define the optimizer
        #optimizer = t.optim.Adam(self.parameters(), lr = lr, amsgrad=amsgrad)
        optimizer = MyAdam(self.parameters(), lr = lr, amsgrad=amsgrad)


        # Define the scheduler
        if schedule:
            scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2,threshold=1e-9)
        else:
            scheduler = None

        return self.AD_optimize(iterations, data_loader, optimizer,
                                scheduler=scheduler,
                                regularization_factor=regularization_factor,
                                thread=thread,
                                calculation_width=calculation_width)


    def LBFGS_optimize(self, iterations, dataset,
                       lr=0.1,history_size=2, subset=None,
                       regularization_factor=None, thread=True,
                       calculation_width=10, line_search_fn=None):
        """Runs a round of reconstruction using the L-BFGS optimizer

        This algorithm is often less stable that Adam, however in certain
        situations or geometries it can be shockingly efficient. Like all
        the other optimization routines, it is defined as a generator
        function which yields the average loss each epoch.

        Note: There is no batch size, because it is a usually a bad idea to use
        LBFGS on anything but all the data at onece

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        dataset : CDataset
            The dataset to reconstruct against
        lr : float
            Optional, the learning rate to use
        history_size : int
            Optional, the length of the history to use.
        subset : list(int) or int
            Optional, a pattern index or list of pattern indices to ues
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation

        """
        if subset is not None:
            # if just one pattern, turn into a list for convenience
            if type(subset) == type(1):
                subset = [subset]
            dataset = torchdata.Subset(dataset, subset)

        # Make a dataloader. This basically does nothing but load all the
        # data at once
        data_loader = torchdata.DataLoader(dataset, batch_size=len(dataset))


        # Define the optimizer
        optimizer = t.optim.LBFGS(self.parameters(),
                                  lr = lr, history_size=history_size,
                                  line_search_fn=line_search_fn)
        #optimizer = MyLBFGS(self.parameters(),
        #                    lr = lr, history_size=history_size)

        return self.AD_optimize(iterations, data_loader, optimizer,
                                regularization_factor=regularization_factor,
                                thread=thread,
                                calculation_width=calculation_width)


    def SGD_optimize(self, iterations, dataset, batch_size=None,
                     lr=0.01, momentum=0, dampening=0, weight_decay=0,
                     nesterov=False, subset=None, regularization_factor=None,
                     thread=True, calculation_width=10):
        """Runs a round of reconstruction using the SGDoptimizer

        This algorithm is often less stable that Adam, but it is simpler
        and is the basic workhorse of gradience descent.

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        dataset : CDataset
            The dataset to reconstruct against
        batch_size : int
            Optional, the size of the minibatches to use
        lr : float
            Optional, the learning rate to use
        momentum : float
            Optional, the length of the history to use.
        subset : list(int) or int
            Optional, a pattern index or list of pattern indices to use
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation
        calculation_width : int
            Default 1, how many translations to pass through at once for each round of gradient accumulation

        """

        if subset is not None:
            # if just one pattern, turn into a list for convenience
            if type(subset) == type(1):
                subset = [subset]
            dataset = torchdata.Subset(dataset, subset)

        # Make a dataloader
        if batch_size is not None:
            data_loader = torchdata.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True)
        else:
            data_loader = torchdata.DataLoader(dataset)


        # Define the optimizer
        optimizer = t.optim.SGD(self.parameters(),
                                lr = lr, momentum=momentum,
                                dampening=dampening,
                                weight_decay=weight_decay,
                                nesterov=nesterov)

        return self.AD_optimize(iterations, data_loader, optimizer,
                                regularization_factor=regularization_factor,
                                thread=thread,
                                calculation_width=calculation_width)


    def report(self):
        """Returns a string informing on the latest reconstruction iteration

        Returns
        -------
        report : str
            A string with basic info on the latest iteration
        """
        if hasattr(self, 'latest_iteration_time'):
            return 'Iteration ' + str(len(self.loss_train)) + \
                  ' completed in %0.2f s with loss ' %\
                  self.latest_iteration_time + str(self.loss_train[-1])
        else:
            return 'No reconstruction iterations performed yet!'

    # By default, the plot_list is empty
    plot_list = []


    def inspect(self, dataset=None, update=True):
        """Plots all the plots defined in the model's plot_list attribute

        If update is set to True, it will update any previously plotted set
        of plots, if one exists, and then redraw them. Otherwise, it will
        plot a new set, and any subsequent updates will update the new set

        Optionally, a dataset can be passed, which then will plot any
        registered plots which need to incorporate some information from
        the dataset (such as geometry or a comparison with measured data).

        Plots can be registered in any subclass by defining the plot_list
        attribute. This should be a list of tuples in the following format:
        ( 'Plot Title', function_to_generate_plot(self),
        function_to_determine_whether_to_plot(self))

        Where the third element in the tuple (a function that returns
        True if the plot is relevant) is not required.

        Parameters
        ----------
        dataset : CDataset
            Optional, a dataset matched to the model type
        update : bool
            Default True, whether to update existing plots or plot new ones

        """

        #print('base models inspect: checking the object')
        #a = self.obj.detach()
        #def saveobj(a, filename):
        #    a = np.abs(a)
        #    plt.imshow(a)
        #    plt.savefig(filename)
        #f = ['base_a.png', 'base_b.png', 'base_c.png', 'base_d.png']
        #comp = [a[i, j, :, :] for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1])]
        #for i in range(4):
        #    saveobj(comp[i], f[i])
        
        first_update = False
        if update and hasattr(self, 'figs') and self.figs:
            figs = self.figs
        elif update:
            figs = None
            self.figs = []
            first_update = True
        else:
            figs = None
            self.figs = []

        idx = 0
        for plots in self.plot_list:
            # If a conditional is included in the plot
            try:
                if len(plots) >=3 and not plots[2](self):
                    continue
            except TypeError as e:
                if len(plots) >= 3 and not plots[2](self, dataset):
                    continue

            name = plots[0]
            plotter = plots[1]

            if figs is None:
                fig = plt.figure()
                self.figs.append(fig)
            else:
                fig = figs[idx]

            try:
                plotter(self,fig)
                plt.title(name)

            except TypeError as e:
                if dataset is not None:
                    try:
                        plotter(self, fig, dataset)
                        plt.title(name)

                    except (IndexError, KeyError, AttributeError, np.linalg.LinAlgError) as e:
                        pass

            except (IndexError, KeyError, AttributeError, np.linalg.LinAlgError) as e:
                pass

            idx += 1

            if update:
                plt.draw()
                fig.canvas.start_event_loop(0.001)

        if first_update:
            plt.pause(0.05 * len(self.figs))


    def save_figures(self, prefix='', extension='.eps'):
        """Saves all currently open inspection figures.

        Note that this function is not very intelligent - so, for example,
        if multiple probe modes are being reconstructed and the probe
        plotting function allows one to scroll between different modes, it
        will simply save whichever mode happens to be showing at the moment.
        Therefore, this should not be treated as a good way of saving out
        the full state of the reconstruction.

        By default, the files will be named by the figure titles as defined
        in the plot_list. Files can be saved with any extension suported by
        matplotlib.pyplot.savefig.

        Parameters
        ----------
        prefix : str
            Optional, a string to prepend to the saved figure names
        extention : strategy
            Default is .eps, the file extension to save with.
        """

        if hasattr(self, 'figs') and self.figs:
            figs = self.figs
        else:
            return # No figures to save

        for fig in self.figs:
            fig.savefig(prefix + fig.axes[0].get_title() + extension,
                        bbox_inches = 'tight')


    def compare(self, dataset, logarithmic=False):
        """Opens a tool for comparing simulated and measured diffraction patterns

        Parameters
        ----------
        dataset : CDataset
            A dataset containing the simulated diffraction patterns to compare against
        """

        fig, axes = plt.subplots(1,3,figsize=(12,5.3))
        fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.96])
        axslider = plt.axes([0.15,0.06,0.75,0.03])


        def update_colorbar(im):
            # If the update brought the colorbar out of whack
            # (say, from clicking back in the navbar)
            # Holy fuck this was annoying. Sorry future for how
            # crappy this solution is.
            #if not np.allclose(im.colorbar.ax.get_xlim(),
            #                   (np.min(im.get_array()),
            #                    np.max(im.get_array()))):
            if hasattr(im, 'norecurse') and im.norecurse:
                im.norecurse=False
                return

            im.norecurse=True
            im.set_clim(vmin=np.min(im.get_array()),vmax=np.max(im.get_array()))

        def update(idx):
            idx = int(idx) % len(dataset)
            fig.pattern_idx = idx
            updating = True if len(axes[0].images) >= 1 else False

            inputs, output = dataset[idx]
            sim_data = self.forward(*inputs).detach().cpu().numpy()
            meas_data = output.detach().cpu().numpy()
            if hasattr(self, 'mask') and self.mask is not None:
                mask = self.mask.detach().cpu().numpy()
            else:
                mask = 1

            if logarithmic:
                sim_data =np.log(sim_data)/np.log(10)
                meas_data = np.log(meas_data)/np.log(10)

            if not updating:
                axes[0].set_title('Simulated')
                axes[1].set_title('Measured')
                axes[2].set_title('Difference')

                sim = axes[0].imshow(sim_data)
                meas = axes[1].imshow(meas_data * mask)
                diff = axes[2].imshow((sim_data-meas_data) * mask)

                cb1 = plt.colorbar(sim, ax=axes[0], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.1,fraction=0.1)
                cb1.ax.tick_params(labelrotation=20)
                cb1.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(sim))
                cb2 = plt.colorbar(meas, ax=axes[1], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.1,fraction=0.1)
                cb2.ax.tick_params(labelrotation=20)
                cb2.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(meas))
                cb3 = plt.colorbar(diff, ax=axes[2], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.1,fraction=0.1)
                cb3.ax.tick_params(labelrotation=20)
                cb3.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(diff))

            else:

                sim = axes[0].images[-1]
                sim.set_data(sim_data)
                update_colorbar(sim)

                meas = axes[1].images[-1]
                meas.set_data(meas_data * mask)
                update_colorbar(meas)

                diff = axes[2].images[-1]
                diff.set_data((sim_data-meas_data) * mask)
                update_colorbar(diff)


        # This is dumb but the slider doesn't work unless a reference to it is
        # kept somewhere...
        self.slider = Slider(axslider, 'Pattern #', 0, len(dataset)-1, valstep=1, valfmt="%d")
        self.slider.on_changed(update)

        def on_action(event):
            if not hasattr(event, 'button'):
                event.button = None
            if not hasattr(event, 'key'):
                event.key = None

            if event.key == 'up' or event.button == 'up':
                update(fig.pattern_idx - 1)
            elif event.key == 'down' or event.button == 'down':
                update(fig.pattern_idx + 1)
            self.slider.set_val(fig.pattern_idx)
            plt.draw()

        fig.canvas.mpl_connect('key_press_event',on_action)
        fig.canvas.mpl_connect('scroll_event',on_action)
        update(0)