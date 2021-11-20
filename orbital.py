import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  # Not redundant
import numpy as np
import scipy.special
import scipy.constants

# force matplotlib to show plot in separate window (for PyCharm)
matplotlib.use('Tkagg')
# get rid of toolbar at bottom of matplotlib window
matplotlib.rcParams['toolbar'] = 'None'


class Orbital:
    # bohr radius. In meters
    bohr_radius = scipy.constants.physical_constants['Bohr radius'][0]

    def __init__(self,
                 normalized_plot_range=100,
                 n_div=200,
                 n_points=100000,
                 plot_settings=None,
                 selection=3,
                 save=0,
                 force_range=0):

        # convert plot range to meters
        self.plot_range = normalized_plot_range * self.bohr_radius
        # 3D lattice has n_div ** 3 points
        self.n_div = n_div
        # number of points to sample from 3D lattice
        self.n_points = n_points
        # settings for 3D plot: [figsize, dpi, point_size]
        self.plot_settings = plot_settings if plot_settings is not None else [8, 100, 0.1]
        # Selection: 3 = plot both, 2 = plot orbital only, 1 = plot radial only, 0 = plot none
        self.selection = selection
        # Choose whether to save orbital image. 1 = save, 0 = do not save
        self.save = save
        # Choose whether to force axis limits to be plot_range. 1 = force, 0 = auto
        self.force_range = force_range

    # mainly square root part of wavefunction
    @staticmethod
    def normalization_constant_part(n, l, r):
        square_root = np.sqrt(
            (2 / (n * Orbital.bohr_radius)) ** 3 * np.math.factorial(n - l - 1) / (2 * n * np.math.factorial(n + l)))
        polynomial = ((2 * r) / (n * Orbital.bohr_radius)) ** l
        return square_root * polynomial

    # generalized laguerre polynomial and exponential
    @staticmethod
    def laguerre_exponential_part(n, l, r):
        polynomial = scipy.special.genlaguerre(n - l - 1, (2 * l) + 1)
        return polynomial((2 * r) / (n * Orbital.bohr_radius)) * np.exp(-r / (n * Orbital.bohr_radius))

    # radial part
    @staticmethod
    def radial_function(n, l, r):
        return Orbital.normalization_constant_part(n, l, r) * Orbital.laguerre_exponential_part(n, l, r)

    # wavefunction depends on 3 quantum numbers and spherical coordinates of a point
    @staticmethod
    def wavefunction(q_num, spherical):
        # get real part of spherical harmonic
        angular = scipy.special.sph_harm(q_num[2], q_num[1], spherical[2], spherical[1]).real
        # compute radial part of wavefunction
        radial = Orbital.radial_function(q_num[0], q_num[1], spherical[0])
        return radial * angular

    # get arctan as (0, 2pi] with correct quadrant
    @staticmethod
    def arctan_corrected(x, y):
        return np.arctan2(-y, -x) + np.pi

    # convert cartesian to spherical coordinates
    @staticmethod
    def spherical_coordinates(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = Orbital.arctan_corrected(z, np.sqrt(x ** 2 + y ** 2))  # from z axis down [0, pi]
        phi = Orbital.arctan_corrected(x, y)  # (0, 2pi]
        return [r, theta, phi]

    # generate the wavefunction value for a point of given cartesian coordinates
    @staticmethod
    def get_wavefunction(x, y, z, quantum_numbers):
        sph = Orbital.spherical_coordinates(x, y, z)
        return Orbital.wavefunction(quantum_numbers, sph)

    # sample from np.arange(len(probability_array)) using distribution defined by probability_array
    @staticmethod
    def sample_points(probability_array, sample_size):
        return np.random.choice(len(probability_array), sample_size, p=probability_array)

    # generate name for orbital using list of quantum numbers
    @staticmethod
    def get_orbital_name(q_nums):
        alphabet = '___fghijklmnoqrtuvwxyz'
        if q_nums[1] <= 2:
            letter = ['s', 'p', 'd'][q_nums[1]]
        else:
            letter = alphabet[q_nums[1]]
        return str(q_nums[0]) + str(letter) + ' Orbital (' + str(q_nums[2]) + ')'

    # generate list of quantum numbers
    @staticmethod
    def generate_quantum_numbers(max_n, max_l):
        quantum_numbers_list = []
        for i in range(max_n):
            for j in range(i + 1):
                for k in range(-j, j + 1):
                    if j <= max_l:
                        quantum_numbers_list.append([i + 1, j, k])
        return quantum_numbers_list

    # generate list of colours from numpy array
    @staticmethod
    def create_colour_list(array):
        return np.where(array <= 0, 'blue', 'red')

    # get sample of 3d points using probability distribution defined by wavefunction. Include colours
    def generate_3d_points(self, quantum_numbers):
        # generate cube of coordinates
        axis_steps = np.arange(-self.plot_range, self.plot_range, 2 * self.plot_range / self.n_div)
        xx, yy, zz = np.meshgrid(axis_steps, axis_steps, axis_steps)
        xx = xx.flatten()
        yy = yy.flatten()
        zz = zz.flatten()

        # get value of wavefunction for each point in cube
        wavefunction_values = self.get_wavefunction(xx, yy, zz, quantum_numbers)

        # generate list of probability densities
        density_list = np.square(wavefunction_values)

        # normalize so that sum = 1
        density_list /= density_list.sum()

        # sample from distribution defined by density_list to get an array of indices
        sampled_indices = self.sample_points(density_list, self.n_points)

        # get coordinates of points to plot
        x_plot = [xx[i] for i in sampled_indices]
        y_plot = [yy[i] for i in sampled_indices]
        z_plot = [zz[i] for i in sampled_indices]

        # generate list of colours for points
        colour_list = self.create_colour_list(np.array([wavefunction_values[i] for i in sampled_indices]))

        return np.array(x_plot), np.array(y_plot), np.array(z_plot), colour_list

    # plot graphs based on selection using set of provided quantum_numbers
    def plot_graphs(self, quantum_numbers):
        # attempt to generate a name for the graph
        try:
            figure_title = self.get_orbital_name(quantum_numbers)
        except IndexError:
            figure_title = 'An Unknown Orbital'

        # plot orbital in 3D
        if self.selection in [3, 2, 0]:
            # create figure using plot_settings
            plt.figure(figsize=(self.plot_settings[0], self.plot_settings[0]), dpi=self.plot_settings[1],
                       num=figure_title)
            self.plot_orbital(quantum_numbers, figure_title)

        # plot radial graphs
        if self.selection in [3, 1]:
            fig = plt.figure(figsize=(7, 7), num=' ' + figure_title + ' ')  # figures can't have same name
            self.plot_radial(quantum_numbers, fig)

        # show plots if necessary
        if self.selection in [3, 2, 1]:
            # show plots
            plt.show()

    # plot orbital in 3D
    def plot_orbital(self, quantum_numbers, figure_title):
        # sample coordinates and generate colour list
        ax = plt.axes(projection='3d')
        x_plot, y_plot, z_plot, colour_list = self.generate_3d_points(quantum_numbers)

        rx, ry, rz = [], [], []
        bx, by, bz = [], [], []

        for i, colour in enumerate(colour_list):
            if colour == 'red':
                rx.append(x_plot[i])
                ry.append(y_plot[i])
                rz.append(z_plot[i])
            else:
                bx.append(x_plot[i])
                by.append(y_plot[i])
                bz.append(z_plot[i])

        # plot points small and colour coded based on the sign of their value
        ax.plot(rx, ry, rz, marker='.', linestyle=' ', ms=self.plot_settings[2], c='red')
        ax.plot(bx, by, bz, marker='.', linestyle=' ', ms=self.plot_settings[2], c='blue')

        # set axis limits if force range is set to 1
        if self.force_range == 1:
            ax.set_xlim((-self.plot_range, self.plot_range))
            ax.set_ylim((-self.plot_range, self.plot_range))
            ax.set_zlim((-self.plot_range, self.plot_range))

        ax.margins(0)
        ax.set_axis_off()
        ax.set_facecolor([0, 0, 0])
        plt.tight_layout()

        # save figure if necessary
        if self.save == 1:
            plt.savefig(figure_title.replace(' ', '_') + '.png')

    # plot radial functions
    def plot_radial(self, quantum_numbers, fig):
        # create subplot for radial part
        ax = fig.add_subplot(211, xlabel='Bohr Radii', title='Radial Part')

        # take x values up to plot_range
        x_list = np.arange(0, self.plot_range, self.plot_range / 5000)
        y_list = Orbital.radial_function(quantum_numbers[0], quantum_numbers[1], x_list)

        # plot lines and show grid
        ax.plot(x_list / self.bohr_radius, y_list)
        plt.grid()

        # create subplot for RDF
        ax = fig.add_subplot(212, xlabel='Bohr Radii', title='Radial Distribution Function')

        # plot lines and show grid
        ax.plot(x_list / self.bohr_radius, y_list ** 2 * x_list ** 2)
        plt.grid()

        plt.tight_layout()


if __name__ == '__main__':
    orbital = Orbital(normalized_plot_range=20,
                      n_div=200,
                      n_points=100000,
                      plot_settings=None,
                      selection=2,
                      save=0,
                      force_range=0)
    orbital.plot_graphs([3, 2, 0])
