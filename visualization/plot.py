import os
import matplotlib.pyplot as plt


def plot_contour_plot(x, y, z, save_path):
    """Function interpolates contour lines and then saves the
    figure of the contour plot.

    # Args
        x: x-axis coordinates of the values in Z.
        y: y-axis coordinates of the values in Z.
        z: The height values over which the contour is drawn.
        save_path: path where the plot figure will be saved.

    # Returns
        None.
    """
    plt.contourf(x, y, z, 100)
    plt.xlabel('R (nm)')
    plt.ylabel('H (nm)')
    plt.colorbar()
    plt.title('Contour plot (mean value)')
    plt.savefig(save_path)


if __name__ == '__main__':
    # Read X values (radius)
    x = []
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        'data',
                                        'parameter_space',
                                        'x_value.txt'))
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    for num in lines:
        x.append(int(num))

    # Read Y values (depth)
    y = []
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        'data',
                                        'parameter_space',
                                        'y_value.txt'))
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    for num in lines:
        y.append(int(num))

    # Read the mean intensity for the given X and Y
    z = []
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        'data',
                                        'parameter_space',
                                        'mean_matrix.txt'))
    f = open(path, 'r')
    for i in range(0, 10):
        new_row = []
        lines = f.readline().split(' ')
        for num in lines:
            new_row.append(float(num))
        z.append(new_row)
    f.close()

    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        'data',
                                        'parameter_space',
                                        'color_plot.png'))

    plot_contour_plot(x, y, z, save_path)
