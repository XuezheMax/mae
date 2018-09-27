import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def show_images(images, cols=1, xlabels=None, ylabels=None, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    label_font_size = 14
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    xlabels_counter = 0
    ylabels_counter = 0
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        # a.set_title(title)
        if n // cols == 0:
            # the first row
            a.title(xlabels[xlabels_counter], fontsize=label_font_size)
            xlabels_counter += 1
        if n % cols == 0:
            a.set_ylabel(ylabels[ylabels_counter], fontsize=label_font_size)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.figlegend()
    plt.show()


if __name__ == "__main__":
    root_dir = "./figs/"
    n = 6
    file_names = [root_dir + str(i) + ".png" for i in range(n)] # row-wise
    # row_legends = [r'$\alpha=$', r'$\alpha=$', r'$\alpha=$']
    # col_legends = [r'$\alpha=$', r'$\alpha=$', r'$\alpha=$']

    row_legends = [r'$\alpha=$', r'$\alpha=$', r'$\alpha=$']
    col_legends = [r'$\alpha=$', r'$\alpha=$']

    assert len(row_legends) * len(col_legends) == n
    imgs = []
    for f in file_names:
        imgs.append(imread(f))

    show_images(imgs, cols=3, xlabels=row_legends, ylabels=col_legends)