import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
# from ggplot.colors.palettes import *
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

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
    scale = 4
    label_font_size = 14
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    xlabels_counter = 0
    ylabels_counter = 0
    for n, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        # a.set_title(title)
        if n // cols == 0:
            # the first row
            ax.set_title("123", fontsize=label_font_size)
            xlabels_counter += 1
        if n % cols == 0:
            ax.set_ylabel(ylabels[ylabels_counter], fontsize=label_font_size)
            ylabels_counter += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images / scale)
    # plt.show()
    fig.savefig('main.png')

    colors = ['orangered', 'goldenrod', 'olivedrab', 'mediumseagreen', 'forestgreen', 'dodgerblue', 'steelblue', 'mediumslateblue',
              'orchid', 'hotpink']
    # pal = color_palette(n_colors=10, name="hls")


def display_multi_images_compare(n_row, n_col, mae_image, vae_image, selects, save_path):
    # image: np.array
    # selects: list, row-wise
    assert n_row * n_col == len(selects)
    n_images = len(selects)
    scale = 6
    fig = plt.figure()
    grid = AxesGrid(fig, 111, nrows_ncols=(n_row, n_col), axes_pad=0.0, )

    print(mae_image.shape, vae_image.shape)
    patch_width = 34
    im_row = mae_image.shape[0] // patch_width
    im_col = mae_image.shape[1] // patch_width

    print(im_row, im_col)
    for idx, i in enumerate(selects):
        r = selects[idx] // im_col
        c = selects[idx] % im_col
        print(idx, r, c)
        if (idx + 1) % 3 == 0:
            im = grid[idx].imshow(vae_image[patch_width * r:patch_width * (r + 1), patch_width * c:patch_width * (c + 1), :])
        else:
            im = grid[idx].imshow(mae_image[patch_width * r:patch_width * (r + 1), patch_width * c:patch_width * (c + 1), :])

    plt.axis('off')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
        labelbottom=False)  # labels along the bottom edge are off
    # fig.tight_layout()
    print(fig.get_size_inches())
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images / scale)
    # plt.show()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def display_from_two_images(n_row, n_col, image1, image2, selects1, selects2, save_path):
    # image: np.array
    # selects: list, row-wise
    assert n_row * n_col == len(selects1) + len(selects2)
    n_images = len(selects1) + len(selects2)
    scale = 6
    fig = plt.figure()
    grid = AxesGrid(fig, 111, nrows_ncols=(n_row, n_col), axes_pad=0.0, )
    patch_width = 34

    print(image1.shape, image1.shape)
    im_row = image1.shape[0] // patch_width
    im_col = image1.shape[1] // patch_width

    print(im_row, im_col)
    for idx, i in enumerate(selects1 + selects2):
        if idx < len(selects1):
            r = selects1[idx] // im_col
            c = selects1[idx] % im_col
            im = grid[idx].imshow(image1[patch_width * r:patch_width * (r + 1), patch_width * c:patch_width * (c + 1), :])
        else:
            r = selects2[idx-len(selects1)] // im_col
            c = selects2[idx-len(selects1)] % im_col
            im = grid[idx].imshow(
                image2[patch_width * r:patch_width * (r + 1), patch_width * c:patch_width * (c + 1), :])

    plt.axis('off')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
        labelbottom=False)  # labels along the bottom edge are off

    print(fig.get_size_inches())
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images / scale)
    # plt.show()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def display_multi_images(n_row, n_col, image, selects, save_path):
    # image: np.array
    # selects: list, row-wise
    assert n_row * n_col == len(selects)
    n_images = len(selects)
    scale = 6
    fig = plt.figure()
    grid = AxesGrid(fig, 111, nrows_ncols=(n_row, n_col), axes_pad=0.0, )
    patch_width = 32

    print(image.shape, image.shape)
    im_row = image.shape[0] // patch_width
    im_col = image.shape[1] // patch_width

    print(im_row, im_col)
    for idx, i in enumerate(selects):
        r = selects[idx] // im_col
        c = selects[idx] % im_col
        im = grid[idx].imshow(image[patch_width * r:patch_width * (r + 1), patch_width * c:patch_width * (c + 1), :])

    plt.axis('off')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
        labelbottom=False)  # labels along the bottom edge are off

    print(fig.get_size_inches())
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images / scale)
    # plt.show()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def show_tsne_images(images, cols=1, xlabels=None, ylabels=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    """
    label_font_size = 18
    from matplotlib import rcParams
    rcParams['axes.titlepad'] = 20
    scale = 4

    rows = int(len(images) / cols)
    n_images = len(images)

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cnames = ['orangered', 'goldenrod', 'olivedrab', 'mediumseagreen', 'forestgreen', 'dodgerblue', 'steelblue', 'mediumslateblue',
              'orchid', 'hotpink']
    labels = [str(i) for i in range(0, 10)]
    custom_legends = [Line2D([0], [0], color=colors[cnames[idx]], lw=4, alpha=1.0) for idx, c in enumerate(cnames)]

    fig = plt.figure()
    grid = AxesGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.0, )
    xlabels_counter = 0
    ylabels_counter = 0
    for idx, image in enumerate(images):
        grid[idx].imshow(image)
        # a.set_title(title)
        if idx in [6, 7, 8]:
            # the first row
            # grid[idx].set_title(xlabels[xlabels_counter], fontsize=label_font_size)
            print(idx, cols)
            grid[idx].set_xlabel(xlabels[xlabels_counter], fontsize=label_font_size)
            xlabels_counter += 1
        if idx in [0, 3, 6]:
            print(idx, cols)
            grid[idx].set_ylabel(ylabels[ylabels_counter], fontsize=label_font_size, labelpad=30, rotation=0)
            ylabels_counter += 1
        grid[idx].xaxis.set_major_locator(plt.NullLocator())
        grid[idx].yaxis.set_major_locator(plt.NullLocator())
    # plt.axis('off')
    # plt.tick_params(
    #     axis='both',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     left=False,
    #     right=False,
    #     labeltop=False,
    #     labelleft=False,
    #     labelright=False,
    #     labelbottom=False)  # labels along the bottom edge are off
    # fig.legend(custom_legends, labels, loc='right', borderaxespad=0)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images / scale)
    # plt.show()
    fig.savefig('tsne_9.pdf', bbox_inches="tight")
    plt.close()

    # pal = color_palette(n_colors=10, name="hls")

def show_tsne_images_2(images, cols=1, xlabels=None, ylabels=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    """
    from matplotlib import rcParams
    rcParams['axes.titlepad'] = 20
    scale = 4
    label_font_size = 8
    rows = int(len(images) / cols)
    n_images = len(images)

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cnames = ['orangered', 'goldenrod', 'olivedrab', 'mediumseagreen', 'forestgreen', 'dodgerblue', 'steelblue', 'mediumslateblue',
              'orchid', 'hotpink']
    labels = [str(i) for i in range(0, 10)]
    custom_legends = [Line2D([0], [0], color=colors[cnames[idx]], lw=4, alpha=1.0) for idx, c in enumerate(cnames)]

    fig = plt.figure()
    grid = AxesGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.0, )
    xlabels_counter = 0
    ylabels_counter = 0
    for idx, image in enumerate(images):
        grid[idx].imshow(image)
        # a.set_title(title)
        if idx // cols == 0:
            # the first row
            # grid[idx].set_title(xlabels[xlabels_counter], fontsize=label_font_size)
            print(idx, cols)
            grid[idx].set_xlabel(xlabels[xlabels_counter], fontsize=label_font_size, labelpad=5)
            xlabels_counter += 1
        grid[idx].xaxis.set_major_locator(plt.NullLocator())
        grid[idx].yaxis.set_major_locator(plt.NullLocator())
        # if idx % cols == 0:
        #     grid[idx].set_ylabel(ylabels[ylabels_counter], fontsize=label_font_size, labelpad=10, rotation=0)
        #     ylabels_counter += 1
        # grid[idx].axis('off')
    # plt.axis('off')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
        labelbottom=False)  # labels along the bottom edge are off
    # fig.legend(custom_legends, labels, loc='right')
    fig.legend(custom_legends, labels, loc='center left', borderaxespad=7, prop={'size': 2.5})
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images / scale)
    # plt.show()
    fig.savefig('tsne_3.pdf', bbox_inches="tight")
    plt.close()

    # pal = color_palette(n_colors=10, name="hls")

if __name__ == "__main__":
    root_dir = "./pics/tsne/"
    # n = 9
    # offset = 4
    n = 12
    offset = 1
    file_names = [root_dir + "tSNE" + str(i+offset) + ".png" for i in range(n)] # row-wise
    # row_legends = [r'$\alpha=$', r'$\alpha=$', r'$\alpha=$']
    # col_legends = [r'$\alpha=$', r'$\alpha=$', r'$\alpha=$']


    row_legends = [r'$\eta=0.5$', r'$\eta=1.0$', r'$\eta=2.0$']
    col_legends = [r'$\gamma=0.1$', r'$\gamma=0.5$', r'$\gamma=1.0$']
    # assert len(row_legends) * len(col_legends) == n

    row_lds = ['ResNet VAE', 'VLAE 1', 'VLAE 2']
    imgs = []
    for f in file_names:
        imgs.append(imread(f))

    show_tsne_images(imgs[3:12], cols=3, xlabels=row_legends, ylabels=col_legends)
    show_tsne_images_2(imgs[:3], cols=3, xlabels=row_lds, ylabels=col_legends)

    # root_dir = "./pics/reconstruct/omniglot/"
    # save_dir = "./pics/reconstruct/"

    # data_set = "cifar10" #"mnist" #"omniglot"
    # dd = "sample" #"reconstruct" #
    # root_dir = "./pics/" + dd + "/" + data_set + "/"
    # save_dir = "./pics/" + dd + "/"
    #
    # # mae_im = imread(root_dir + "mae.png")[1:-1, 1:-1, :]
    # # vae_im = imread(root_dir + "vlae.png")[1:-1, 1:-1, :]
    #
    # n_row = 9
    # n_col = 9
    #
    # mae_im1 = imread(root_dir + "mae1.png")[1:-1, 1:-1, :]
    # mae_im2 = imread(root_dir + "mae2.png")[1:-1, 1:-1, :]
    # select1 = [2, 3, 4, 6, 7, 9, 10, 13, 17,
    #            30, 32, 35, 34, 43, 45, 46, 48,
    #            50, 53, 58, 60, 63, 62, 65, 67,
    #            69, 73, 98, 103, 111, 142, 141, 129, 148,
    #            157, 161, 162, 168, 177, 180, 192, 198, 202,
    #            241]
    # select2 = [4, 5, 17, 22, 23, 27, 29, 62, 60,
    #            56, 73, 74, 81, 85, 86, 255, 137, 111,
    #            139, 142, 145, 149, 153, 156, 160, 168, 125,
    #            180, 185, 256, 255, 209, 213, 235, 124, 107, 102]
    # select1 = [s-1 for s in select1]
    # select2 = [s-1 for s in select2]
    # print(len(select1), len(select2))
    # mae_save_to = save_dir + "mae_" + data_set + "_sample.png"
    # display_from_two_images(n_row, n_col, mae_im1, mae_im2, select1, select2, mae_save_to)

    # save_to = save_dir + data_set + "_com_recon.png"
    # selects = [1, 2, 2, 23, 24, 24, 31, 32, 32,
    #            33, 34, 34, 45, 46, 46, 49, 50, 50,
    #            53, 54, 54, 55, 56, 56, 61, 62, 62,
    #            79, 80, 80, 81, 82, 82, 89, 90, 90,
    #            91, 92, 92, 101, 102, 102, 119, 120, 120,
    #            133, 134, 134, 135, 136, 136, 143, 144, 144,
    #            145, 146, 146, 149, 150, 150, 151, 152, 152,
    #            161, 162, 162, 165, 166, 166, 167, 168, 168,
    #            175, 176, 176, 191, 192, 192, 215, 216, 216]
    # selects = [s - 1 for s in selects]
    # display_multi_images_compare(n_row, n_col, mae_im, vae_im, selects, save_to)

    # save_to = save_dir + data_set + "_com_recon.png"
    # selects = [3, 4, 4, 7, 8, 8, 15, 16, 16, 27, 28, 28, 47, 48, 48, 79, 80, 80, 95, 96, 96, 105, 106, 106,
    #            109, 110, 110, 131, 132, 132, 139, 140, 140, 143, 144, 144, 155, 156, 156,
    #            159, 160, 160, 167, 168, 168, 175, 176, 176, 191, 192, 192, 225, 226, 226,
    #            237, 238, 238, 255, 256, 256, 263, 264, 264, 287, 288, 288, 319, 320, 320,
    #            415, 416, 416, 431, 432, 432, 223, 224, 224, 503, 504, 504]
    # # selects = [7, 8, 8, 11, 12, 12, 15, 16, 16, 39, 40, 40, 47, 48, 48, 51, 52, 52, 63, 64, 64, 67, 68, 68,
    # #            91, 92, 92, 103, 104, 104, 135, 136, 136, 139, 140, 140, 143, 144, 144,
    # #            151, 152, 152, 163, 164, 164, 171, 172, 172, 175, 176, 176, 191, 192, 192,
    # #            195, 196, 196, 203, 204, 204, 211, 212, 212, 227, 228, 228, 239, 240, 240,
    # #            511, 512, 512, 419, 420, 420, 383, 384, 384, 395, 396, 396]
    # selects = [s - 1 for s in selects]
    # display_multi_images_compare(n_row, n_col, mae_im, vae_im, selects, save_to)

    # mae_save_to = save_dir + "mae_" + data_set + "_sample.png"
    # vae_save_to = save_dir + "vae_" + data_set + "_sample.png"
    # np.random.seed(10)
    # select_mae = [2, 4, 7, 10, 11, 15, 17,19, 21,
    #               23, 24, 39, 28, 29, 30, 38, 29, 40,
    #               41, 81, 86, 103, 105, 107, 109, 119, 121,
    #               123, 124, 126, 129, 400, 399, 397, 396, 394,
    #               166, 198, 156, 264, 377, 361, 363, 364, 365,
    #               366, 367, 368, 369, 370, 371, 372, 373, 374,
    #               375, 376, 320, 319, 321, 322, 325, 327, 328,
    #               329, 330, 332, 331, 333, 334, 335, 336, 337,
    #               221, 222, 223, 224, 225, 226, 227, 228, 229,
    #               230, 231, 232, 233, 234, 241, 242, 243, 244,
    #               245, 246]
    # select_mae = select_mae[:81]
    # select_mae = [s - 1 for s in select_mae]
    # # select_mae = np.random.permutation(range(400))[:81]
    # selects = np.random.permutation(range(400))[:81]
    # display_multi_images(n_row, n_col, mae_im, select_mae, mae_save_to)
    # display_multi_images(n_row, n_col, vae_im, selects, vae_save_to)