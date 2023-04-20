import monai
import numpy as np
import matplotlib.pyplot as plt


def plot_2D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot an x-y grid warped by this deformation.

    vector_field should be a tensor of shape (2,H,W)
        Note: vector_field spatial indices are swapped to match the conventions of imshow and quiver
    kwargs are passed to matplotlib plotting
    """
    # phi in the following line is the deformation mapping.
    # Note that we are swapping the spatial x and y when we evaluate vector_field;
    # the reason for this is that we want to match the the "matrix" or "image" style
    # conventions used by matplotlib imshow and quiver, where the axis used for "rows"
    # precedes the axis used for "columns"
    phi = lambda pt: pt + vector_field[:, pt[1], pt[0]].numpy()  # deformation mapping

    _, xmax, ymax = vector_field.shape
    xvals = np.arange(0, xmax, grid_spacing)
    yvals = np.arange(0, ymax, grid_spacing)
    for x in xvals:
        pts = [phi(np.array([x, y])) for y in yvals]
        pts = np.array(pts)
        plt.plot(pts[:, 0], pts[:, 1], **kwargs)
    for y in yvals:
        pts = [phi(np.array([x, y])) for x in xvals]
        pts = np.array(pts)
        plt.plot(pts[:, 0], pts[:, 1], **kwargs)


def preview_3D_deformation(vector_field, grid_spacing, figsize=(18, 6), is_show=False, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot warped grids along three orthogonal slices.

    vector_field should be a tensor of shape (3,H,W,D)
    kwargs are passed to matplotlib plotting

    Deformations are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """
    x, y, z = np.array(vector_field.shape[1:]) // 2  # half-way slices
    figure = plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_deformation(vector_field[[1, 2], x, :, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 2], :, y, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 1], :, :, z], grid_spacing, **kwargs)
    if is_show:
        plt.show()
    return figure


def plot_2D_vector_field(vector_field, downsampling):
    """Plot a 2D vector field given as a tensor of shape (2,H,W).

    The plot origin will be in the lower left.
    Using "x" and "y" for the rightward and upward directions respectively,
      the vector at location (x,y) in the plot image will have
      vector_field[1,y,x] as its x-component and
      vector_field[0,y,x] as its y-component.
    """
    downsample2D = monai.networks.layers.factories.Pool['AVG', 2](
        kernel_size=downsampling)
    vf_downsampled = downsample2D(vector_field.unsqueeze(0))[0]
    plt.quiver(
        vf_downsampled[1, :, :], vf_downsampled[0, :, :],
        angles='xy', scale_units='xy', scale=downsampling,
        headwidth=4.
    )


def preview_3D_vector_field(vector_field, figsize=(18, 6), is_show=False, downsampling=None):
    """
    Display three orthogonal slices of the given 3D vector field.

    vector_field should be a tensor of shape (3,H,W,D)

    Vectors are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """

    if downsampling is None:
        # guess a reasonable downsampling value to make a nice plot
        downsampling = max(1, int(max(vector_field.shape[1:])) >> 5)

    x, y, z = np.array(vector_field.shape[1:]) // 2  # half-way slices
    figure = plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[1, 2], x, :, :], downsampling)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 2], :, y, :], downsampling)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 1], :, :, z], downsampling)
    if is_show:
        plt.show()
    return figure



