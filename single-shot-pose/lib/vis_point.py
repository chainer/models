from chainercv.visualizations.vis_image import vis_image


def vis_point(img, point, ax=None):
    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    for i in range(len(point)):
        for j in range(len(point[i])):
            ax.scatter(point[i][j][0], point[i][j][1])
    return ax
