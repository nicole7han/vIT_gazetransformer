from random import choices
import numpy as np
import math


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, color="#61a4b2"):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    #    # Plot data on ax
    #    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
    #                     edgecolor='C0', fill=False, linewidth=1)

    patches = ax.bar(x=bins[:-1], height=radius, width=widths, linewidth=1,
                     edgecolor="white", color=color, align='edge')
    #    # Set angle label
    #    ax.set_xticks(np.array([0, 45, 90, 135, 180, -135, -90, -45])/180*np.pi)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def rotate(row, param='est'):
    """
    Rotate a point clockwise by a given angle around a given origin.

    positive for rotating in clockwise
    negative for rotating in counterclockwise

    The angle should be given in degrees.
    """
    angle = row['humanvec_ang2hor']
    # flip y coordinates before rotation because (0,0) is on top left
    ox, oy = row['gaze_start_x'], 1-row['gaze_start_y']  # origin
    px, py = row['{}_x'.format(param)], 1-row['{}_y'.format(param)]  # point to rotate

    # positive angle clockwise rotation in normal coordinates
    qx = ox + math.cos(-angle * np.pi / 180) * (px - ox) - math.sin(-angle * np.pi / 180) * (py - oy)
    qy = oy + math.sin(-angle * np.pi / 180) * (px - ox) + math.cos(-angle * np.pi / 180) * (py - oy)

    # flip y coordinates back with (0,0) on top left
    qy = 1-qy
    return qx, qy

def compute_ang2hori(row):
    # compute vector angle relative to horizontal line
    v1 = np.array([row['gazed_x'] - row['gaze_start_x'], row['gazed_y'] - row['gaze_start_y']])
    v2 = np.array([row['gazed_x'] - row['gaze_start_x'], 0])

    signed_angle = (math.atan2(v1[1], v1[0]) - math.atan2(v2[1], v2[0])) * 180 / np.pi  # in degrees
    # clockwise positive, counterclockwise negative
    return signed_angle


def angular_error(row, param='est'):
    v1 = np.array([row['gaze_start_x'] - row['gazed_x'], row['gaze_start_y'] - row['gazed_y']]) # base vector
    unit_vector_1 = v1 / np.linalg.norm(v1)
    v2 = np.array([row['gaze_start_x'] - row['{}_x'.format(param)], row['gaze_start_y'] - row['{}_y'.format(param)]])
    unit_vector_2 = v2 / np.linalg.norm(v2)

    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * 180 / np.pi
    return angle


def plot_vectors(image, plot_data, gazer=1 ,start='gaze_start', end='est'):
    # plot arrow vectors on images based on plot_data
    # gaze_start_xy -> est_xy
    model_orders = ['Humans', 'CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer']
    plot_data = plot_data[plot_data['gazer']==gazer].reset_index()
    colors = sns.color_palette("Set2")
    colors = colors[:4] + [colors[6]]
    H, W, _ = image.shape
    start_x, start_y = int(plot_data.iloc[0]['{}_x'.format(start)] * W), \
                     int(plot_data.iloc[0]['{}_y'.format(start)] * H)
    gazed_x, gazed_y = int(plot_data.iloc[0]['gazed_x'] * W), \
                       int(plot_data.iloc[0]['gazed_y'] * H)

    cv2.circle(image, (gazed_x, gazed_y), 20, [255, 255, 255], -1)
    plt.imshow(image)
    for i,model in enumerate(model_orders):
        row = plot_data[plot_data['model']==model]
        end_x, end_y = int(row['{}_x'.format(end)] * W), \
                       int(row['{}_y'.format(end)] * H)
        plt.arrow(start_x, start_y, end_x-start_x, end_y-start_y, color=colors[i],
                  head_width=30, head_length=50, label=model)
    plt.legend()
    plt.axis('off')




def cleanimagename(row):
    if 'nh' in row['image']:
        return row['image'].replace('_nh','')
    elif 'nb' in row['image']:
        return row['image'].replace('_nb','')
    else:
        return row['image']

def bootstrap(dataframe1, dataframe2, variable, boot_num, stats, within=None):
    vals1 = []
    vals2 = []
    data1 = dataframe1[variable]

    if type(dataframe2 )==int:  # one frame with one value
        data2 = dataframe2
        vals2 = data2
        for i in range(boot_num):
            try:
                temp1 = choices(data1, k=len(data1))
            except:
                temp1 = choices(data1.tolist(), k=len(data1))

            if stats == "mean":
                vals1.append(np.array(temp1).mean())
            elif stats == "var":
                vals1.append(np.array(temp1).var())
        ci2 = None

    else: # two dataframe
        data2 = dataframe2[variable]
        for i in range(boot_num):
            if i % 1000 == 0:
                print(i)
            if within!=None:  # randomly sample subjects to compare within-subject difference
                dataframe1 = dataframe1[[variable, within]]
                dataframe2 = dataframe2[[variable, within]]
                dataframe1['cond'] = 'df1'
                dataframe2['cond'] = 'df2'
                subj_diff = pd.concat([dataframe1, dataframe2])
                subj_dff = subj_diff.pivot(index=within, columns='cond', values=variable).reset_index()
                subj_dff['diff'] = subj_dff['df1' ] -subj_dff['df2']
                boot_subjs = np.random.choice(np.unique(dataframe1[within]).tolist(), len(np.unique(dataframe1.subj)))
                temp1 = []
                temp2 = []
                for s in boot_subjs:
                    temp1.append(subj_dff[subj_dff[within ]==s]['diff'].item())
                    temp2.append(0)
            else:
                try:
                    temp1 = choices(data1, k=len(data1))
                except:
                    temp1 = choices(data1.tolist(), k=len(data1))
                try:
                    temp2 = choices(data2, k=len(data2))
                except:
                    temp2 = choices(data2.tolist(), k=len(data2))

            if stats == "mean":
                vals1.append(np.array(temp1).mean())
                vals2.append(np.array(temp2).mean())
            elif stats == "var":
                vals1.append(np.array(temp1).var())
                vals2.append(np.array(temp2).var())
        ci2 = np.percentile(vals2, [2.5 ,97.5])  # 95% ci

    ci1 = np.percentile(vals1, [2.5 ,97.5])  # 95% ci

    try:
        p = 1 - len([1 for (i ,j) in zip(vals1, vals2) if i< j]) / boot_num
    except:
        p = 1 - len([1 for i in vals1 if i < data2]) / boot_num
    p = np.min([p, 1 - p])
    return ci1, ci2, p

