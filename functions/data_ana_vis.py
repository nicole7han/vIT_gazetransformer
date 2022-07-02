import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# sys.path.append('/Users/nicolehan/Documents/Research/gazetransformer/')
# from train_model import *
import numpy as np

mycolors = sns.color_palette("RdBu", 10)
deeppallet = sns.color_palette("deep")

def sns_setup(sns):
    sns.set(rc={'figure.figsize':(12,8)})
    sns.set_context("paper", rc={"font.size":30,"axes.titlesize":40,"axes.labelsize":30,
                                 "legend.title_fontsize":40,"legend.fontsize":25,
                                 "xtick.labelsize":30, "ytick.labelsize":30,
                                 'legend.frameon': False})
    sns.set_style("white")
    sns.set_palette("deep")
    return

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)
        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

def compute_angle(row, model_name):
    img = plt.imread(row.image)
    try:
        h, w, _ = img.shape
    except:
        h, w = img.shape
    if model_name == 'centroid':
        vector_2 = [.5 - row['gaze_start_x'],.5 - row['gaze_start_y']]
    else:
        vector_2 = [row['{}_est_x'.format(model_name)] - row['gaze_start_x'],
                    row['{}_est_y'.format(model_name)] - row['gaze_start_y']]
    vector_1 = [row['gazed_x']-row['gaze_start_x'], row['gazed_y']-row['gaze_start_y']]
    vector_1, vector_2 = [vector_1[0] * w, vector_1[1] * h], [vector_2[0] * w, vector_2[1] * h]
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)*180/np.pi #angle in degrees
    return angle

def analyze_error(result_df, epoch, filename):

    # result_df['chong_eucli_error2'] = np.sqrt( (result_df['gazed_x']-result_df['chong_est_x'])**2 + (result_df['gazed_y']-result_df['chong_est_y'])**2 )
    # result_df['chong_ang_error2'] = result_df.apply(lambda row: compute_angle(row, 'chong'), axis=1)
    # result_df['transformer_eucli_error2'] = np.sqrt( (result_df['gazed_x']-result_df['transformer_est_x'])**2 + (result_df['gazed_y']-result_df['transformer_est_y'])**2 )
    result_df['centroid_ang_error'] = result_df.apply(lambda row: compute_angle(row, 'centroid'), axis=1)
    result_df['centroid_eucli_error'] = np.sqrt( (result_df['gazed_x']-.5)**2 + (result_df['gazed_y']-.5)**2 )

    # plot euclidean error histogram
    try:
        plot_data = result_df[['transformer_eucli_error','chong_eucli_error','centroid_eucli_error']]
        plot_data.columns = ['Transformer','CNN','Centroid']
    except:
        plot_data = result_df[['transformer_eucli_error','centroid_eucli_error']]
        plot_data.columns = ['Transformer','Centroid']
    plot_data = plot_data.melt()
    sns_setup(sns)
    ax = sns.histplot(
        data=plot_data, x='value', hue='variable', multiple='dodge',stat='probability',shrink=.8,
    )
    ax.set(xlabel='Euclidean Error (pixels)', ylabel='Probability')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.figure.savefig("{}/errors_epoch{}.png".format(filename,epoch),  bbox_inches='tight')
    plt.close()

    # plot euclidean error barplot
    sns_setup(sns)
    ax = sns.barplot(data=plot_data, x='variable', y='value', color=deeppallet[0])
    ax.set(xlabel='', ylabel='Euclidean Error (pixels)')
    change_width(ax, .4)
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.figure.savefig("{}/errors_epoch{}.png".format(filename,epoch),  bbox_inches='tight')
    plt.close()

    # plot angular error histogram
    try:
        plot_data = result_df[['transformer_ang_error','chong_ang_error','centroid_ang_error']]
        plot_data.columns = ['Transformer','CNN','Centroid']
    except:
        plot_data = result_df[['transformer_ang_error','centroid_ang_error']]
        plot_data.columns = ['Transformer','Centroid']
    plot_data = plot_data.melt()
    sns_setup(sns)
    ax = sns.histplot(
        data=plot_data, x='value', hue='variable', multiple='dodge',stat='probability',shrink=.8,
    )
    ax.set(xlabel='Angular Error (degree)', ylabel='Probability')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.figure.savefig("{}/ang_errors_epoch{}.png".format(filename,epoch),  bbox_inches='tight')
    plt.close()
