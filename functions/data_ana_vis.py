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


def compute_angle(row, model_name):
    vector_1 = [row['gazed_x']-row['gaze_start_x'], row['gazed_y']-row['gaze_start_y']]
    vector_2 = [row['{}_est_x'.format(model_name)]-row['gaze_start_x'], row['{}_est_y'.format(model_name)]-row['gaze_start_y']]

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)*180/np.pi #angle in degrees
    return angle

def analyze_error(result_df, epoch, filename):
    try:
        result_df['chong_eucli_error'] = np.sqrt( (result_df['gazed_x']-result_df['chong_est_x'])**2 + (result_df['gazed_y']-result_df['chong_est_y'])**2 )
        result_df['chong_ang_error'] = result_df.apply(lambda row: compute_angle(row, 'chong'), axis=1)
    except:
        pass
    result_df['transformer_eucli_error'] = np.sqrt( (result_df['gazed_x']-result_df['transformer_est_x'])**2 + (result_df['gazed_y']-result_df['transformer_est_y'])**2 )
    result_df['transformer_ang_error'] = result_df.apply(lambda row: compute_angle(row, 'transformer'), axis=1)

    # plot histogram
    sns_setup(sns)
    try:
        ax = sns.histplot(data=result_df, x="chong_eucli_error", color=mycolors[0], binwidth=.1, stat='probability', label="Chong visual target")
        plt.axvline(result_df['chong_eucli_error'].median(), 0, 1, linewidth=4, c=mycolors[0])
    except:
        pass
    ax = sns.histplot(data=result_df, x="transformer_eucli_error",  color=mycolors[2], binwidth=.1, stat='probability', label="gaze transformer")
    plt.axvline(result_df['transformer_eucli_error'].median(), 0, 1, linewidth=4, c=mycolors[2])
    plt.legend()
    ax.set(xlabel='Euclidean Error', ylabel='Probability')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.figure.savefig("{}/errors_epoch{}.png".format(filename,epoch),  bbox_inches='tight')
    plt.close()

    # plot histogram
    sns_setup(sns)
    try:
        ax = sns.histplot(data=result_df, x="chong_ang_error", color=mycolors[0], binwidth=10, stat='probability', label="Chong visual target")
        plt.axvline(result_df['chong_ang_error'].median(), 0, 1, linewidth=4, c=mycolors[0])
    except:
        pass
    ax = sns.histplot(data=result_df, x="transformer_ang_error",  color=mycolors[2], binwidth=10, stat='probability', label="gaze transformer")
    plt.axvline(result_df['transformer_ang_error'].median(), 0, 1, linewidth=4, c=mycolors[2])
    plt.legend()
    ax.set(xlabel='Angular Error', ylabel='Probability')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.figure.savefig("{}/ang_errors_epoch{}.png".format(filename,epoch),  bbox_inches='tight')
    plt.close()
