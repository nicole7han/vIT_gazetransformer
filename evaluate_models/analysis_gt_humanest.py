import pandas as pd
import glob, os
from script.model import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from scipy import stats
from statannot import add_stat_annotation
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
setpallet = sns.color_palette("Set2")
custom_colors = sns.color_palette("Set1", 10)


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
summaries = glob.glob('data/GroundTruth_humanest/*summary*')
results = pd.DataFrame()
for f in summaries:
    data = pd.read_excel(f)
    results = results.append(data)
results['test_cond'] = results['test_cond'].astype('category')
results['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
results['model'] = results['model'].astype('category')
results['model'].cat.reorder_categories(['Humans', 'CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer'], inplace=True)


''' Human, CNN, 3 Transformer performance (human estimates as groundtruth) '''
plot_data = results[results['test_cond']=='intact']
plot_data = plot_data[[ 'Euclidean_error_meanest', 'Angular_error_meanest',
       'Euclidean_error_lou', 'Angular_error_lou','model']]

error = 'Euclidean_error_meanest'
# aov_data = plot_data[[error, 'model']].melt(id_vars='model')
# aov = pg.anova(dv='value', between=['model'], data=aov_data,
#              detailed=True)
# print(aov)
# postdoc =aov_data.pairwise_ttests(dv='value',
#                                    between=['model'],
#                                    padjust='fdr_bh',
#                                    parametric=True).round(3)
# sig_results = postdoc[postdoc['p-corr']<0.05]
# box_pairs = []
# ps = []
# for _, row in sig_results.iterrows():
#     box_pairs.append((row['A'],row['B']))
#     ps.append(max(0.001, row['p-corr']))

sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = error, y = 'model',color=setpallet[0])
ax.set(xlabel='Euclidean Error', ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.figure.savefig("figures/intact_gt_humanest{}.png".format(error), dpi=300, bbox_inches='tight')
plt.close()



error = 'Euclidean_error_lou'
sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = error, y = 'model', color=setpallet[0])
ax.set(xlabel='Euclidean Error', ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
# add_stat_annotation(ax, data=plot_data, x = 'model', y = error,
#                     box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
#                     loc='outside', verbose=2)
# ax.legend(title='test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
ax.figure.savefig("figures/intact_gt_humanest{}.png".format(error), dpi=300, bbox_inches='tight')
plt.close()


sns_setup_small(sns, (8,6))
error = 'Angular_error_meanest'
ax = sns.barplot(data = plot_data, x = error, y = 'model',color=setpallet[0])
ax.set(xlabel='Angular Error (˚)', ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.figure.savefig("figures/intact_gt_humanest{}.png".format(error), dpi=300, bbox_inches='tight')
plt.close()


sns_setup_small(sns, (8,6))
error = 'Angular_error_lou'
ax = sns.barplot(data = plot_data, x = error, y = 'model',color=setpallet[0])
ax.set(xlabel='Angular Error (˚)', ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.figure.savefig("figures/intact_gt_humanest{}.png".format(error), dpi=300, bbox_inches='tight')
plt.close()




''' Human-Human, Human-CNN, Human-Transformer Correlation '''

