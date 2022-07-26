import pandas as pd
import glob, os
from script.model import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statannot import add_stat_annotation
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
setpallet = sns.color_palette("Set2")


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
summaries = glob.glob('data/*summary*')
results = pd.DataFrame()
for f in summaries:
    data = pd.read_excel(f)
    data['train_cond'] = os.path.split(f)[-1].split('_')[0]
    results = results.append(data)
results['test_cond'] = results['test_cond'].astype('category')
results['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
results['model'] = results['model'].astype('category')
results['model'].cat.reorder_categories(['Humans', 'CNN', 'Transformer'], inplace=True)

''' human, cnn, transformer performance on intact, floating heads, headless bodies (transformer only trained with heads) '''
plot_data = results.copy()
plot_data = plot_data[plot_data['train_cond']=='Head']
plot_data['model'] = plot_data['model'].cat.rename_categories(['Humans', 'CNN', 'Head Gazetransformer'])
aov = pg.anova(dv='Euclidean_error', between=['model', 'test_cond'], data=plot_data,
             detailed=True)
print(aov)
postdoc =plot_data.pairwise_ttests(dv='Euclidean_error',
                                   between=['model', 'test_cond'],
                                   padjust='fdr_bh',
                                   parametric=True).round(3)
sig_results = postdoc[postdoc['p-corr']<0.05]
box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
             (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
             (('Head Gazetransformer','floating heads'),('Head Gazetransformer','intact')), (('Head Gazetransformer','floating heads'),('Head Gazetransformer','headless bodies'))]
ps = [0.001, 0.001,0.001, 0.001,0.001,0.001]

sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond')
ax.set(xlabel='', ylabel='Euclidean Error')#, title='Transformer Trained with Heads')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')

add_stat_annotation(ax, data=plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
ax.legend(title='test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.3, 1], )
ax.figure.savefig("figures/modelxtest_cond.png", dpi=300, bbox_inches='tight')
plt.close()