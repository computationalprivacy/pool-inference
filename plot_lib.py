import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import pickle
import pandas as pd
from sklearn.metrics import auc
from matplotlib.colors import SymLogNorm

cm2in = 1 / 2.54
mm2in = cm2in / 10
FIGSIZES = {
    'ieee': (78 * mm2in, 78 * mm2in),
    'ieee_small': (60 * mm2in, 60 * mm2in),
    'ieee_large': (100 * mm2in, 100 * mm2in),
    'ieee_double': (150 * mm2in, 50 * mm2in),
    'ieee_triple': (150 * mm2in, 50 * mm2in),
    'ieee_quadruple': (160 * mm2in, 50 * mm2in)
}
DEFAULT_FIGURE_TYPE = 'ieee'
palette = plt.cm.jet(np.linspace(0, 1, 11))
colors = [matplotlib.colors.to_hex(color, keep_alpha=True)
          for color in palette]
colors_to_use = {
    1: [colors[0]],
    # colors to use if there are 4 lines to be drawn
    4: [colors[0], colors[3], colors[8], colors[10]],
    # colors to use if there are 5 lines to be drawn
    5: [colors[0], colors[3], colors[8], colors[10], colors[6]]
}
markers = ['x', '*', '^', '.']


def str_eps(eps):
    return '\\infty' if eps == math.inf else str(eps)


def init_plotting(default_figure_type, latex=True):
    # matplotlib.rcParams['font.family'] = prop.get_name()
    matplotlib.rcParams['font.size'] = 14
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} '
    r'\usepackage[libertine]{newtxmath} \usepackage{sfmath}'
    matplotlib.rcParams['figure.dpi'] = 300

    plt.rc('figure', figsize=FIGSIZES[default_figure_type], dpi=150)
    plt.rc('savefig', dpi=300)

    plt.rc('font', weight='regular', size=8, family='serif')
    plt.rc('axes', titlesize=9)
    plt.rcParams['svg.fonttype'] = 'none'

    # Axes and ticks
    plt.rc('axes', linewidth=.5, grid=True, titlesize=8, labelpad=2)
    plt.rc('xtick', labelsize=5, direction='out')
    plt.rc('ytick', labelsize=5, direction='out')

    plt.rc('xtick.major', width=.5, pad=2.5)
    plt.rc('ytick.major', width=.5, pad=2.5)
    plt.rc('xtick.minor', width=.2, pad=2.5)
    plt.rc('ytick.minor', width=.2, pad=2.5)

    if latex is True:
        plt.rc('text', usetex=True)
        # plt.rcParams['text.latex.unicode'] = True
        matplotlib.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} " \
            r"\usepackage{amssymb} \usepackage[T1]{fontenc} " \
            r"\usepackage{mathptmx}"

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + \
        plt.rcParams['font.serif']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.pad_inches'] = 0


def init_figure(nrows=1, ncols=1, figure_type=DEFAULT_FIGURE_TYPE):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=FIGSIZES[figure_type])
    # grid
    if nrows == 1 and ncols == 1:
        ax.grid(color='#DCDCDC', linestyle='-', linewidth=1)
    else:
        for curr_ax in ax.flat:
            curr_ax.grid(color='#DCDCDC', linestyle='-', linewidth=1)

    return fig, ax


def parse_pickle_prec_nr(exp_pickle_file=None,
                         num_users=None, n_bins=101, results=None):
    """
    Parse experiment pickle to extract precision at each null rate

    Parameters
    ----------
    exp_pickle_file : string
        path to experiment pickle file (.../exp_pickles/{unique_id}.pickle)
    num_users : int | None
        focus only on the first num_users for faster analysis
    n_bins : int
        number of bins to use for null rates
    results : list[(int, list[float], float, float]
        instead of supplying a results file, the results themselves can be
        passed to get the auc

    Returns
    -------
    null_rates : list[float]
        list of "valid" null rates (null rates where the number of obs > 0)
    accuracies : list[float]
        list of accuracies for each "valid" null rate
    auc : float
        the area under the curve
    """
    if exp_pickle_file is not None:
        results = pickle.load(open(exp_pickle_file, "rb"))

    if num_users is not None:
        results = results[:num_users]

    num_outcomes = len(results)
    # normalize scores and choose maximum score
    if len(results[0][1]) == 4:
        # old format
        results = [(max(scores) / np.sum(scores), success)
                   for (success, (scores, _, _, _)) in results
                   if np.sum(scores) != 0]
    else:
        # new format
        results = [(max(scores) / np.sum(scores), success)
                   for (success, scores, _, _) in results
                   if np.sum(scores) != 0]

    # sort descending and convert to np array
    results.sort(reverse=True)
    results = np.array(results)

    # define null rates
    null_rates = np.linspace(0, 1, n_bins)
    accuracies = []
    valid_nrs = []

    for null_rate in null_rates:
        if null_rate == 1:
            # null rate = 1 is an edge case, ensure continuity of graph
            accuracy = accuracies[-1]
            valid_nrs.append(null_rate)
            accuracies.append(accuracy)
        else:
            num_outcomes = int((1 - null_rate) * len(results))
            if num_outcomes == 0:
                # "invalid" null rate
                continue
            outcomes_of_interest = results[:num_outcomes, 1]
            accuracy = np.sum(outcomes_of_interest) / len(outcomes_of_interest)
            valid_nrs.append(null_rate)
            accuracies.append(accuracy)

    return valid_nrs, accuracies, auc(valid_nrs, accuracies)


def parse_pickle_conf(exp_pickle_file, num_users=None, n_bins=12):
    """
    Parse experiment pickle to extract accuracy at each confidence threshold

    Parameters
    ----------
    exp_pickle_file : string
        path to experiment pickle file (.../exp_pickles/{unique_id}.pickle)
    num_users : int | None
        focus only on the first num_users for faster analysis
    n_bins : int
        number of bins to use for thresholds

    Returns
    -------
    threshs : list[float]
        list of "valid" thresholds (thresholds where the number of obs > 0)
    accuracies : list[float]
        list of accuracies for each "valid" threshold
    """
    results = pickle.load(open(exp_pickle_file, "rb"))
    if num_users is not None:
        results = results[:num_users]

    # normalize scores and choose maximum score
    if len(results[0][1]) == 4:
        # old format
        results = [(max(scores) / np.sum(scores), success)
                   for (success, (scores, _, _, _)) in results
                   if np.sum(scores) != 0]
    else:
        # new format
        results = [(max(scores) / np.sum(scores), success)
                   for (success, scores, _, _) in results
                   if np.sum(scores) != 0]
    # convert to np array
    results = np.array(results)

    thresholds = np.linspace(-0.05, 1.05, n_bins)
    valid_threshs = []
    accs = []
    for (i, curr_thresh) in enumerate(thresholds[1:]):
        prev_thresh = thresholds[i]
        query = (results[:, 0] >= prev_thresh) & (results[:, 0] < curr_thresh)
        outcomes_of_interest = results[query][:, 1]
        num_interest = len(outcomes_of_interest)
        if (num_interest > 0):
            valid_threshs.append((curr_thresh + prev_thresh) / 2)
            accs.append(np.sum(outcomes_of_interest) / num_interest)

    return valid_threshs, accs


def parse_pickle_gammadelta(exp_pickle_file, gammas, deltas):
    """
    Parse experiment pickle to extract accuracy for each gamma, delta

    Parameters
    ----------
    exp_pickle_file : string
        path to experiment pickle file (.../exp_pickles/{unique_id}.pickle)
    gammas : list[float]
        list of gammas indicating bins
    deltas : list[float]
        list of deltas indicating bins

    Returns
    -------
    Z : list[list[float]]
        3D data that can be plotted as a contour plot
    gammas : list[float]
        list of gammas that are at the center of the bins
    deltas : list[float]
        list of deltas that are at the center of the bins
    """
    PLOT_DB = pd.DataFrame(columns=['gamma', 'delta', 'accuracy'])

    # fill python dictionary with appropriate empty lists
    Z = dict()
    for i in range(len(gammas) - 1):
        for j in range(len(deltas) - 1):
            curr_gamma = (gammas[i] + gammas[i + 1]) / 2
            curr_delta = (deltas[j] + deltas[j + 1]) / 2
            if curr_gamma not in Z.keys():
                Z[curr_gamma] = dict()

            Z[curr_gamma][curr_delta] = []

    # utility function to find nearest gamma, delta value
    def find_nearest(gamma, delta):
        nearest_gamma, nearest_delta = -1, -1
        for i in range(len(gammas) - 1):
            for j in range(len(deltas) - 1):
                curr_gamma = (gammas[i] + gammas[i + 1]) / 2
                curr_delta = (deltas[j] + deltas[j + 1]) / 2
                if gammas[i] <= gamma <= gammas[i + 1]:
                    nearest_gamma = curr_gamma

                if deltas[j] <= delta <= deltas[j + 1]:
                    nearest_delta = curr_delta
        return nearest_gamma, nearest_delta

    # extract accuracies
    results = pickle.load(open(exp_pickle_file, 'rb'))
    for (success, _, gamma, delta) in results:
        nearest_gamma, nearest_delta = find_nearest(gamma, delta)
        Z[nearest_gamma][nearest_delta].append(success)

    # transfer data from python dictionary to df
    for (i, gamma) in enumerate(Z.keys()):
        for (j, delta) in enumerate(Z[gamma].keys()):
            curr_acc = np.mean(Z[gamma][delta])
            PLOT_DB = PLOT_DB.append(
                {'gamma': gamma, 'delta': delta, 'accuracy': curr_acc},
                ignore_index=True)

    # convert df into 2D array
    gammas = sorted(PLOT_DB['gamma'].unique())
    deltas = PLOT_DB['delta'].unique()
    deltas.sort()

    Z_arr = []
    for gamma in gammas:
        n_obs = []
        for delta in deltas:
            query = (PLOT_DB['gamma'] == gamma) & (PLOT_DB['delta'] == delta)
            if len(PLOT_DB[query]) > 0:
                curr_obs = PLOT_DB[query]['accuracy'].iloc[0]
            else:
                curr_obs = -1
            n_obs.append(curr_obs)
        Z_arr.append(n_obs)

    return Z_arr, gammas, deltas


def plot_prec_nr(exp, ax, n_pools=5, include_private=True, ns=None,
                 include_baseline=True, nonprivate=False, AUC_DB=None,
                 exps_folder='pickles/experiments/', num_users=None):
    """
    Plots the precision vs null rate curve
    (optionally update an existing AUC_DB)

    Returns
    -------
    aucs : DataFrame
        list of aucs for each setting packed into a dataframe
    """
    folder = f'{exps_folder}/{exp}'
    EXP_DB = pickle.load(open(f'{folder}/EXP_DB.pickle', 'rb'))

    if ns is not None:
        EXP_DB = EXP_DB[EXP_DB['n'].isin(ns)]
    ns = EXP_DB['n'].unique()

    # (optional) plot baseline at 1 / number of pools (guessing pools randomly)
    if include_baseline:
        ax.plot([0.0, 1.0], [1 / n_pools, 1 / n_pools],
                color='black', linestyle='--', label='baseline', linewidth=1)

    # for calculating AUCs
    if AUC_DB is None:
        column_names = ['Setting', 'Prior', 'Private', 'n', 'AUC']
        AUC_DB = pd.DataFrame(columns=column_names)
    setting = 'News' if 'news' in exp else 'Emojis'
    prior = 'Estimated' if 'estimated' in exp else 'Uniform'

    linestyle = '-.' if nonprivate else '-'
    private_str = 'Non-Private' if nonprivate else 'Private'
    for (i, n) in enumerate(ns):
        # parse pickle file
        unique_id = EXP_DB[EXP_DB['n'] == n].iloc[0]['unique_id']
        exp_pickle_file = f'{folder}/exp_pickles/{unique_id}.pickle'
        valid_nrs, accuracies, curr_auc = \
            parse_pickle_prec_nr(exp_pickle_file, num_users=num_users)

        # plot result
        color = colors_to_use[len(ns)][i]
        ax.plot(valid_nrs, accuracies, label=f'$n={n}$',
                color=color, linestyle=linestyle, linewidth=1)

        # add to AUC_DB
        AUC_DB = AUC_DB.append({'Setting': setting, 'Prior': prior,
                                'Private': private_str, 'n': n,
                                'AUC': curr_auc}, ignore_index=True)

    ax.set_ylabel('Precision')
    ax.set_xlabel('Null rate')

    # square plot
    ax.set_aspect('equal')
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlim([0, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylim([0, 1])

    return AUC_DB


def plot_conf(exp, ax, ns=None, exps_folder='pickles/experiments/',
              num_users=None):
    """
    Plots the confidence score plot
    """
    folder = f'{exps_folder}/{exp}/'
    EXP_ORIG_DB = pickle.load(open(f'{folder}/EXP_DB.pickle', 'rb'))
    EXP_DB = EXP_ORIG_DB[(EXP_ORIG_DB['k'] != 0)]

    if ns is not None:
        EXP_DB = EXP_DB[EXP_DB['n'].isin(ns)]
    ns = EXP_DB['n'].unique()

    for (i, n) in enumerate(ns):
        # parse pickle file
        unique_id = EXP_DB[EXP_DB['n'] == n].iloc[0]['unique_id']
        exp_pickle_file = f'{folder}/exp_pickles/{unique_id}.pickle'
        valid_threshs, accs = \
            parse_pickle_conf(exp_pickle_file, num_users=num_users)

        # plot result
        color = colors_to_use[len(ns)][i]
        marker = markers[i]
        ax.plot(valid_threshs, accs, label=f'$n={n}$', color=color,
                marker=marker, linewidth=1)

        xs = np.linspace(0, 1, 10)
        ys = xs
        ax.plot(xs, ys, color='lightgray', linestyle='--', linewidth=1)

        ax.grid(color='#DCDCDC', linestyle='-', linewidth=1)
        ax.set_ylim(ymin=0, ymax=1)
        ax.set_xlim(xmin=0, xmax=1)

    ax.set_ylabel('Success rate')
    ax.set_xlabel('Confidence score')

    ax.set_aspect('equal')
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlim([0, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylim([0, 1])


def plot_gammadelta(exp, axs, num_pools, ns=None,
                    exps_folder='pickles/experiments/'):
    """
    Plots the contour plot showing the difference in attack accuracy wrt
    gamma and delta
    """
    folder = f'{exps_folder}/{exp}/'
    EXP_ORIG_DB = pickle.load(open(f'{folder}/EXP_DB.pickle', 'rb'))
    EXP_DB = EXP_ORIG_DB[(EXP_ORIG_DB['k'] != 0)]

    if ns is not None:
        EXP_DB = EXP_DB[EXP_DB['n'].isin(ns)]
    ns = EXP_DB['n'].unique()

    # define gammas and deltas of interest
    gammas = [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    deltas = [1 / num_pools + gamma * (1 - 1 / num_pools) for gamma in gammas]

    for (i, (n, ax)) in enumerate(zip(ns, axs)):
        # parse pickle file
        unique_id = EXP_DB[EXP_DB['n'] == n].iloc[0]['unique_id']
        exp_pickle_file = f'{folder}/exp_pickles/{unique_id}.pickle'
        Z, Z_gammas, Z_deltas = parse_pickle_gammadelta(
            exp_pickle_file, gammas, deltas)

        # plot result
        levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        cmap = plt.cm.get_cmap('autumn', len(levels)).reversed()
        img = ax.contour(Z_deltas, Z_gammas, Z, levels=levels, cmap=cmap,
                         min=0.0, vmax=1.0, extent=[Z_deltas[0], Z_deltas[-1],
                                                    Z_gammas[0], Z_gammas[-1]],
                         extend='neither', linewidths=1)
        ax.clabel(img, inline=True, fontsize=6, fmt=lambda x: f'{x:.1f}')

        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(xmin=Z_deltas[0], xmax=Z_deltas[-1])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(ymin=0, ymax=1)
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

        ax.set_xlabel('Polarization ($\\delta_{\\text{Usr}}$)')
        ax.set_ylabel('Relevant interest\n($\\gamma_{\\text{Usr}}$)')

        ax.grid(color='#DCDCDC', linestyle='-', linewidth=1)
        ax.set_title(f'$n = {n}$')


def plot_gammadelta_heatmap(results_file, ax, plot='acc'):
    """
    Plots heatmap of attack accuracy wrt to gamma delta (plot = 'acc')
    or distribution of gamma delta (plot = 'dist')
    """
    gammas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    deltas = [0.2, 0.4, 0.6, 0.8, 1.0]

    results = pickle.load(open(results_file, 'rb'))
    total_res = len(results)

    def get_acc(gamma_low, delta_low, gamma_high, delta_high):
        num_res = 0
        acc = 0
        for (success, _, curr_gamma, curr_delta) in results:
            if (gamma_low < curr_gamma <= gamma_high) and (
                    delta_low < curr_delta <= delta_high):
                num_res += 1
                acc += success

        if num_res > 0:
            acc /= num_res

        return num_res, acc

    Z = np.zeros((len(gammas) - 1, len(deltas) - 1))
    for (i, gamma) in enumerate(gammas[1:]):
        for (j, delta) in enumerate(deltas[1:]):
            num_res, acc = get_acc(gammas[i], deltas[j], gamma, delta)
            if plot == 'acc':
                Z[len(gammas) - 2 - i][j] = -1 if num_res == 0 else acc
            else:
                Z[len(gammas) - 2 - i][j] = (num_res / total_res) * 100

    # Using matshow here just because it sets the ticks up nicely. imshow is
    # faster.
    ax.matshow(Z, norm=SymLogNorm(1.0),
               cmap=plt.cm.get_cmap('autumn', 10).reversed())
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        top=False,         # ticks along the top edge are off
        labelbottom=True,
        labeltop=False)    # labels along the top edge are off
    ax.set_xticks([-0.5, 0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('Polarization ($\\delta_{\\text{Usr}}$)')

    ax.set_yticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticklabels([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    ax.set_ylabel('Relevant interest\n($\\gamma_{\\text{Usr}}$)')

    for (i, j), z in np.ndenumerate(Z):
        if plot == 'acc':
            if z < 0:
                ax.text(j, i, '', ha='center', va='center')
            else:
                ax.text(j, i, f'{z:0.2f}', ha='center', va='center')
        else:
            ax.text(j, i, f'{z:0.1f}\\%', ha='center', va='center')
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


def plot_spl_conf(results_file, ax, color=None, label=None):
    """
    Plots special confidence score plot for twitter data
    """
    threshs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = pickle.load(open(results_file, 'rb'))

    def get_acc(thresh_low, thresh_high):
        num_res = 0
        acc = 0
        for (success, scores, _, _) in results:
            total_score = sum(scores)
            curr_thresh = max(scores) / total_score
            if thresh_low < curr_thresh < thresh_high:
                num_res += 1
                acc += success

        if num_res > 0:
            acc /= num_res

        return num_res, acc

    valid_threshs = []
    accs = []
    for (i, thresh) in enumerate(threshs[1:]):
        num_res, acc = get_acc(threshs[i], thresh)
        if num_res >= 10:
            valid_threshs.append((threshs[i] + thresh) / 2)
            accs.append(acc)

    ax.scatter(valid_threshs, accs, color=color, label=label)

    # plot baseline
    xs = np.linspace(0, 1, 10)
    ys = xs
    ax.plot(xs, ys, color='lightgray', linestyle='--')

    ax.set_xticks(threshs)
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Confidence score')
    ax.set_ylabel('Success rate')
