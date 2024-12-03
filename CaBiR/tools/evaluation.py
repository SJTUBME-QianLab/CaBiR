import numpy as np
import pandas as pd
import os
import re
import pickle
import sklearn
import itertools

from tools import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class SettleResults:
    def __init__(self, label_df, task, exp_dir, metrics, ave):
        self.label_df = label_df
        self.task = task
        self.exp_dir = exp_dir
        self.metrics = [kk for kk in metrics if kk != 'auc']
        self.ave = ave
        self.num_class, self.label_name = utils.parse_label(self.task)
        if self.num_class <= 2:
            assert ave is None
        assert ave in ['macro', 'micro', None]

    def concat_trend_scores(self, num_epoch, subset='test', start_epoch=0):
        if os.path.isfile(os.path.join(self.exp_dir, 'epoch', f'epoch{start_epoch}_{subset}_score.pkl')):
            if not os.path.isfile(os.path.join(self.exp_dir, 'epoch', f'epoch{num_epoch-1}_{subset}_score.pkl')):
                print('------------------%s does not complete!' % self.exp_dir)
                return True
            evals_list = []
            for epo in range(start_epoch, num_epoch):
                with open(os.path.join(self.exp_dir, 'epoch', f'epoch{epo}_{subset}_score.pkl'), 'rb') as f:
                    score_pred_df = pickle.load(f)
                score_pred_df = pd.DataFrame(score_pred_df)
                scores = pd.concat([score_pred_df, self.label_df], axis=0).dropna(axis=1)
                evals_list.append(get_evaluations(scores, self.num_class, self.metrics, self.ave))
            evals_df = pd.DataFrame(evals_list, columns=get_columns(self.num_class, self.metrics, self.ave))
            evals_df.to_csv(os.path.join(self.exp_dir, f'trend_metrics_{subset}.csv'), index=False)
        elif os.path.isfile(os.path.join(self.exp_dir, 'epoch', f'allepo_{start_epoch}.{num_epoch}_{subset}_score.pkl')):
            with open(os.path.join(self.exp_dir, 'epoch', f'allepo_{start_epoch}.{num_epoch}_{subset}_score.pkl'), 'rb') as f:
                all_score_pred = pickle.load(f)
            evals_list = []
            for epo, score_pred_df in all_score_pred.items():
                score_pred_df = pd.DataFrame(score_pred_df)
                scores = pd.concat([score_pred_df, self.label_df], axis=0).dropna(axis=1)
                evals_list.append(get_evaluations(scores, self.num_class, self.metrics, self.ave))
            evals_df = pd.DataFrame(evals_list, columns=get_columns(self.num_class, self.metrics, self.ave))
            evals_df.to_csv(os.path.join(self.exp_dir, f'trend_metrics_{subset}.csv'), index=False)
        elif not os.path.isfile(os.path.join(self.exp_dir, 'log.txt')):
            print('------------------%s missing log.txt!' % self.exp_dir)
            return True  # failed
        else:
            print(f'Cannot find epoch{start_epoch}_{subset}_score.pkl or allepo_{start_epoch}.{num_epoch}_{subset}_score.pkl')
            return True

    def merge_pkl(self, num_epoch, start_epoch=0, subset='test'):
        if os.path.isfile(os.path.join(self.exp_dir, 'epoch', f'allepo_{start_epoch}.{num_epoch}_{subset}_score.pkl')):
            print('-' * 10 + f' {subset} have merged!' + '-' * 10)
            return False
        elif not os.path.isfile(os.path.join(self.exp_dir, 'epoch', f'epoch{start_epoch}_{subset}_score.pkl')):
            if not os.path.isfile(os.path.join(self.exp_dir, 'log.txt')):
                print('-' * 10 + f' {subset} missing log.txt!' + '-' * 10)
                return True  # failed
        if not os.path.isfile(os.path.join(self.exp_dir, 'epoch', f'epoch{num_epoch-1}_{subset}_score.pkl')):
            print('-' * 10 + f' {subset}  does not complete!' + '-' * 10)
            return False
        score_all = {}
        for epo in range(start_epoch, num_epoch):
            with open(os.path.join(self.exp_dir, 'epoch', f'epoch{epo}_{subset}_score.pkl'), 'rb') as f:
                score_pred = pickle.load(f)
            score_all[epo] = score_pred
        with open(os.path.join(self.exp_dir, 'epoch', f'allepo_{start_epoch}.{num_epoch}_{subset}_score.pkl'), 'wb') as f:
            pickle.dump(score_all, f)
        utils.remove_files(os.path.join(self.exp_dir, 'epoch'), f'epoch[0-9]*_{subset}_score.pkl')

    def get_final_score(self, subset='test'):
        file = [kk for kk in os.listdir(os.path.join(self.exp_dir, 'epoch'))
                if re.search(f'allepo_[0-9]*.[0-9]*_{subset}_score.pkl', kk) is not None]
        if not (len(file) == 1 and file[0][:6] == 'allepo'):
            print(os.path.join(self.exp_dir, 'epoch'), str(file))
            return None
        with open(os.path.join(self.exp_dir, 'epoch', file[0]), 'rb') as f:
            score_pred_df = pickle.load(f)
        final_index = sorted(score_pred_df.keys())[-1]
        score_pred_df = pd.DataFrame(score_pred_df[final_index])
        if len(score_pred_df) != self.num_class:
            score_pred_df = score_pred_df.iloc[-self.num_class:, :]
        scores = pd.concat([score_pred_df, self.label_df], axis=0).dropna(axis=1)
        return scores

    def confusion_matrix(self, out_path, subset='test'):
        scores = self.get_final_score(subset=subset)
        cm = eval_CM(scores.iloc[-1, :], scores.iloc[:-1, :])
        plot_confusion_matrix(out_path, cm=cm, classes=self.label_name)


# ------------------------ Functions for evaluation performance ------------------------ #


def onehot_code(Y, num_class):
    Y = np.array(Y)
    Yc_onehot = np.zeros((len(Y), num_class))
    for i in range(num_class):
        Yc_onehot[np.where(Y == i)[0], i] = 1.0
    return Yc_onehot


def get_evaluations(scores, num_class, metrics, ave=None):
    if num_class <= 2:
        assert ave is None
    if num_class == 1:
        evals = metric_regression(true=scores.iloc[-1, :], score=scores.iloc[:-1, :])
        return [evals[kk] for kk in metrics]
    elif num_class == 2:
        evals = metric_cl2(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
        return [evals[kk] for kk in metrics]
    else:
        evals = metric_clx(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
        return [evals[ave][kk] for kk in metrics]


def get_auc(scores, num_class, ave=None):
    if num_class <= 2:
        assert ave is None
    if num_class == 1:
        return None
    elif num_class == 2:
        auc, _, _ = auc_cl2(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
        return auc
    else:
        auc, _, _ = auc_clx(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :], num_class=num_class)
        return auc[ave]


def get_columns(num_class, metrics, ave=None):
    if num_class <= 2:
        assert ave is None
    if num_class == 1:
        columns = metrics
    elif num_class == 2:
        columns = metrics
    else:
        columns = [kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics]
    return columns


def metric_regression(true, score):
    if isinstance(score, pd.DataFrame):
        score = score.values.reshape(*true.shape)
    mae = sklearn.metrics.mean_absolute_error(true, score)  # np.mean(abs(true-score))
    mse = sklearn.metrics.mean_squared_error(true, score)  # np.mean((true-score)**2)
    rmse = np.sqrt(mse)
    r2 = sklearn.metrics.r2_score(true, score)  # R2 in [0,1]
    corr = np.corrcoef(true, score)[0, 1]  # [-1,1]
    evals = {
        'mae': mae, 'mse': mse, 'rmse': rmse,
        'r2': r2,
        'Pear': corr, 'corr': corr,
    }
    return evals


def metric_cl2(true, prob):
    assert len(prob.shape) == 2
    num_class = prob.shape[0] if prob.shape[1] == len(true) else prob.shape[1]
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    pred = list(prob.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    # acc
    acc = sum([pred[i] == true[i] for i in range(len(true))]) / len(true)

    # confusion matrix
    con_matrix = np.array(
        [[sum([pred[i] == k1 and true[i] == k2 for i in range(len(true))]) for k1 in range(num_class)] for k2 in range(num_class)])
    tn, fp, fn, tp = con_matrix.ravel()
    con_arr = con_matrix.ravel()

    SEN_cal = lambda tn, fp, fn, tp: tp / (tp + fn) if (tp + fn) != 0 else 0
    PRE_cal = lambda tn, fp, fn, tp: tp / (tp + fp) if (tp + fp) != 0 else 0
    SPE_cal = lambda tn, fp, fn, tp: tn / (tn + fp) if (tn + fp) != 0 else 0
    NPV_cal = lambda tn, fp, fn, tp: tn / (tn + fn) if (tn + fn) != 0 else 0
    MCC_cal = lambda tn, fp, fn, tp: (tp*tn - fp*fn) / (np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-8)

    sen = SEN_cal(*con_arr)
    pre = PRE_cal(*con_arr)
    spe = SPE_cal(*con_arr)
    npv = NPV_cal(*con_arr)
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) != 0 else 0
    mcc = MCC_cal(*con_arr)
    bacc = (sen + spe) / 2

    evals = {
        'confusion_matrix': con_matrix,
        'acc': acc, 'accuracy': acc,
        'pre': pre, 'precision': pre, 'ppv': pre,
        'npv': npv,
        'sen': sen, 'sensitivity': sen, 'recall': sen, 'tpr': sen,
        'spe': spe, 'specificity': spe, 'tnr': spe,
        'fpr': 1-spe,
        'f1': f1, 'f1_score': f1, 'f1score': f1,
        'mcc': mcc,
        'bacc': bacc,
    }

    return evals


def metric_clx(true, prob):
    assert len(prob.shape) == 2
    num_class = prob.shape[0] if prob.shape[1] == len(true) else prob.shape[1]
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    pred = list(prob.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    # acc
    acc = sum([pred[i] == true[i] for i in range(len(true))]) / len(true)

    # confusion matrix
    con_matrix = np.array(
        [[sum([pred[i] == k1 and true[i] == k2 for i in range(len(true))]) for k1 in range(num_class)] for k2 in range(num_class)])

    con_arr = np.zeros((num_class, 4))
    for k in range(num_class):
        tp = sum([pred[i] == k and true[i] == k for i in range(len(true))])
        fp = sum([pred[i] == k and true[i] != k for i in range(len(true))])
        tn = sum([pred[i] != k and true[i] != k for i in range(len(true))])
        fn = sum([pred[i] != k and true[i] == k for i in range(len(true))])
        # print(tn, fp, fn, tp)
        con_arr[k, :] = [tn, fp, fn, tp]

    SEN_cal = lambda tn, fp, fn, tp: tp / (tp + fn) if (tp + fn) != 0 else 0
    PRE_cal = lambda tn, fp, fn, tp: tp / (tp + fp) if (tp + fp) != 0 else 0
    SPE_cal = lambda tn, fp, fn, tp: tn / (tn + fp) if (tn + fp) != 0 else 0
    NPV_cal = lambda tn, fp, fn, tp: tn / (tn + fn) if (tn + fn) != 0 else 0
    MCC_cal = lambda tn, fp, fn, tp: (tp*tn - fp*fn) / (np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-8)

    # macro
    sen = np.nansum(np.array([SEN_cal(*cc) for cc in con_arr])) / num_class
    pre = np.nansum(np.array([PRE_cal(*cc) for cc in con_arr])) / num_class
    spe = np.nansum(np.array([SPE_cal(*cc) for cc in con_arr])) / num_class
    npv = np.nansum(np.array([NPV_cal(*cc) for cc in con_arr])) / num_class
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) != 0 else 0
    mcc = np.nansum(np.array([MCC_cal(*cc) for cc in con_arr])) / num_class
    bacc = (sen + spe) / 2

    # micro
    sen_mi = SEN_cal(*list(np.sum(con_arr, axis=0)))
    pre_mi = PRE_cal(*list(np.sum(con_arr, axis=0)))
    spe_mi = SPE_cal(*list(np.sum(con_arr, axis=0)))
    npv_mi = NPV_cal(*list(np.sum(con_arr, axis=0)))
    f1_mi = 2 * pre_mi * sen_mi / (pre_mi + sen_mi) if (pre_mi + sen_mi) != 0 else 0
    mcc_mi = MCC_cal(*list(np.sum(con_arr, axis=0)))
    bacc_mi = (sen_mi + spe_mi) / 2

    evals = {
        'macro': {
            'confusion_matrix': con_matrix,
            'acc': acc, 'accuracy': acc,
            'pre': pre, 'precision': pre, 'ppv': pre,
            'npv': npv,
            'sen': sen, 'sensitivity': sen, 'recall': sen, 'tpr': sen,
            'spe': spe, 'specificity': spe, 'tnr': spe,
            'fpr': 1-spe,
            'f1': f1, 'f1_score': f1, 'f1score': f1,
            'mcc': mcc,
            'bacc': bacc,
        },
        'micro': {
            'confusion_matrix': con_matrix,
            'acc': acc, 'accuracy': acc,
            'pre': pre_mi, 'precision': pre_mi, 'ppv': pre_mi,
            'npv': npv_mi,
            'sen': sen_mi, 'sensitivity': sen_mi, 'recall': sen_mi, 'tpr': sen_mi,
            'spe': spe_mi, 'specificity': spe_mi, 'tnr': spe_mi,
            'fpr': 1-spe_mi,
            'f1': f1_mi, 'f1_score': f1_mi, 'f1score': f1_mi,
            'mcc': mcc_mi,
            'bacc': bacc_mi,
        }
    }

    return evals


def auc_cl2(true, prob, num_class=None):
    if num_class is None:
        num_class = int(max(true) + 1)
    assert num_class == 2
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    # y_onehot = onehot_code(true, num_class)

    if not np.allclose(prob.sum(axis=1), np.ones(len(prob))):
        prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True)
        assert np.allclose(prob.sum(axis=1), np.ones(len(prob)))

    fpr, tpr, _ = sklearn.metrics.roc_curve(true, prob[:, 1])
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    return roc_auc, fpr, tpr


def auc_clx(true, prob, num_class=None):
    if num_class is None:
        num_class = int(max(true) + 1)
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    y_onehot = onehot_code(true, num_class)

    if not np.allclose(prob.sum(axis=1), np.ones(len(prob))):
        prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True)
        if np.isinf(prob).any() or np.isnan(prob).any():
            return [{'macro': -1, 'micro': -1}] * 3
        assert np.allclose(prob.sum(axis=1), np.ones(len(prob)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_onehot[:, i], prob[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_onehot.ravel(), prob.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    return roc_auc, fpr, tpr


def ap_cl2(true, prob, num_class=None):
    if num_class is None:
        num_class = int(max(true) + 1)
    assert num_class == 2
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    # y_onehot = onehot_code(true, num_class)

    if not np.allclose(prob.sum(axis=1), np.ones(len(prob))):
        prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True)
        assert np.allclose(prob.sum(axis=1), np.ones(len(prob)))

    pre, rec, _ = sklearn.metrics.precision_recall_curve(true, prob[:, 1])
    ap = sklearn.metrics.average_precision_score(true, prob[:, 1])
    return ap, pre, rec


def ap_clx(true, prob, num_class=None):
    if num_class is None:
        num_class = int(max(true) + 1)
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    y_onehot = onehot_code(true, num_class)

    if not np.allclose(prob.sum(axis=1), np.ones(len(prob))):
        prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True)
        if np.isinf(prob).any() or np.isnan(prob).any():
            return [{'macro': -1, 'micro': -1}] * 3
        assert np.allclose(prob.sum(axis=1), np.ones(len(prob)))

    pre = dict()
    rec = dict()
    ap = dict()
    for i in range(num_class):
        pre[i], rec[i], _ = sklearn.metrics.precision_recall_curve(y_onehot[:, i], prob[:, i])
        ap[i] = sklearn.metrics.average_precision_score(y_onehot[:, i], prob[:, i])

    # micro
    pre["micro"], rec["micro"], _ = sklearn.metrics.precision_recall_curve(y_onehot.ravel(), prob.ravel())
    ap["micro"] = sklearn.metrics.average_precision_score(y_onehot, prob, average='micro')

    # macro
    # First aggregate all false positive rates
    all_pre = np.unique(np.concatenate([pre[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_rec = np.zeros_like(all_pre)
    for i in range(num_class):
        mean_rec += np.interp(all_pre, pre[i], rec[i])
    # Finally average it and compute AUC
    mean_rec /= num_class
    pre["macro"] = all_pre
    rec["macro"] = mean_rec
    ap["macro"] = sklearn.metrics.average_precision_score(y_onehot, prob, average='macro')
    
    return ap, pre, rec


def ave_line(fpr, tpr):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(fpr)):
        interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
        tprs.append(interp_tpr)
    return mean_fpr, tprs


def eval_CM(true, prob, num_class=None, print_info=False):
    if num_class is None:
        num_class = int(max(true) + 1)
    assert num_class in prob.shape
    pred = list(prob.values.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    con_matrix = sklearn.metrics.confusion_matrix(true, pred)
    if print_info:
        acc = sklearn.metrics.accuracy_score(true, pred)
        print(con_matrix)
        print(acc)

    return con_matrix


def plot_confusion_matrix(out_path, cm, classes, cmap=plt.cm.GnBu, size=10):  # PuBu
    """
    This function prints and plots the confusion matrix.
    Input
    - cm : confusion matrix
    - classes : class names
    """
    if size > 10:
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300

    num_cm = cm
    if isinstance(classes, dict):
        if isinstance(list(classes.keys())[0], str):
            classes = {int(vv): kk for kk, vv in classes.items()}
        classes = [classes[i] for i in range(len(classes))]

    f, ax = plt.subplots(figsize=(size/4, size/4))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set_ylabel('True Label', weight="bold", size=size)
    ax.set_xlabel('Predicted Label', weight="bold", size=size)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.xaxis.set_tick_params(labelsize=size)
    ax.yaxis.set_tick_params(labelsize=size, rotation=90)
    ax.xaxis.set_ticks_position('top')

    # add number
    ax.set_ylim(len(classes) - 0.5, -0.5)
    thresh = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(num_cm[i, j], 'd'),
                horizontalalignment="center", verticalalignment="bottom",
                color="white" if cm[i, j] > thresh else "black", fontsize=int(size*1.2))
        ax.text(j, i, '({:.2f})'.format(cm[i, j]),
                horizontalalignment="center", verticalalignment="top",
                color="white" if cm[i, j] > thresh else "black", fontsize=int(size*0.9))

    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

    # ax.set_title(name, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    plt.close()

