import os

import pandas as pd
from multiprocessing import Pool
from contextlib import closing
from PIL import Image
import matplotlib.pyplot as plt

from custom_loader import *
from dl_manager import binary_evaluate, export_metric
from dl_core import NetTypes2D
from statisitcs import *
from preprocessor import find_big_obj

def basic_statistics():
    #input_path = 'D:/Data/POPF_data/POPF_preOP EMR+BCA_220111.csv'
    #input_path = 'D:/Data/POPF_data/POPF_intraOP EMR_220111.csv'
    # input_path = 'D:/Dropbox/data/POPF/POPF_BCA.csv'
    input_path = 'D:/Dropbox/data/POPF/POPF_RS_2018.csv'
    df = pd.read_csv(input_path)
    label_lst = df.values[:, 2].astype(int)
    data = df.values[:, 4:]
    col_lst = df.columns[4:]

    result_dict = {'Name':[]}
    for idx in range(len(col_lst)):
        val_lst, name = data[:, idx].astype(float), col_lst[idx]
        med = np.nanmedian(val_lst)
        val_lst = [med if np.isnan(x) else x for x in val_lst]
        result_dict['Name'].append(name)
        stat = run_statistic(label_lst, val_lst)

        for key, val in stat.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(val)

    output_path = './result/stat_popf_risk.csv'
    pd.DataFrame(result_dict).to_csv(output_path, index=False)
    print('done')


class BatchAnalyzer:
    PROCESS_NUM = 10

    def __init__(self, input_dir):
        self.input_dir = input_dir

    def collect_roi_size(self, args):
        export_msg, path = args
        image = Image.open(path)
        image = np.array(image)[:, :, 0]
        image[image < 200] = 0.
        image[image >= 200] = 1.

        obj2coord = dict()
        shape = image.shape
        obj_matrix = np.zeros((shape[0], shape[1]))
        obj_idx = 1

        for x in range(shape[0]):
            for y in range(shape[1]):
                if image[x, y] > 0:
                    obj_lst = []

                    if x + 1 < shape[0] and obj_matrix[x + 1, y] != 0:
                        obj_lst.append(obj_matrix[x + 1, y])
                    if y + 1 < shape[1] and obj_matrix[x, y + 1] != 0:
                        obj_lst.append(obj_matrix[x, y + 1])
                    if x - 1 >= 0 and obj_matrix[x - 1, y] != 0:
                        obj_lst.append(obj_matrix[x - 1, y])
                    if y - 1 >= 0 and obj_matrix[x, y - 1] != 0:
                        obj_lst.append(obj_matrix[x, y - 1])

                    if len(obj_lst) == 0:
                        obj_matrix[x, y] = obj_idx
                        if obj_idx not in obj2coord:
                            obj2coord[obj_idx] = []
                        obj2coord[obj_idx].append([x, y])
                        obj_idx += 1
                    elif len(obj_lst) == 1:
                        now_obj = obj_lst[0]
                        obj_matrix[x, y] = now_obj
                        obj2coord[now_obj].append([x, y])
                    else:
                        now_obj = min(obj_lst)
                        obj_matrix[x, y] = now_obj
                        obj2coord[now_obj].append([x, y])

                        for other_obj in obj_lst:
                            if now_obj == other_obj:
                                continue

                            for other_x, other_y in obj2coord[other_obj]:
                                obj_matrix[other_x, other_y] = now_obj
                                obj2coord[now_obj].append([other_x, other_y])
                            obj2coord.pop(other_obj)

        width, height = 0, 0
        if len(obj2coord) == 0:
            return width, height

        tot_coord_lst = []
        for obj, coord_lst in obj2coord.items():
            if len(coord_lst) > 50:
                tot_coord_lst.extend(coord_lst)

        coord_lst = tot_coord_lst
        if len(coord_lst) < 10:
            return width, height

        x_sorted = sorted(np.array(coord_lst)[:, 0])
        y_sorted = sorted(np.array(coord_lst)[:, 1])
        width = x_sorted[-1] - x_sorted[0]
        height = y_sorted[-1] - y_sorted[0]

        # pannel = np.zeros(image.shape)
        # pannel[x_sorted[0], y_sorted[0]:y_sorted[-1]+1] = 1
        # pannel[x_sorted[-1], y_sorted[0]:y_sorted[-1]+1] = 1
        # pannel[x_sorted[0]:x_sorted[-1]+1, y_sorted[0]] = 1
        # pannel[x_sorted[0]:x_sorted[-1]+1, y_sorted[-1]] = 1
        #
        # #plt.title('width %s, height %s' % (width, height))
        # plt.title('y %s-%s, x %s-%s' % (x_sorted[0], x_sorted[-1], y_sorted[0], y_sorted[-1]))
        # plt.imshow(image)
        # plt.imshow(pannel, alpha=0.5)
        # plt.show()
        if export_msg:
            print('%s has been analyzed. (width:%s, height:%s)' % (os.path.basename(path), width, height))

        return width, height

    def run(self, fnc, use_parallel=True):
        input_dir = self.input_dir
        file_lst = os.listdir(input_dir)
        path_lst = []
        idx = 0
        for name in file_lst:
            if not name.endswith('jpg'):
                continue

            if idx % 1000 == 0:
                export_msg = True
            else:
                export_msg = False

            path = '%s/%s' % (input_dir, name)
            path_lst.append([export_msg, path])
            idx += 1

        result_lst = []
        if use_parallel:
            with closing(Pool(self.PROCESS_NUM)) as p:
                result_lst = p.map(fnc, path_lst)
        else:
            for path in path_lst:
                result = fnc(path)
                result_lst.append(result)

        result_dict = {'width':[], 'height':[]}
        for width, height in result_lst:
            if width == 0 or height == 0:
                continue

            result_dict['width'].append(width)
            result_dict['height'].append(height)

        pd.DataFrame(result_dict).to_csv('./result/roi_size.csv', index=False)

def batch_evaluate():
    args_lst = [
        #['./data/pre-op_popf_results.csv', './result/pre_popf.csv'],
       # ['./data/pre-op_cr_results.csv', './result/pre_cr_popf.csv'],
        # ['./data/popf_risk_score.csv', './result/pre_popf.csv'],
        # ['./data/cr-popf_risk_score.csv', './result/pre_cr_popf.csv'],
        ['./data/post-op_popf_results.csv', './result/post_popf2.csv'],
        ['./data/post-op_cr_results.csv', './result/post_cr_popf2.csv']
    ]

    for input_path, output_path in args_lst:
        evaluate_df(input_path, output_path, DataPurpose.Train)
        evaluate_df(input_path, output_path, DataPurpose.Validation)
        evaluate_df(input_path, output_path, DataPurpose.Test)

def evaluate_df(input_path, output_path, data_purpose=DataPurpose.Test):
    col2cut_off = {
        'a-FRS': 0.2,
        'POPF_score': 0.2
    }

    df = pd.read_csv(input_path)

    if data_purpose == DataPurpose.Test:
        df = df.loc[df['dataset']==2]
    elif data_purpose == DataPurpose.Validation:
        df = df.loc[df['dataset']==1]
    elif data_purpose == DataPurpose.Train:
        df = df.loc[df['dataset']==0]


    label_lst = df['target'].values
    score_mat = df.iloc[:, 3:].values
    columns = list(df.columns)[3:]

    for c_idx in range(len(columns)):
        col_name = columns[c_idx]
        cut_off = 0.5
        if col_name in col2cut_off:
            cut_off = col2cut_off[col_name]
        score_lst = score_mat[:, c_idx]
        metric = binary_evaluate(label_lst, score_lst, cut_off)
        export_metric(output_path, metric, col_name)
    print('...evaluate %s and export to %s' % (os.path.basename(input_path), os.path.basename(output_path)))


def merge_attention_map(processes=8):
    dataset = ImageDataset(DataPurpose.Test)
    # net_lst = [NetTypes2D.Resnet50, NetTypes2D.Inception3]
    # att_dir = './data/220603/d2_popf/attention/pkl'
    # out_dir = './data/220603/d2_popf/attention/png'

    net_lst = [NetTypes2D.Resnet50, NetTypes2D.Densenet121, NetTypes2D.Resnext50]
    att_dir = './data/220603/d2_cr-popf/attention/pkl'
    out_dir = './data/220603/d2_cr-popf/attention/png'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    path_lst, p_idx = dataset.path_lst, 0
    args_lst = []
    for image, l1, l2 in dataset:
        path = path_lst[p_idx]
        pid, idx = parse_path(path)
        p_idx += 1

        att_path = '%s/%s.pkl' % (att_dir, pid)
        pid2att = load_obj(att_path)
        out_path = '%s/%s.png' % (out_dir, pid)
        args = [net_lst, image, pid2att, out_path]
       # export_fusion_attention(args)
        args_lst.append(args)

    print('%s/%s args are ready!' % (p_idx, len(path_lst)))

    with closing(Pool(processes)) as p:
        p.map(export_fusion_attention, args_lst)


def export_fusion_attention(args):
    net_lst, image, pid2att, out_path = args
    att_lst = []
    for net_type in net_lst:
        tag = net_type.value
        att = pid2att[tag]
        val_lst = sorted(list(att.flatten()))
        idx = int(np.round(len(val_lst) * 0.2))
        threshold = val_lst[idx]
        att[att<threshold] = threshold
        att_lst.append(att)
    tot_att = np.mean(np.array(att_lst), axis=0)
    val_lst = sorted(list(tot_att.flatten()))
    idx = int(np.round(len(val_lst) * 0.8))
    threshold = val_lst[idx]
    alpha = np.zeros(tot_att.shape)
    alpha[tot_att >= threshold] = 0.25
    tot_att = (tot_att - np.min(tot_att)) / (np.max(tot_att) - np.min(tot_att)) * 100

    plt.imshow(image[0], cmap='gray')
    plt.imshow(tot_att, cmap='jet', alpha=alpha)
    plt.axis('off')
    #plt.show()

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    pid = os.path.basename(out_path).split('.')[0]
    print('..%s has been exported' % pid)


def plot_attenion(net_lst, pid2att, image):
    idx = 1
    row = len(net_lst)
    for net_type in net_lst:
        tag = net_type.value
        att = pid2att[tag]
        msg = '[%s] max:%.2f, mean:%.2f, min:%.2f' % (tag, np.max(att), np.mean(att), np.min(att))
        plt.subplot(row, 1, idx)
        plt.imshow(image[0], cmap='gray')
        plt.title(msg)
        plt.imshow(att, cmap='reds', alpha=0.25)
        idx += 1
    plt.show()
