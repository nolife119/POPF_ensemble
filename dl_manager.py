import os

from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import ParameterGrid
from medcam import medcam
import scipy.ndimage as sci_img

from common import *
from custom_loader import *
from dl_core import *
from statistics import *


class CMtypes(enum.Enum):
    TP = 'True Positive'
    TN = 'True Negative'
    FP = 'False Positive'
    FN = 'False Negative'

    ACC = 'Accuracy'
    SEN = 'Sensitivity'
    SPE = 'Specificity'
    PPV = 'Positive predictive values'
    NPV = 'Negative predictive values'
    F1 = 'F1score'
    F2 = 'F2score'

    AUROC = 'AUROC'
    AUPRC = 'AUPRC'


def export_metric(output_path, metric_dict, tag=None):
    head = 'tag'
    line = tag if tag is not None else ' '
    for key, value in metric_dict.items():
        head = '%s,%s' % (head, key)
        if type(value) == int:
            line = '%s,%d' % (line, value)
        else:
            line = '%s,%.3f' % (line, value)
    head = '%s\n' % head
    line = '%s\n' % line

    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write(head)

    with open(output_path, 'a') as f:
        f.write(line)


def binary_evaluate(y_real, y_pred, cut_off=0.5):
    b_pred = [1 if y >= cut_off else 0 for y in y_pred]

    tp = tn = fp = fn = 0
    for i in range(len(y_real)):
        actual = y_real[i]
        predict = b_pred[i]
        if predict == 1 and actual == 1:
            tp += 1
        elif predict == 0 and actual == 0:
            tn += 1
        elif predict == 1 and actual == 0:
            fp += 1
        elif predict == 0 and actual == 1:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fn + fp) if (tp + tn + fn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall = sensitivity
    f1_value = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2_value = (5 * precision * recall) / ((4 * precision) + recall) if (precision + recall) > 0 else 0

    result_dict = dict()
    result_dict[CMtypes.TP.value] = tp
    result_dict[CMtypes.TN.value] = tn
    result_dict[CMtypes.FP.value] = fp
    result_dict[CMtypes.FN.value] = fn

    result_dict[CMtypes.ACC.value] = accuracy
    result_dict[CMtypes.SEN.value] = sensitivity
    result_dict[CMtypes.SPE.value] = specificity
    result_dict[CMtypes.PPV.value] = precision
    result_dict[CMtypes.NPV.value] = npv
    result_dict[CMtypes.F1.value] = f1_value
    result_dict[CMtypes.F2.value] = f2_value

    auroc = roc_auc_score(y_real, y_pred)
    auprc = average_precision_score(y_real, y_pred)
    result_dict[CMtypes.AUROC.value] = auroc
    result_dict[CMtypes.AUPRC.value] = auprc
    return result_dict


class DeepLearningManager:
    def __init__(self, result_dir, net_type=NetTypes2D.Resnet50, config=None):
        self.result_dir = result_dir
        self.start_epoch = 0
        self.net_type = net_type

        if config is None:
            config = ConfigManager().load()

        self.batch_size = int(config['batch_size'])
        self.max_epoch = int(config['max_epoch'])
        self.tolerance = int(config['tolerance'])
        self.n_classes = int(config['n_classes'])
        self.config = config

    def export_attention_map(self, weight_path, gpu_num=0):
        device = torch.device('cuda:%s' % gpu_num)

        net_C = DeepLearningModel2D(self.net_type).to(device)
        net_C.load_state_dict(torch.load(weight_path, map_location=device))
        net_C = medcam.inject(net_C, backend='gcampp', layer='auto', label=0)
        net_C.eval()

        dataset = ImageDataset(DataPurpose.Test, self.config)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        path_lst, p_idx = dataset.path_lst, 0
        tag = self.net_type.value
        for volume, label1, label2 in dataloader:
            path = path_lst[p_idx]
            pid, idx = parse_path(path)
            p_idx += 1

            input_tensor = Variable(volume).to(device, dtype=torch.float32)
            out1 = net_C(input_tensor)

            attention = net_C.get_attention_map()[0, 0, :, :]
            real_resize_factor = input_tensor.shape[2:] / np.array(attention.shape)
            attention = sci_img.interpolation.zoom(attention, real_resize_factor, mode='nearest')

            lable_pred = torch.sigmoid(out1).detach().cpu().numpy()[0]
            lable_real = label2.numpy()[0]
            # lable_pred = np.argmax(prediction)
            msg = '[ID:%s] POPF %s vs prediction %.3f' % (pid, lable_real, lable_pred)

            img = volume.numpy()[0]
            att = np.expand_dims(attention, 0)

            # viewer = StackViewer(img, att, msg)
            # viewer.show()

            output_path = '%s/%s.pkl' % (self.result_dir, pid)
            if os.path.exists(output_path):
                result_dict = load_obj(output_path)
            else:
                result_dict = dict()

            result_dict[tag] = attention
            save_obj(output_path, result_dict)

            print(msg)
            #break


    def test_model(self, gpu_num=0):
        print('..start testing %s model (gpu:%s)' % (self.net_type.value, gpu_num))
        device = torch.device('cuda:%s' % gpu_num)

        config_path = '%s/config.ini' % self.result_dir
        config = ConfigManager(config_path).load()

        test_set = ImageDataset(DataPurpose.Test, config)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        net_C = DeepLearningModel2D(self.net_type).to(device)
        weight_path = '%s/best.pt' % self.result_dir
        net_C.load_state_dict(torch.load(weight_path, map_location=device))
        net_C.eval()

        y_real1, y_pred1, y_real2, y_pred2, loss = self._run(DataPurpose.Test, net_C, device, test_loader)
        test_metric = binary_evaluate(y_real1, y_pred1)
        output_path = '%s/test_total_metric.csv' % self.result_dir
        export_metric(output_path, test_metric, 'popf')
        test_metric = binary_evaluate(y_real2, y_pred2)
        export_metric(output_path, test_metric, 'cr-popf')

        path_lst = test_set.path_lst
        pid2result = dict()
        for idx in range(len(path_lst)):
            path = path_lst[idx]
            f_name = os.path.basename(path)

            name_only = f_name.split('.')[0]
            now_id, img_idx = map(int, name_only.split('_'))
            if now_id not in pid2result:
                pid2result[now_id] = []
            r1, p1, r2, p2 = y_real1[idx], y_pred1[idx], y_real2[idx], y_pred2[idx]
            pid2result[now_id].append([img_idx, r1, p1, r2, p2])
            # if (idx + 1) >= len(y_pred1):
            #     break

        pid_lst = []
        r_lst1, p_lst1, r_lst2, p_lst2 = [], [], [], []
        for pid, result_lst in pid2result.items():
            result_lst = sorted(result_lst)
            result_mat = np.array(result_lst)[:, 1:]
            # print(np.mean(result_mat, axis=0).shape)

            r1, p1, r2, p2 = np.mean(result_mat, axis=0)
            print([pid, r1, p1, r2, p2])
            pid_lst.append(pid)
            r_lst1.append(r1)
            p_lst1.append(p1)
            r_lst2.append(r2)
            p_lst2.append(p2)

        result_dict = {'pid': pid_lst, 'is_popf': r_lst1, 'popf_model': p_lst1, 'is_cr-popf': r_lst2,
                       'cr-popf_model': p_lst2}
        output_path = '%s/result.csv' % self.result_dir
        pd.DataFrame(result_dict).to_csv(output_path, index=False)

        result_dict = {'tag': []}
        stat1 = run_statistic(r_lst1, p_lst1)
        result_dict['tag'].append('popf')
        for key, val in stat1.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(val)
        threshold = stat1["threshold"]
        p_lst1 = [1 if y >= threshold else 0 for y in p_lst1]
        test_metric = binary_evaluate(r_lst1, p_lst1)
        output_path = '%s/test_sample_metric.csv' % self.result_dir
        export_metric(output_path, test_metric, 'popf')

        stat2 = run_statistic(r_lst2, p_lst2)
        result_dict['tag'].append('cr-popf')
        for key, val in stat1.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(val)
        threshold = stat1["threshold"]
        p_lst2 = [1 if y >= threshold else 0 for y in p_lst2]
        test_metric = binary_evaluate(r_lst2, p_lst2)
        # output_path = '%s/test_metric.csv' % self.result_dir
        export_metric(output_path, test_metric, 'cr-popf')

        output_path = '%s/statistic.csv' % self.result_dir
        pd.DataFrame(result_dict).to_csv(output_path, index=False)

    def train_model(self, gpu_num=0):
        print('..start training %s model (gpu:%s)' % (self.net_type.value, gpu_num))
        device = torch.device('cuda:%s' % gpu_num)
        train_set = ImageDataset(DataPurpose.Train, self.config)
        # sampler = ImbalancedDatasetSampler(train_set)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        val_set = ImageDataset(DataPurpose.Validation, self.config)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        net_C = DeepLearningModel2D(self.net_type).to(device)
        optim_C = torch.optim.Adam(net_C.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-5)

        log_path = '%s/log.csv' % self.result_dir
        logger = Logger(log_path)

        config_path = '%s/config.ini' % self.result_dir
        ConfigManager().save(self.config, config_path)

        start_epoch, max_epoch = 1, self.max_epoch
        max_score, patience = 0, 0
        is_end = False
        for epoch in range(start_epoch, max_epoch):
            y_real1, y_pred1, y_real2, y_pred2, train_loss = self._run(DataPurpose.Train, net_C, device, train_loader,
                                                                       epoch, optim_C)
            train_metric = binary_evaluate(y_real1, y_pred1)
            train_score1 = train_metric[CMtypes.F1.value]
            train_metric = binary_evaluate(y_real2, y_pred2)
            train_score2 = train_metric[CMtypes.F1.value]

            y_real1, y_pred1, y_real2, y_pred2, val_loss = self._run(DataPurpose.Validation, net_C, device, val_loader,
                                                                     epoch)
            val_metric = binary_evaluate(y_real1, y_pred1)
            val_score1 = val_metric[CMtypes.F1.value]
            val_metric = binary_evaluate(y_real2, y_pred2)
            val_score2 = val_metric[CMtypes.F1.value]

            logger.log(str(epoch), {"train_loss": train_loss, 'val_loss': val_loss, \
                                    'train_score1': train_score1, 'train_score2': train_score2, \
                                    'val_score1': val_score1, 'val_score2': val_score2})

            if epoch == 1:
                torch.save(net_C.state_dict(), '%s/best.pt' % self.result_dir)
            elif max_score < val_score1 and val_score1 <= (train_score1 + 0.05):
                max_score = val_score1
                torch.save(net_C.state_dict(), '%s/best.pt' % self.result_dir)
                patience = 0
            elif patience + 1 < self.tolerance:
                if train_score1 < 0.9:
                    patience += 1
                else:
                    patience += 10
                    if self.tolerance <= patience:
                        is_end = True
            else:
                is_end = True

            msg = '...epoch %s-popf:train:%.3f, val:%.3f (best:%.3f, %s) cr-popf:train:%.3f, val:%.3f' % \
                  (epoch, train_score1, val_score1, max_score, patience, train_score2, val_score2)
            print(msg)
            if is_end:
                break
        print('..end training %s model (best:%.3f)' % (self.net_type.value, max_score))

    def _run(self, data_purpose, net_C, device, loader, epoch='E', optim_C=None):
        n_classes = self.n_classes
        crit_bce = nn.BCEWithLogitsLoss()
        crit_cel = nn.CrossEntropyLoss()

        idx = 0
        c_loss_sum = 0
        if data_purpose == DataPurpose.Train:
            net_C.train()
        else:
            net_C.eval()

        y_real1, y_pred1 = [], []
        y_real2, y_pred2 = [], []
        for volume, target1, target2 in loader:
            vol_cuda = Variable(volume).to(device, dtype=torch.float32)
            target1_cuda = Variable(target1).to(device)
            target2_cuda = Variable(target2).to(device)

            if data_purpose == DataPurpose.Train:
                optim_C.zero_grad()
                out1, out2 = net_C(vol_cuda)
            else:
                with torch.no_grad():
                    out1, out2 = net_C(vol_cuda)

            if n_classes == 1:
                if out1.shape[0] > 1:
                    out1 = out1.squeeze()
                    out2 = out2.squeeze()
                else:
                    out1 = out1[0]
                    out2 = out2[0]

                c_loss = crit_bce(out1, target1_cuda.float()) + crit_bce(out2, target2_cuda.float())
            else:
                c_loss = crit_cel(out1, target1_cuda.long()) + crit_cel(out2, target2_cuda.long())

            if data_purpose == DataPurpose.Train:
                c_loss.backward()
                optim_C.step()

            loss_itm = c_loss.item()
            if not np.isnan(loss_itm):
                c_loss_sum += loss_itm
            idx += 1

            label_arr1 = target1.numpy()
            label_arr2 = target2.numpy()
            if n_classes == 1:
                pred_arr1 = torch.sigmoid(out1).cpu().detach().numpy()
                pred_arr2 = torch.sigmoid(out2).cpu().detach().numpy()
            else:
                pred_arr1 = torch.softmax(out1, dim=1).cpu().detach().numpy()
                pred_arr2 = torch.softmax(out2, dim=1).cpu().detach().numpy()

            for i in range(len(target1)):
                y_real1.append(label_arr1[i])
                y_real2.append(label_arr2[i])
                if n_classes == 1:
                    # pred1 = 1 if pred_arr1[i] >= 0.5 else 0
                    # pred2 = 1 if pred_arr2[i] >= 0.5 else 0
                    pred1 = pred_arr1[i]
                    pred2 = pred_arr2[i]
                else:
                    pred1 = np.argmax(pred_arr1[i])
                    pred2 = np.argmax(pred_arr2[i])
                y_pred1.append(pred1)
                y_pred2.append(pred2)

            if idx % (len(loader) // 3) == 0:
                msg = '...%s [%s-%s], loss_C:%.3f' % (data_purpose.value, epoch, idx, c_loss_sum / idx)
                print(msg)

        loss = c_loss_sum / idx
        return y_real1, y_pred1, y_real2, y_pred2, loss



class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_dict = {'tag': []}
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            self.log_dict = df.to_dict(orient='list')

    def log(self, tag, loss_dict=None):
        log_dict = self.log_dict
        log_dict['tag'].append(tag)
        if loss_dict is not None:
            for key, value in loss_dict.items():
                if key not in log_dict:
                    log_dict[key] = []
                log_dict[key].append(value)

        df = pd.DataFrame(log_dict)
        df.to_csv(self.log_path, index=False)