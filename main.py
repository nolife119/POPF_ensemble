import os
from preprocessor import *
from analyzer import *
from common import *
from custom_loader import load_pkl_and_show
from esm_manager import *

def test_dataset():
    dataset = ImageDataset(DataPurpose.Test)
    path_lst = dataset.path_lst
    p_idx = 0
    for image, t1, t2 in dataset:
        path = path_lst[p_idx]
        p_idx += 1
        n_only = os.path.basename(path).split('.')[0]
        if '3146' not in n_only:
            continue

        img = image[0, :, :]
        plt.imshow(img, cmap='gray')
        plt.title(str(n_only))
        plt.axis('off')
        out_path = './result/%s_all.png' % n_only
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        break
    print('done!')




def preprocess():
    config = ConfigManager().load()
    dcm_dir = config['dcm_root']
    jpg_dir = config['jpg_root']
    pp_dir = config['pp_root']

    result_dir = config['roi_root']
    result_dir = './result/segment'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    pp = Preprocessor(pp_dir, result_dir, '.jpg')
    pp.batch_run(pp.segment, False)

def run_batch_analysis():
    roi_dir = 'D:/Data/POPF_data/POPF_roi'
    analyzer = BatchAnalyzer(roi_dir)
    analyzer.run(analyzer.collect_roi_size)

def show_image_and_roi(id):
    roi_info = pd.read_csv('./config/roi_info.csv')
    roi_info_part = roi_info.loc[roi_info['id']==id]
    if len(roi_info_part) == 0:
        print('fail to find %s..' % id)
        return

    config = ConfigManager().load()
    jpg_root = config["jpg_root"]
    roi_root = config["roi_root"]

    v_lst, r_lst = [], []
    for idx, row in roi_info_part.iterrows():
        name = row.values[0]
        jpg_path = '%s/%s.jpg' % (jpg_root, name)
        roi_path = '%s/%s.jpg' % (roi_root, name)
        if not os.path.exists(roi_path):
            continue

        img = np.array(Image.open(jpg_path))[:,:,0]
        roi = np.array(Image.open(roi_path))[:,:,0]

        v_lst.append(img)
        r_lst.append(roi)

    volume, roi_volume = np.array(v_lst), np.array(r_lst)

    StackViewer(volume, roi_volume).show()


def export_attention_map():
    result_dir = './result/attention'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # weight_dir = './data/220603/d2_popf/best'
    weight_dir = './data/220603/d2_cr-popf/best'
    #net_type = NetTypes2D.Resnet50
    for net_type in NetTypes2D:
        print('now net type : %s' % net_type.value)
        weight_path = '%s/%s.pt' % (weight_dir, net_type.value)

        dl_manager = DeepLearningManager(result_dir, net_type)
        dl_manager.export_attention_map(weight_path)


if __name__ == '__main__':
    print('POPF prj 220524')
    #export_attention_map()
    # merge_attention_map()
    test_dataset()
