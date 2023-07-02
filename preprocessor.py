import os

from multiprocessing import Pool
from contextlib import closing
from random import randint
import traceback

import numpy as np
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut
import matplotlib.pyplot as plt
import matplotlib.image
import scipy.ndimage as sci_img
import matplotlib
import cv2
from scipy.interpolate import interp1d
import pandas as pd
import nibabel as nib
from shutil import copyfile

from common import *


def parse_header(dcm):
    header = dict()
    for name in dcm.trait_names():
        if name == 'PixelData' or '_' in name :
            continue
        try:
            a = dcm[name]
        except KeyError as E:
            continue

        header[name] = dcm[name].value
    return header


def interpolate_2d(dcm_header, pixel_array, trg_spacing=[0.5, 0.5]):
    if 'PixelSpacing' not in dcm_header:
        print('PixelSpacing has not found!')
        return pixel_array

    pixel_space = dcm_header['PixelSpacing']
    now_spacing = np.array(pixel_space, dtype=np.float32)

    resize_factor = now_spacing / trg_spacing
    new_real_shape = pixel_array.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / pixel_array.shape

    # cubic spline interpolation (order=3)
    new_pixel = sci_img.interpolation.zoom(pixel_array, real_resize_factor, mode='nearest')
    return new_pixel

def find_body(matrix):
    shape = matrix.shape
    flag_matrix = np.zeros(shape)
    x, y = shape[0] // 2, shape[1] // 2
    min_val = np.min(matrix)

    queue = [(x, y)]
    cnt, iter = 0, 0
    while cnt == 0:
        while len(queue) > 0:
            x, y = queue.pop(0)
            if matrix[x, y] > min_val:
                flag_matrix[x, y] = 1
                cnt += 1
                if x+1 < shape[0] and flag_matrix[x+1, y] == 0:
                    flag_matrix[x + 1, y] = -1 if flag_matrix[x + 1, y] != 1 else 1
                    queue.append((x+1, y))
                if y+1 < shape[1] and flag_matrix[x, y+1] == 0:
                    flag_matrix[x, y+1] = -1 if flag_matrix[x, y+1] != 1 else 1
                    queue.append((x, y+1))
                if x-1 >= 0 and flag_matrix[x-1, y] == 0:
                    flag_matrix[x-1, y] = -1 if flag_matrix[x-1, y] != 1 else 1
                    queue.append((x-1, y))
                if y-1 >= 0 and flag_matrix[x, y-1] == 0:
                    flag_matrix[x, y-1] = -1 if flag_matrix[x, y-1] != 1 else 1
                    queue.append((x, y-1))

        if cnt < 10000:
            x = shape[0] // 2 + randint(-150, 150)
            y = shape[1] // 2 + randint(-150, 150)
            queue = [(x, y)]
            flag_matrix = np.zeros(shape)
            cnt = 0
        iter += 1
        if iter >= 100:
            break

    flag_matrix[flag_matrix == -1] = 0

    return flag_matrix

def find_big_obj(image_matrix, size_thres=100):
    obj2coord = dict()
    shape = image_matrix.shape
    obj_matrix = np.zeros((shape[0], shape[1]))
    obj_idx = 1

    for x in range(shape[0]):
        for y in range(shape[1]):
            if image_matrix[x, y] > 0:
                obj_lst = []

                if x + 1 < shape[0] and obj_matrix[x + 1, y] != 0:
                    obj_lst.append(obj_matrix[x + 1, y])
                if y + 1 < shape[1] and obj_matrix[x, y + 1] != 0:
                    obj_lst.append(obj_matrix[x, y+1])
                if x - 1 >= 0 and obj_matrix[x - 1, y] != 0:
                    obj_lst.append(obj_matrix[x-1, y])
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

    result_matrix = image_matrix.copy()
    for obj, coord_lst in obj2coord.items():
        if len(coord_lst) >= size_thres:
            continue

        for x, y in coord_lst:
            result_matrix[x, y] = 0

    return result_matrix

def remove_bowl(image):
    shape = image.shape
    image = find_big_obj(image, 10000)

    obj_len_lst = []
    obj_width = 0
    x_lst, y_lst = [], []
    for x_idx in range(shape[1]):
        state = 0
        start, end = -1, -1
        for y_idx in np.arange(shape[0] - 1, 0, -1):
            val = image[y_idx, x_idx]

            if state == 0 and val > 0:
                state = 1
                end = y_idx + 1
            elif state == 1 and val == 0:
                state = 2
                start = y_idx + 1
                break
        obj_len = end - start
        obj_len_lst.append(obj_len)
        obj_width = obj_width + 1 if state == 2 else obj_width

        if obj_len < 30:
            x_lst.append(x_idx)
            y_lst.append(end)

    med_height = np.median(obj_len_lst)

    if med_height < 30 or (med_height < 70 and obj_width == shape[1]):
        # print('height: %s, widht: %s/%s' % (med_height,obj_width,shape[1]))
        fnc = interp1d(x_lst, y_lst)
        new_x_lst = [x_lst[0]]

        for idx in range(1, len(x_lst)):
            old, now = x_lst[idx - 1], x_lst[idx]
            if now - old < 40:
                for x in range(old + 1, now):
                    new_x_lst.append(x)
            new_x_lst.append(now)

        new_y_lst = fnc(new_x_lst)

        result = image.copy()

        for idx in range(len(new_x_lst)):
            x_idx, y_end = new_x_lst[idx], int(new_y_lst[idx])
            y_start = int(y_end - (med_height + 3))
            if y_start < 0:
                continue
            for y_idx in range(y_start, shape[0]):
                result[y_idx, x_idx] = 0

            # result[y_end, x_idx, :] = 255
        result = find_big_obj(result, 10000)

        return result
    return image


def find_rectangle(matrix):
    y_histogram = np.sum(matrix, axis=1)
    x_histogram = np.sum(matrix, axis=0)

    y_min, y_max = -1, len(y_histogram)
    for y_idx in range(len(y_histogram)):
        if y_min < 0 and 0 < y_histogram[y_idx]:
            y_min = y_idx
        elif y_min >= 0 and 0 == y_histogram[y_idx] and y_idx - y_min > 400:
            y_max = y_idx
            break
        elif y_min >= 0 and 0 < y_histogram[y_idx]:
            y_max = y_idx + 1

    # print(y_histogram)

    x_min, x_max = -1, len(x_histogram)
    for x_idx in range(len(x_histogram)):
        if x_min < 0 and 0 < x_histogram[x_idx]:
            x_min = x_idx
        elif x_min >= 0 and 0 == x_histogram[x_idx] and x_idx - x_min > 400:
            x_max = x_idx
            break
        elif x_min >= 0 and 0 < x_histogram[x_idx]:
            x_max = x_idx + 1

    # 상하좌우 index를 찾음 단 max 의 경우 index+1 값임
    return y_min, y_max, x_min, x_max


def crop_image(image, mask=None, hu_lower=-200, size=(640, 896)):
    shape = image.shape
    # image = find_big_obj(image, 10000)
    flag_matrix = np.zeros((shape[0], shape[1]))
    for y_idx in range(shape[0]):
        for x_idx in range(shape[1]):
            if image[y_idx, x_idx] > 0:
                flag_matrix[y_idx, x_idx] = 1

    y_min, y_max, x_min, x_max = find_rectangle(flag_matrix)
    # if x_max - x_min == 1:
    #     print(y_min, y_max, x_min, x_max)
    #     y_min, y_max, x_min, x_max = find_rectangle(flag_matrix)

    sub_image = image[y_min:y_max, x_min:x_max]

    w = x_max - x_min
    h = y_max - y_min

    cropped = np.zeros((size[0], size[1]))

    cropped[:, :] = hu_lower
    y1 = (size[0] - h) // 2
    y2 = y1 + h
    x1 = (size[1] - w) // 2
    x2 = x1 + w
    cropped[y1:y2, x1:x2] = sub_image[:, :]

    if mask is not None:
        sub_mask = mask[y_min:y_max, x_min:x_max]
        new_mask =np.zeros((size[0], size[1]))
        new_mask[y1:y2, x1:x2] = sub_mask[:, :]
        return cropped, new_mask
    return cropped

def load_nii_file(nii_path):
    proxy = nib.load(nii_path)
    total_array = np.swapaxes(proxy.get_fdata(), 0, 2)
    mask_lst = []
    for si in np.arange(total_array.shape[0])[::-1]:
        mask_lst.append(total_array[si, :, :])
    #roi_volume = np.array(roi_volume)
    return mask_lst


class Preprocessor:
    PROCESS_NUM = 6
    SD_TAG = 'SeriesDescription'
    FILTER_SIZE = 512
    SD_TAG_LIST = ['abdomen_post  5.0  b30f', 'abdomen  5.0  b30f', 'post(ap)  5.0  b30f', 'post',
                   'portal  3.0  b30f', 'portal  3.0  b30f  rr', 'pacs', 'portal_phase  5.0  b50s', '/ce/fc01',
                   'portal  5.0  b30f'
                   ]
    ADD_NUM = 5

    def __init__(self, dcm_dir, nii_path, pp_root):
        self.dcm_dir = dcm_dir
        self.nii_path = nii_path
        self.pp_root = pp_root
        self.mask = None
        config = ConfigManager().load()
        self.hu_lower = int(config['hu_lower'])
        self.hu_upper = int(config['hu_upper'])
        self.l3_lower = int(config['l3_lower'])
        self.l3_upper = int(config['l3_upper'])

    def filter_dcm(self, path):
        is_filter = False

        dcm_file = pydicom.dcmread(path, stop_before_pixels=True)
        tag_lst = self.SD_TAG_LIST

        if self.SD_TAG in dcm_file:
            tag = str(dcm_file[self.SD_TAG].value)
            if tag.lower() not in tag_lst:
                is_filter = True

        if not is_filter:
            size = self.FILTER_SIZE
            dcm = pydicom.dcmread(path)
            pixel_array = dcm.pixel_array
            if pixel_array.shape[0] != size or pixel_array.shape[1] != size:
                is_filter = True

        if not is_filter:
            case_name = os.path.basename(self.dcm_dir)
            output_dir = '%s/%s' % (self.pp_root, case_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dcm_name = os.path.basename(path).split('.')[0]
            output_path = '%s/%s.dcm' % (output_dir, dcm_name)
            copyfile(path, output_path)

        return is_filter

    def batch_filter_dcm(self,  use_parallel=False):
        dcm_dir, nii_path, pp_root = self.dcm_dir, self.nii_path, self.pp_root
        case_name = os.path.basename(dcm_dir)

        dcm_file_lst = os.listdir(dcm_dir)

        path_lst = []
        for name in dcm_file_lst:
            if not name.endswith('.dcm'):
                continue

            path = '%s/%s' % (dcm_dir, name)
            path_lst.append(path)

        result_lst = []
        if use_parallel:
            with closing(Pool(self.PROCESS_NUM)) as p:
                result_lst = p.map(self.filter_dcm, path_lst)
        else:
            for path in path_lst:
                result = self.filter_dcm(path)
                result_lst.append(result)

        mask = load_nii_file(nii_path)
        mask_cnt = mask.shape[0]

        filter_cnt, ok_cnt = 0, 0
        for flag in result_lst:
            if flag:
                filter_cnt += 1
            else:
                ok_cnt += 1

        if mask_cnt != ok_cnt:
            print('..%s case have some problem: Mask %s vs Filtered %s' % (case_name, mask_cnt, ok_cnt))
            if ok_cnt == 0:
                output_dir = '%s/%s' % (self.pp_root, case_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if len(path_lst) == mask_cnt:
                    for path in path_lst:
                        dcm_name = os.path.basename(path).split('.')[0]
                        output_path = '%s/%s.dcm' % (output_dir, dcm_name)
                        copyfile(path, output_path)

                    print('...so %s files has been moved' % len(path_lst))

    def filter_path_with_l3(self, path_lst, l3_idx):
        temp_lst = []
        for path in path_lst:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            dcm_header = parse_header(dcm)
            id = int(dcm_header['InstanceNumber'])

            # tokens = os.path.basename(path).split('.')
            # id = int(tokens[-2])
            temp_lst.append([id, path])
        path_lst = list(np.array(sorted(temp_lst))[:, 1])

        if len(path_lst) <= l3_idx:
            print('!!....%s files and l3 : %s' % (len(path_lst), l3_idx))
            return path_lst

        l3_path = path_lst[(l3_idx-1)]
        dcm = pydicom.dcmread(l3_path, stop_before_pixels=True)
        dcm_header = parse_header(dcm)
        sbs, st = 'SpacingBetweenSlices', 'SliceThickness'
        tag = sbs
        if sbs not in dcm_header:
            tag = st
            if st not in dcm_header:
                return path_lst

        distance = dcm_header[tag]
        upper_bound = int(np.round(self.l3_upper / distance))
        start = l3_idx - 1 - upper_bound

        lower_bound = int(np.round(self.l3_lower / distance))
        end = l3_idx - 1 + lower_bound
        if end - start <= 0:
            return path_lst

        return path_lst[start:end]

    def batch_preprocess(self, l3_info, use_parallel=True):
        dcm_dir, nii_path, pp_root = self.dcm_dir, self.nii_path, self.pp_root
        case_name = os.path.basename(dcm_dir)
        now_info = l3_info.loc[l3_info['PID'] == int(case_name)]
        if now_info.shape[0] == 0:
            print('..%s has no l3 info.' % case_name)
            return

        pid, l3 = now_info.values[0]

        img_dir, msk_dir = '%s' % pp_root, '%s/mask' % pp_root
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        dcm_file_lst = os.listdir(dcm_dir)
        path_lst = []
        for name in dcm_file_lst:
            if not name.endswith('.dcm'):
                continue

            path = '%s/%s' % (dcm_dir, name)
            path_lst.append(path)
        # path_lst = self.filter_path_with_l3(path_lst, l3)

        result_lst = []
        if use_parallel:
            with closing(Pool(self.PROCESS_NUM)) as p:
                result_lst = p.map(self.preprocess, path_lst)
        else:
            for path in path_lst:
                result = self.preprocess(path)
                result_lst.append(result)

        print('..%s has been processed. (%s files)' % (case_name, sum(result_lst)))

    def preprocess(self, path):
        hu_lower, hu_upper = self.hu_lower, self.hu_upper
        dcm_dir, nii_path, pp_root = self.dcm_dir, self.nii_path, self.pp_root
        img_dir = '%s' % pp_root
        case_name = os.path.basename(dcm_dir)

        try:
            dcm = pydicom.dcmread(path)
            dcm_header = parse_header(dcm)

            # 현재 dicom파일의 instance number 불러오기
            instance_num = int(dcm_header['InstanceNumber'])
            output_path = '%s/%s_%s.jpg' % (img_dir, case_name, instance_num)

            if os.path.exists(output_path):
                return 0

            # HU 값으로 변경
            pixel_array = dcm.pixel_array
            hu_array = apply_modality_lut(pixel_array, dcm)

            # 2차원 보간 진행
            hu_array = interpolate_2d(dcm_header, hu_array)

            hu_array[hu_array < hu_lower] = hu_lower
            hu_array[hu_array > hu_upper] = hu_upper

            # 몸통 부분을 제외하고 지우기
            flag_matrix = find_body(hu_array)
            fi_array = hu_array.copy()
            for y_idx in range(fi_array.shape[0]):
                for x_idx in range(fi_array.shape[1]):
                    if flag_matrix[y_idx, x_idx] != 1:
                        fi_array[y_idx, x_idx] = hu_lower

            # 정해진 사이즈의 이미지로 복사하기
            fi_array = crop_image(fi_array, hu_lower=hu_lower)

            matplotlib.image.imsave(output_path, fi_array, vmin=hu_lower, vmax=hu_upper, cmap='gray')
        except Exception as E:
            print('..%s file had problem: %s\n%s' % (os.path.basename(path), E, traceback.format_exc()))
            return 0
        return 1


    def batch_preprocess_with_mask(self, use_parallel=True):
        dcm_dir, nii_path, pp_root = self.dcm_dir, self.nii_path, self.pp_root
        add_num = self.ADD_NUM
        case_name = os.path.basename(dcm_dir)

        img_dir, msk_dir = '%s/image' % pp_root, '%s/mask' % pp_root
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(msk_dir):
            os.makedirs(msk_dir)

        dcm_file_lst = os.listdir(dcm_dir)
        path_lst = []
        for name in dcm_file_lst:
            if not name.endswith('.dcm'):
                continue

            path = '%s/%s' % (dcm_dir, name)
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            dcm_header = parse_header(dcm)

            # 현재 dicom파일에 맞는 mask 정보 가져오기
            instance_num = int(dcm_header['InstanceNumber'])
            path_lst.append([instance_num, path])
        path_lst = np.array(sorted(path_lst))[:, 1]
        #print(path_lst)

        mask_lst = load_nii_file(nii_path)
        selected_indexes = []
        for m_idx in range(len(mask_lst)):
            mask = mask_lst[m_idx]
            if np.max(mask) > 0:
                selected_indexes.append(m_idx)
                mask[mask == 3.] = 1.
        min_idx, max_idx = np.min(selected_indexes), np.max(selected_indexes) + 1
        s_idx = min_idx - add_num if min_idx - add_num >= 0 else 0
        e_idx = max_idx + add_num if max_idx + add_num <= len(mask_lst) else len(mask_lst)
        # selected_indexes.extend(list(range(s_idx, min_idx)))
        # selected_indexes.extend(list(range(max_idx, e_idx)))
        selected_indexes = list(range(s_idx, e_idx))

        if len(path_lst) != len(mask_lst):
            msg = '..%s path %s, mask %s' % (case_name, len(path_lst), len(mask_lst))
            print(msg)
            return

        info_lst = [[idx, path_lst[idx], mask_lst[idx]] for idx in selected_indexes]
        result_lst = []
        if use_parallel:
            with closing(Pool(self.PROCESS_NUM)) as p:
                result_lst = p.map(self.preprocess_with_mask, info_lst)
        else:
            for info in info_lst:
                result = self.preprocess_with_mask(info)
                result_lst.append(result)

        # img_lst, mask_lst = [], []
        # for img, mask in result_lst:
        #     img_lst.append(img)
        #     mask_lst.append(mask)
        # volume = np.array(img_lst)
        # mask = np.array(mask_lst)
        # StackViewer(volume, roi_volume=mask).show()

        print('..%s has been processed. (%s files)' % (case_name, sum(result_lst)))


    def preprocess_with_mask(self, info):
        hu_lower, hu_upper = self.hu_lower, self.hu_upper
        dcm_dir, nii_path, pp_root = self.dcm_dir, self.nii_path, self.pp_root
        num, path, now_mask = info

        case_name = os.path.basename(dcm_dir)
        img_dir, msk_dir = '%s/image' % pp_root, '%s/mask' % pp_root
        output_path = '%s/%s_%s.jpg' % (img_dir, case_name, num)
        if os.path.exists(output_path):
            return 0

        try:
            # dcm file 불러오기
            dcm = pydicom.dcmread(path)
            dcm_header = parse_header(dcm)

            # HU 값으로 변경
            pixel_array = dcm.pixel_array
            hu_array = apply_modality_lut(pixel_array, dcm)

            # 2차원 보간 진행
            hu_array = interpolate_2d(dcm_header, hu_array)
            new_mask = interpolate_2d(dcm_header, now_mask)

            hu_array[hu_array < hu_lower] = hu_lower
            hu_array[hu_array > hu_upper] = hu_upper

            # 몸통 부분을 제외하고 지우기
            flag_matrix = find_body(hu_array)
            fi_array = hu_array.copy()
            for y_idx in range(fi_array.shape[0]):
                for x_idx in range(fi_array.shape[1]):
                    if flag_matrix[y_idx, x_idx] != 1:
                        fi_array[y_idx, x_idx] = hu_lower

            # 정해진 사이즈의 이미지로 복사하기
            fi_array, new_mask = crop_image(fi_array, new_mask, hu_lower)

            # 결과 이미지, 마스크 출력
            matplotlib.image.imsave(output_path, fi_array, vmin=hu_lower, vmax=hu_upper, cmap='gray')
            mssk_path = '%s/%s_%s.jpg' % (msk_dir, case_name, num)
            matplotlib.image.imsave(mssk_path, new_mask, vmin=0, vmax=1, cmap='gray')
        except Exception as E:
            print('...Error on %s: %s\n%s' % (path, E, traceback.format_exc()))
            return 0

        return 1



class CopyMaster:
    PROCESS_NUM = 8
    INFO_PATH = './config/folder_info.csv'

    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root
        self.info_df = pd.read_csv(self.INFO_PATH)

    def run(self, use_parallel=True):
        input_root, output_root, info_df = self.input_root, self.output_root, self.info_df
        n1, n2 = os.path.basename(input_root), os.path.basename(output_root)
        print('Start Copy dcm files from [%s] to [%s]' % (n1, n2))
        folder_lst = os.listdir(input_root)

        path_lst = []
        for name in folder_lst:
            now_dir = '%s/%s' % (input_root, name)
            if not os.path.isdir(now_dir):
                continue
            try:
                now_id = int(name)
            except ValueError:
                now_id = int(name[:4])

            info = info_df.loc[info_df['PID'] == now_id].values
            if len(info) == 0:
                # print('..there are no directory info %s' % now_id)
                continue

            pid, dir_name = info[0]

            input_dir = '%s/%s' % (now_dir, dir_name)
            if not os.path.exists(input_dir):
                print('..fail to find! %s case %s directory' % (pid, dir_name))

            output_dir = '%s/%s' % (output_root, pid)
            path_lst.append([input_dir, output_dir])

        if use_parallel:
            with closing(Pool(self.PROCESS_NUM)) as p:
                p.map(self.copy, path_lst)
        else:
            for path in path_lst:
                self.copy(path)

    def copy(self, args):
        input_dir, output_dir = args
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        name_lst = os.listdir(input_dir)
        cnt = 0
        for name in name_lst:
            if not name.endswith('.dcm'):
                continue
            input_path = '%s/%s' % (input_dir, name)
            output_path = '%s/%s' % (output_dir, name)
            if os.path.exists(output_path):
                continue
            cnt += 1
            copyfile(input_path, output_path)

        pid = os.path.basename(output_dir)
        print('....%s files of %s case has been moved' % (cnt, pid))


class MappingMaster:
    PROCESS_NUM = 8
    INFO_PATH = './config/neck_info_mod2.csv'

    def __init__(self, output_root, output_path):
        config = ConfigManager().load()
        self.input_root = config['dcm_root']
        self.output_root = output_root
        self.output_path = output_path
        self.info_df = pd.read_csv(self.INFO_PATH)

        self.hu_upper = int(config['hu_upper'])
        self.hu_lower = int(config['hu_lower'])

    def run(self, use_parallel=True):
        info_df = self.info_df

        args_lst = []
        for idx, row in info_df.iterrows():
            pid, z, x, y = row.values
            args_lst.append([pid, z, x, y])

        result_lst = []
        if use_parallel:
            with closing(Pool(self.PROCESS_NUM)) as p:
                result_lst = p.map(self.map_coordinate, args_lst)
        else:
            for args in args_lst:
                r = self.map_coordinate(args)
                result_lst.append(r)

        r_dict = {'pid':[], 'z':[], 'y':[], 'x':[]}
        for pid, z, y, x in result_lst:
            r_dict['pid'].append(pid)
            r_dict['z'].append(z)
            r_dict['y'].append(y)
            r_dict['x'].append(x)
        output_path = self.output_path
        pd.DataFrame(r_dict).to_csv(output_path, index=False)
        print('done!')

    def map_coordinate(self, args):
        input_root, output_root = self.input_root, self.output_root
        hu_lower, hu_upper = self.hu_lower, self.hu_upper
        pid, z, x, y = args
        n_x2, n_y2 = -1, -1
        dcm_dir = '%s/%s' % (input_root, pid)

        before_path = '%s/before/%s.png' % (output_root, pid)
        after_path = '%s/after/%s.png' % (output_root, pid)
        # if os.path.exists(output_path):
        #     return pid, z, n_y2, n_x2

        if not os.path.exists(dcm_dir):
            print('.. there are no dcm dir, pid: %s, path: %s' % (pid, dcm_dir))
            return pid, z, n_y2, n_x2

        name_lst = os.listdir(dcm_dir)
        path_lst = []
        for name in name_lst:
            if not name.endswith('.dcm'):
                continue
            path = '%s/%s' % (dcm_dir, name)

            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            dcm_header = parse_header(dcm)
            ist_num = int(dcm_header['InstanceNumber'])
            path_lst.append([ist_num, path])
        path_lst = sorted(path_lst)

        if z >= len(path_lst):
            print('.. !!path error occurred!! pid: %s, z: %s, size: %s' % (pid, z, len(path_lst)))
            return pid, z, n_y2, n_x2

        now_path = path_lst[z]

        dcm = pydicom.dcmread(now_path[1])
        dcm_header = parse_header(dcm)

        # HU 값으로 변경
        pixel_array = dcm.pixel_array
        hu_array = apply_modality_lut(pixel_array, dcm)

        # export the "before preprocess" image
        plt.imshow(hu_array, cmap='gray')
        plt.scatter(x, y, c='red')
        plt.axis('off')
        plt.savefig(before_path, bbox_inches='tight')
        plt.close()

        # 2차원 보간 진행
        hu_array = interpolate_2d(dcm_header, hu_array)

        n_x = int(np.round(x * hu_array.shape[1] / pixel_array.shape[1]))
        n_y = int(np.round(y * hu_array.shape[0] / pixel_array.shape[0]))

        hu_array[hu_array < hu_lower] = hu_lower
        hu_array[hu_array > hu_upper] = hu_upper

        # 몸통 부분을 제외하고 지우기
        flag_matrix = find_body(hu_array)
        fi_array = hu_array.copy()
        for y_idx in range(fi_array.shape[0]):
            for x_idx in range(fi_array.shape[1]):
                if flag_matrix[y_idx, x_idx] != 1:
                    fi_array[y_idx, x_idx] = hu_lower

        mask = np.zeros(fi_array.shape)
        mask[n_y, n_x] = 1

        # 정해진 사이즈의 이미지로 복사하기
        fi_array, mask = crop_image(fi_array, mask, hu_lower=hu_lower)

        try:
            n_x2 = [x for x in range(fi_array.shape[1]) if np.argmax(mask[:, x]) > 0][0]
            n_y2 = [y for y in range(fi_array.shape[0]) if np.argmax(mask[y, :]) > 0][0]
        except IndexError:
            print('.. !!index error occurred!! pid: %s, z:%s, x: %s, y: %s' % (pid, z, x, y))
            return pid, z, n_y2, n_x2

        # export the "before preprocess" image
        plt.imshow(fi_array, cmap='gray')
        plt.scatter(n_x2, n_y2, c='red')
        plt.axis('off')
        plt.savefig(after_path, bbox_inches='tight')
        plt.close()

        # print('...%s have new coordinate %s, %s, %s' % (pid, n_x2, n_y2, z))
        return pid, z, n_y2, n_x2


def crop_2d(image, centroid, crop_size):
    cy, cx = centroid
    th, tw = crop_size
    sh, sw = image.shape

    s_y1 = cy - th // 2
    s_y2 = cy + th // 2
    s_x1 = cx - tw // 2
    s_x2 = cx + tw // 2

    t_y1, t_x1 = 0, 0
    t_y2, t_x2 = th, tw

    if s_y1 < 0:
        gap = -s_y1
        s_y1 += gap
        t_y1 += gap
    if s_x1 < 0:
        gap = -s_x1
        s_x1 += gap
        t_x1 += gap
    if s_y2 > sh:
        gap = s_y2 - sh
        s_y2 -= gap
        t_y2 -= gap
    if s_x2 > sw:
        gap = s_x2 - sw
        s_x2 -= gap
        t_x2 -= gap

    # print(tl, th, tw)
    # print(t_z1, t_z2, t_y1, t_y2, t_x1, t_x2)
    # print(s_z1, s_z2, s_y1, s_y2, s_x1, s_x2)

    new_image = np.zeros((th, tw))
    new_image[t_y1:t_y2, t_x1:t_x2] = image[s_y1:s_y2, s_x1:s_x2]
    return new_image
