import os
import pydicom
import scipy.ndimage as sci_img
import numpy as np
import pandas as pd
import nibabel as nib
from pydicom.pixel_data_handlers import apply_modality_lut
from pydicom import config

from common import *

config.INVALID_KEY_BEHAVIOR = "IGNORE"


def collect_dcm_info(root_dir):
    dir_lst = os.listdir(root_dir)
    key_lst = ['Manufacturer', 'Rows', 'Columns', 'SliceThickness', 'PixelSpacing']

    result_dict = {'ID': [], 'slices': [], 'SeriesDescription': []}
    for key in key_lst:
        result_dict[key] = []

    for dir_name in dir_lst:
        now_direc = '%s/%s' % (root_dir, dir_name)
        data = DcmData(now_direc).load()

        header = data.get_header()
        result_dict['ID'].append(dir_name)
        result_dict['slices'].append(data.volume.shape[0])
        if header is None:
            for key in key_lst:
                result_dict[key].append(' ')
            continue

        for key in key_lst:
            if key in header:
                value = header[key]
            else:
                value = ''
            result_dict[key].append(value)
        sd_set = data.get_series_descriptions()
        sd_str = ' | '.join(sd_set)
        result_dict['SeriesDescription'].append(sd_str)

    output_path = './result/dcm_info.csv'
    pd.DataFrame(result_dict).to_csv(output_path, index=False)


class DcmData(object):
    SIZE = 512
    SD_TAG = 'SeriesDescription'

    def __init__(self, direc_path, load_now=False):
        self.volume = None
        self.roi_volume = None
        self.path_lst = None
        self.ref_dcm = None
        self.header = None
        self.direc_path = direc_path

        configuration = ConfigManager().load()
        self.NII_FMT = configuration['nii_fmt']

        path_lst = []
        for dirName, subdirList, fileList in os.walk(direc_path):
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    path = "%s/%s" % (dirName, filename)
                    path_lst.append(path)

        self.path_lst = path_lst
        if load_now:
            self.load()

    def select_slides(self, edge_margin=5, verbose=False):
        roi_volume = self.roi_volume
        volume = self.volume

        si, ei = -1, -1
        for roi_idx in range(roi_volume.shape[0]):
            mask = roi_volume[roi_idx]
            if np.nanmax(mask) > 0:
                si = roi_idx if si < 0 else si
                ei = roi_idx + 1
        si = si if si - edge_margin < 0 else si - edge_margin
        ei = ei if ei + edge_margin > roi_volume.shape[0] else ei + edge_margin

        self.roi_volume = roi_volume[si:ei, :, :]
        self.volume = volume[si:ei, :, :]
        if verbose:
            print('...slides have been selected %s->%s' % (volume.shape[0], self.volume.shape[0]))
        return self

    def filter_series(self, verbose=False):
        sd_set = self.get_series_descriptions()
        if len(sd_set) <= 1:
            return self
        pass_lst = ['Abdomen_post  5.0  B30f', 'Abdomen  5.0  B30f', 'POST(AP)  5.0  B30f', 'POST']

        new_path_lst = []
        for path in self.path_lst:
            dcm_file = pydicom.dcmread(path, stop_before_pixels=True)
            if self.SD_TAG in dcm_file:
                description = str(dcm_file[self.SD_TAG].value)
                if description in pass_lst:
                    new_path_lst.append(path)
        if verbose:
            print('... %s has been filtered %s->%s' % (self.direc_path, len(self.path_lst), len(new_path_lst)))
        self.path_lst = new_path_lst
        return self

    def load(self, verbose=False):
        slice_lst = []
        for path in self.path_lst:
            # name = os.path.basename(path).split('.')[0]
            dcm = pydicom.dcmread(path)
            pixel_array = dcm.pixel_array
            if pixel_array.shape[0] != self.SIZE or pixel_array.shape[1] != self.SIZE:
                continue

            hu_array = apply_modality_lut(pixel_array, dcm)
            slice_lst.append(hu_array)
        self.volume = np.array(slice_lst)
        case_id = os.path.basename(self.direc_path)
        if verbose:
            msg = '..%s: total of %s slices have been loaded' % (case_id, len(slice_lst))
            print(msg)

        nii_path = self.NII_FMT % case_id
        if os.path.exists(nii_path):
            # 1. Proxy 불러오기
            proxy = nib.load(nii_path)

            # 2. Header 불러오기
            # header = proxy.header
            # header_size = header['sizeof_hdr']

            # 3. 전체 Image Array 불러오기
            total_array = np.swapaxes(proxy.get_fdata(), 0, 2)
            roi_volume = []
            for si in np.arange(total_array.shape[0])[::-1]:
                roi_volume.append(total_array[si, :, :])
            roi_volume = np.array(roi_volume)

            self.roi_volume = roi_volume
            if verbose:
                msg = '..%s: ROI loading successful' % case_id
                print(msg)
        return self

    def get_header(self):
        if self.header is not None:
            return self.header

        if len(self.path_lst) == 0:
            print('%s- There are no dcm file..' % self.direc_path)
            return
        elif self.ref_dcm is None:
            self.ref_dcm = pydicom.dcmread(self.path_lst[0], stop_before_pixels=True)

        ref_dcm = self.ref_dcm
        header = dict()
        for name in ref_dcm.trait_names():
            if name == 'PixelData' or '_' in name or (name not in ref_dcm):
                continue
            # msg = 'key [%s] value [%s]' % (name, ref_dcm[name].value)
            # print(msg)
            header[name] = ref_dcm[name].value
        self.header = header
        return header

    def interpolate(self, new_spacing=[2.5, 0.5, 0.5], verbose=False):
        # Determine current pixel spacing
        header = self.get_header()
        volume = self.volume
        slice_thick, pixel_space = header['SliceThickness'],  header['PixelSpacing']
        now_spacing = np.array([slice_thick] + list(pixel_space), dtype=np.float32)

        resize_factor = now_spacing / new_spacing
        new_real_shape = volume.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / volume.shape
        # new_spacing = now_spacing / real_resize_factor

        new_volume = sci_img.interpolation.zoom(volume, real_resize_factor, mode='nearest')
        self.volume = new_volume

        if self.roi_volume is not None:
            new_roi = sci_img.interpolation.zoom(self.roi_volume, real_resize_factor, mode='nearest')
            self.roi_volume = np.round(new_roi).astype('int')
        if verbose:
            print(' interpolate volume %s to %s' % (volume.shape, new_volume.shape))
        return self

    def get_series_descriptions(self):
        sd_lst = []
        for path in self.path_lst:
            dcm_file = pydicom.dcmread(path, stop_before_pixels=True)
            if self.SD_TAG in dcm_file:
                value = dcm_file[self.SD_TAG].value
                sd_lst.append(str(value))
        sd_set = list(set(sd_lst))

        return sd_set