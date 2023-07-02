import matplotlib.pyplot as plt
import numpy as np
import configparser
import pickle
from shutil import copyfile

def save_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class ConfigManager:
    DEFAULT_PATH = './config/config_doai.ini'
    SECTION_DEFAULT = 'DEFAULT'
    config_path = None

    def __init__(self, file_name=DEFAULT_PATH):
        self.config_path = file_name

    def copy_file(self, output_path):
        copyfile(self.DEFAULT_PATH, output_path)

    def load(self):
        config_file = configparser.ConfigParser()
        config_file.read(self.config_path)

        return config_file[self.SECTION_DEFAULT]

    def save(self, config_dict, output_path=None):
        if output_path is None:
            output_path = self.config_path
        if output_path is None:
            return

        config_file = configparser.ConfigParser()

        for key in config_dict:
            val = None
            if type(config_dict[key]) == int:
                val = "%d" % config_dict[key]
            elif type(config_dict[key]) == float:
                val = "%f" % config_dict[key]
            elif type(config_dict[key]) == bool:
                if config_dict[key]:
                    val = 'True'
                else:
                    val = 'False'
            elif type(config_dict[key]) == str:
                val = config_dict[key]

            if '%s' in val:
                val = val.replace('%s', '%%s')

            if val:
                config_file[self.SECTION_DEFAULT][key] = val

        with open(output_path, 'w') as writeFile:
            config_file.write(writeFile)

    # 현재 설정 내용을 출력함
    def dump(self, config_dict):
        for key in config_dict:
            if type(config_dict[key]) == int:
                print("%s = %d[int]" % (key, config_dict[key]))
            elif type(config_dict[key]) == float:
                print("%s = %f[float]" % (key, config_dict[key]))
            elif type(config_dict[key]) == bool:
                if config_dict[key]:
                    print("%s = True[bool]" % key)
                else:
                    print("%s = False[bool]" % key)
            elif type(config_dict[key]) == str:
                print("%s = %s[str]" % (key, config_dict[key]))


class StackViewer(object):
    def __init__(self, volume, roi_volume=None, title='Abdomen Mask'):
        fig = plt.figure(figsize=(8, 8))
        self.roi_volume = None
        if roi_volume is None:
            idx = 0
            image = volume[idx, :, :]
            plt.imshow(image, cmap='gray', vmin=-100, vmax=200)
        else:
            is_find_mask = False
            mask_limit = 3
            while not is_find_mask:
                for idx in range(volume.shape[0]):
                    # roi_idx = volume.shape[0] - idx -1
                    mask = roi_volume[idx, :, :]
                    if np.max(mask) >= mask_limit:
                        image = volume[idx, :, :]
                        plt.imshow(image, cmap='gray')
                        #plt.imshow(mask, alpha=0.2, cmap='cubehelix')
                        plt.imshow(mask, alpha=0.2, cmap='Reds')
                        is_find_mask = True
                        break
                mask_limit -= 1
                if mask_limit < 0:
                    print('fail to find mask!')

            idx = 0
            if is_find_mask:
                ax = fig.axes[0]
                ax.images[0].set_array(volume[idx])
                ax.images[1].set_array(roi_volume[idx])
                self.roi_volume = roi_volume
            else:
                image = volume[idx, :, :]
                plt.imshow(image, cmap='gray')

        plt.title(title)
        fig.canvas.mpl_connect('key_press_event', self.process_key)

        self.volume = volume
        self.idx = idx

    def show(self):
        plt.show()

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        # print(event.key)

        if event.key == 'up':
            self.previous_slice(ax)
        elif event.key == 'down':
            self.next_slice(ax)
        fig.canvas.draw()

    def previous_slice(self, ax):
        if self.idx - 1 < 0:
            return

        volume = self.volume
        idx = (self.idx - 1) % volume.shape[0]
        ax.images[0].set_array(volume[idx])

        if self.roi_volume is not None:
            roi_volume = self.roi_volume
            #roi_idx = volume.shape[0] - idx - 1
            mask = roi_volume[idx]
            ax.images[1].set_array(mask)
        print('previous slice: %s' % idx)
        self.idx = idx

    def next_slice(self, ax):
        if self.idx + 1 >= self.volume.shape[0]:
            return

        volume = self.volume

        idx = (self.idx + 1) % volume.shape[0]
        ax.images[0].set_array(volume[idx])

        if self.roi_volume is not None:
            roi_volume = self.roi_volume
            mask = roi_volume[idx]
            ax.images[1].set_array(mask)
        print('next slice: %s' % idx)
        self.idx = idx