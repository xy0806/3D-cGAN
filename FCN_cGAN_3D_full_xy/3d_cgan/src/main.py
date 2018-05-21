import os
import tensorflow as tf

from ini_file_io import load_train_ini
from model import cgan_unet_xy


def main(_):

    # load training parameter #
    ini_file = '../outcome/model/ini/tr_param.ini'
    param_sets = load_train_ini(ini_file)
    param_set = param_sets[0]

    print '====== Phase >>> %s <<< ======' % param_set['phase']

    if not os.path.exists(param_set['chkpoint_dir']):
        os.makedirs(param_set['chkpoint_dir'])
    if not os.path.exists(param_set['labeling_dir']):
        os.makedirs(param_set['labeling_dir'])

    with tf.Session() as sess:
        model = cgan_unet_xy(sess, param_set)

        if param_set['phase'] == 'train':
            model.train()
        elif param_set['phase'] == 'test':
            model.test()
        elif param_set['phase'] == 'crsv':
            model.test4crsv()

if __name__ == '__main__':
    tf.app.run()
