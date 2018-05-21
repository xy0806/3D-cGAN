from __future__ import division
import os
import time
from glob import glob
import cv2
import scipy.ndimage
from ops import *
from utils import *
from seg_eval import *


class cgan_unet_xy(object):
    """ Implementation of 3D U-net"""
    def __init__(self, sess, param_set):
        self.sess           = sess
        self.phase          = param_set['phase']
        self.batch_size     = param_set['batch_size']
        self.inputI_size    = param_set['inputI_size']
        self.inputI_chn     = param_set['inputI_chn']
        self.outputI_size   = param_set['outputI_size']
        self.output_chn     = param_set['output_chn']
        self.resize_r       = param_set['resize_r']
        self.pad_w          = param_set['pad_w']
        self.traindata_dir  = param_set['traindata_dir']
        self.chkpoint_dir   = param_set['chkpoint_dir']
        self.lr             = param_set['learning_rate']
        self.beta1          = param_set['beta1']
        self.epoch          = param_set['epoch']
        self.model_name     = param_set['model_name']
        self.save_intval    = param_set['save_intval']
        self.testdata_dir   = param_set['testdata_dir']
        self.labeling_dir   = param_set['labeling_dir']
        self.ovlp_ita       = param_set['ovlp_ita']

        self.rename_map     = param_set['rename_map']
        self.rename_map     = [int(s) for s in self.rename_map.split(',')]

        self.L1_lambda      = param_set['L1_lambda']

        # build model graph
        self.build_cgan_model()

    # build 3d unet based cgan graph
    def build_cgan_model(self):
        # input
        self.real_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_size, self.inputI_chn], name='inputI')
        self.real_label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_size], name='target')
        self.real_label_flt = tf.cast(self.real_label, dtype=tf.float32, name='target_float')
        self.real_label_chn = tf.reshape(self.real_label_flt, [self.batch_size, self.inputI_size, self.inputI_size, self.inputI_size, 1], name='target_flt_reshape')
        # unet as generator
        # self.main_prob, self.fake_label, self.aux0_prob, self.aux1_prob = self.unet_3D_model(self.real_I)
        self.main_prob, self.fake_label, self.aux0_prob, self.aux1_prob, self.aux2_prob = self.unet_3D_model(self.real_I)
        self.fake_label_flt = tf.cast(self.fake_label, dtype=tf.float32, name='fake_label_float')
        self.fake_label_chn = tf.reshape(self.fake_label_flt, [self.batch_size, self.inputI_size, self.inputI_size, self.inputI_size, 1], name='fake_label_reshape')
        # build pairs
        self.real_pair = tf.concat([self.real_I, self.real_label_chn], axis=4)
        self.fake_pair = tf.concat([self.real_I, self.fake_label_chn], axis=4)
        # discrimination
        self.D_r, self.D_r_logits = self.discriminator(self.real_pair, reuse=False)
        self.D_f, self.D_f_logits = self.discriminator(self.fake_pair, reuse=True)
        # ====== loss
        # === generator
        self.g2d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_f), logits=self.D_f_logits)) + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_label_flt - self.fake_label_flt))
        # unet loss with deep supervision
        self.g2g_main_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.real_label, logits=self.main_prob, name='main_loss'))
        self.g2g_aux0_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.real_label, logits=self.aux0_prob, name='aux0_loss'))
        self.g2g_aux1_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.real_label, logits=self.aux1_prob, name='aux1_loss'))
        self.g2g_loss = self.g2g_main_loss + tf.constant(0.5)*self.g2g_aux0_loss + tf.constant(0.5)*self.g2g_aux1_loss
        self.g_loss = self.g2d_loss + self.g2g_loss
        # === discriminator
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_r), logits=self.D_r_logits))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_f), logits=self.D_f_logits))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()
        # self.g_vars = [var for var in t_vars if 'unet3D_model' in var.name and 'gamma:0' not in var.name]
        self.g_vars = [var for var in t_vars if 'discriminator' not in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name and 'discriminator' in var.name]
        # model saver
        self.saver = tf.train.Saver()
        #
        self.saver_unet = tf.train.Saver(self.g_vars)

    # 3D unet graph
    def unet_3D_model(self, inputI):
        """3D U-net"""
        phase_flag = (self.phase =='train')
        concat_dim = 4
        # with tf.variable_scope("unet3D_model") as scope:
        # down-sampling path
        # compute down-sample path in gpu0
        with tf.device("/gpu:0"):
            # conv1_1 = conv_bn_relu(input=inputI, output_chn=64, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv1')
            conv1_1 = conv3d(input=inputI, output_chn=64, kernel_size=3, stride=1, use_bias=False, name='conv1')
            conv1_bn = tf.contrib.layers.batch_norm(conv1_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv1_batch_norm")
            conv1_relu = tf.nn.relu(conv1_bn, name='conv1_relu')
            pool1 = tf.layers.max_pooling3d(inputs=conv1_relu, pool_size=2, strides=2, name='pool1')
            #
            # conv2_1 = conv_bn_relu(input=pool1, output_chn=128, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv2')
            conv2_1 = conv3d(input=pool1, output_chn=128, kernel_size=3, stride=1, use_bias=False, name='conv2')
            conv2_bn = tf.contrib.layers.batch_norm(conv2_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv2_batch_norm")
            conv2_relu = tf.nn.relu(conv2_bn, name='conv2_relu')
            pool2 = tf.layers.max_pooling3d(inputs=conv2_relu, pool_size=2, strides=2, name='pool2')
            #
            # conv3_1 = conv_bn_relu(input=pool2, output_chn=256, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv3a')
            conv3_1 = conv3d(input=pool2, output_chn=256, kernel_size=3, stride=1, use_bias=False, name='conv3a')
            conv3_1_bn = tf.contrib.layers.batch_norm(conv3_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv3_1_batch_norm")
            conv3_1_relu = tf.nn.relu(conv3_1_bn, name='conv3_1_relu')
            # conv3_2 = conv_bn_relu(input=conv3_1, output_chn=256, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv3b')
            conv3_2 = conv3d(input=conv3_1_relu, output_chn=256, kernel_size=3, stride=1, use_bias=False, name='conv3b')
            conv3_2_bn = tf.contrib.layers.batch_norm(conv3_2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv3_2_batch_norm")
            conv3_2_relu = tf.nn.relu(conv3_2_bn, name='conv3_2_relu')
            pool3 = tf.layers.max_pooling3d(inputs=conv3_2_relu, pool_size=2, strides=2, name='pool3')
            #
            # conv4_1 = conv_bn_relu(input=pool3, output_chn=512, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv4a')
            conv4_1 = conv3d(input=pool3, output_chn=512, kernel_size=3, stride=1, use_bias=False, name='conv4a')
            conv4_1_bn = tf.contrib.layers.batch_norm(conv4_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv4_1_batch_norm")
            conv4_1_relu = tf.nn.relu(conv4_1_bn, name='conv4_1_relu')
            # conv4_2 = conv_bn_relu(input=conv4_1, output_chn=512, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv4b')
            conv4_2 = conv3d(input=conv4_1_relu, output_chn=512, kernel_size=3, stride=1, use_bias=False, name='conv4b')
            conv4_2_bn = tf.contrib.layers.batch_norm(conv4_2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv4_2_batch_norm")
            conv4_2_relu = tf.nn.relu(conv4_2_bn, name='conv4_2_relu')
            pool4 = tf.layers.max_pooling3d(inputs=conv4_2_relu, pool_size=2, strides=2, name='pool4')
            #
            conv5_1 = conv_bn_relu(input=pool4, output_chn=512, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='conv5_1')
            conv5_2 = conv_bn_relu(input=conv5_1, output_chn=512, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='conv5_2')

        # up-sampling path
        # compute up-sample path in gpu1
        with tf.device("/gpu:1"):
            deconv1_1 = deconv_bn_relu(input=conv5_2, output_chn=512, is_training=phase_flag, name='deconv1_1')
            #
            concat_1 = tf.concat([deconv1_1, conv4_2], axis=concat_dim, name='concat_1')
            deconv1_2 = conv_bn_relu(input=concat_1, output_chn=256, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='deconv1_2')
            deconv2_1 = deconv_bn_relu(input=deconv1_2, output_chn=256, is_training=phase_flag, name='deconv2_1')
            #
            concat_2 = tf.concat([deconv2_1, conv3_2], axis=concat_dim, name='concat_2')
            deconv2_2 = conv_bn_relu(input=concat_2, output_chn=128, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='deconv2_2')
            deconv3_1 = deconv_bn_relu(input=deconv2_2, output_chn=128, is_training=phase_flag, name='deconv3_1')
            #
            concat_3 = tf.concat([deconv3_1, conv2_1], axis=concat_dim, name='concat_3')
            deconv3_2 = conv_bn_relu(input=concat_3, output_chn=64, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='deconv3_2')
            deconv4_1 = deconv_bn_relu(input=deconv3_2, output_chn=64, is_training=phase_flag, name='deconv4_1')
            #
            concat_4 = tf.concat([deconv4_1, conv1_1], axis=concat_dim, name='concat_4')
            deconv4_2 = conv_bn_relu(input=concat_4, output_chn=32, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='deconv4_2')
            # predicted probability
            pred_prob = conv3d(input=deconv4_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='pred_prob')

            # ======================
            # auxiliary prediction 0
            aux0_conv = conv3d(input=deconv1_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='aux0_conv')
            aux0_deconv_1 = Deconv3d(input=aux0_conv, output_chn=self.output_chn, name='aux0_deconv_1')
            aux0_deconv_2 = Deconv3d(input=aux0_deconv_1, output_chn=self.output_chn, name='aux0_deconv_2')
            aux0_prob = Deconv3d(input=aux0_deconv_2, output_chn=self.output_chn, name='aux0_prob')
            # auxiliary prediction 1
            aux1_conv = conv3d(input=deconv2_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='aux1_conv')
            aux1_deconv_1 = Deconv3d(input=aux1_conv, output_chn=self.output_chn, name='aux1_deconv_1')
            aux1_prob = Deconv3d(input=aux1_deconv_1, output_chn=self.output_chn, name='aux1_prob')
            # auxiliary prediction 2
            aux2_conv = conv3d(input=deconv3_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='aux2_conv')
            aux2_prob = Deconv3d(input=aux2_conv, output_chn=self.output_chn, name='aux2_prob')

        with tf.device("/cpu:0"):
            # predicted labels
            soft_prob = tf.nn.softmax(pred_prob, name='pred_soft')
            pred_label = tf.argmax(soft_prob, axis=4, name='argmax')

        return pred_prob, pred_label, aux0_prob, aux1_prob, aux2_prob

    # network as discriminator
    def discriminator(self, im_pair, reuse=False):
        phase_flag = (self.phase == 'train')

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = conv_bn_relu(input=im_pair, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='d_h0_conv')
            h1 = conv_bn_relu(input=h0, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='d_h1_conv')
            h2 = conv_bn_relu(input=h1, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='d_h2_conv')
            h3 = conv_bn_relu(input=h2, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='d_h3_conv')
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    # train function
    def train(self):
        """Train 3D U-net"""
        d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        # initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # =============
        self.initialize_unet()
        # =============

        # save .log
        self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # load volume files
        # modalities
        # pair_list = glob('{}/*.nii.gz'.format(self.traindata_dir))
        # pair_list.sort()

        pair_list = []
        for p in range(150):
            img_path = os.path.join(self.traindata_dir, (str(p) + '.nii'))
            gt_path = os.path.join(self.traindata_dir, (str(p) + '_seg.nii'))
            pair_list.append(img_path)
            pair_list.append(gt_path)

        # a_img_clec, a_label_clec = load_data_pairs(a_pair_list, self.resize_r, self.rename_map)
        img_clec, label_clec = load_data_pairs_padding(pair_list, self.resize_r, self.rename_map, pad_w=self.pad_w)

        # temporary file to save loss
        loss_log = open("loss.txt", "w")

        all_loss = []
        for epoch in np.arange(self.epoch):
            start_time = time.time()
            # train batch
            batch_img, batch_label = get_batch_patches(img_clec, label_clec, self.inputI_size, self.batch_size, chn=1, flip_flag=True, rot_flag=True)
            # ==================
            # Update D network
            self.sess.run([d_optim], feed_dict={self.real_I: batch_img, self.real_label: batch_label})
            # Update G network
            self.sess.run([g_optim], feed_dict={self.real_I: batch_img, self.real_label: batch_label})
            # # Update G network
            # self.sess.run([g_optim], feed_dict={self.real_I: batch_img, self.real_label: batch_label})
            # # Update G network to make sure that d_loss does not go to zero
            # self.sess.run([g_optim], feed_dict={self.real_I: batch_img, self.real_label: batch_label})
            # ==================
            # errors
            errD_fake   = self.d_loss_fake.eval({self.real_I: batch_img})
            errD_real   = self.d_loss_real.eval({self.real_I: batch_img, self.real_label: batch_label})
            errG        = self.g_loss.eval({self.real_I: batch_img, self.real_label: batch_label})
            err_unet    = self.g2g_loss.eval({self.real_I: batch_img, self.real_label: batch_label})

            counter += 1
            print("============")
            print("Epoch: [%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, unet_loss: %.8f" % (epoch, time.time() - start_time, errD_fake + errD_real, errG, err_unet))

            all_loss.append([errD_fake + errD_real, errG, err_unet])
            # record error
            with open("cgan_err.txt", 'wb') as err_fid:
                np.savetxt(err_fid, all_loss, fmt='%s')

            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)

            # validation batch
            batch_val_img, batch_val_label = get_batch_patches(img_clec, label_clec, self.inputI_size, self.batch_size, chn=1, flip_flag=True, rot_flag=True)
            # current and validation loss
            # cur_valid_loss = self.g2g_loss.eval({self.real_I: batch_val_img, self.real_label: batch_val_label})
            cube_label = self.sess.run(self.fake_label, feed_dict={self.real_I: batch_val_img})
            print np.unique(batch_label)
            print np.unique(cube_label)
            # dice value
            dice_c = []
            for c in range(self.output_chn):
                ints = np.sum(((batch_val_label[0,:,:,:]==c)*1)*((cube_label[0,:,:,:]==c)*1))
                union = np.sum(((batch_val_label[0,:,:,:]==c)*1) + ((cube_label[0,:,:,:]==c)*1)) + 0.0001
                dice_c.append((2.0*ints)/union)
            print dice_c

        loss_log.close()

    # test the model
    def test(self):
        """Test 3D U-net"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # get file list of testing dataset
        test_list = glob('{}/*.nii.gz'.format(self.testdata_dir))
        test_list.sort()

        test_list = []
        for p in range(56):
            img_path = os.path.join(self.testdata_dir, (str(p) + '.nii'))
            test_list.append(img_path)

        # test
        for k in range(0, len(test_list)):
            print "processing No. %d volume..." % k
            # load the volume
            vol_file = nib.load(test_list[k])
            ref_affine = vol_file.affine
            # get volume data
            vol_data = vol_file.get_data().copy()
            resize_dim = (np.array(vol_data.shape) * self.resize_r).astype('int')
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)
            # padding
            vol_rz_pad = np.lib.pad(vol_data_resz, ((self.pad_w, self.pad_w), (self.pad_w, self.pad_w), (self.pad_w, self.pad_w)), 'constant',
                                    constant_values=np.array(((0, 0), (0, 0), (0, 0))))
            vol_pad_dim = vol_rz_pad.shape

            # normalization
            vol_rz_pad = vol_rz_pad.astype('float32')
            vol_rz_pad = vol_rz_pad / 255.0

            # decompose volume into list of cubes
            cube_list = decompose_vol2cube(vol_rz_pad, self.batch_size, self.inputI_size, self.inputI_chn, self.ovlp_ita)
            # predict on each cube
            cube_label_list = []
            for c in range(len(cube_list)):
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp

                cube_label = self.sess.run(self.fake_label, feed_dict={self.real_I: cube2test_norm})
                cube_label_list.append(cube_label)
                # print np.unique(cube_label)
            # compose cubes into a volume
            composed_orig = compose_label_cube2vol(cube_label_list, vol_pad_dim, self.inputI_size, self.ovlp_ita, self.output_chn)
            composed_label = np.zeros(composed_orig.shape, dtype='int16')
            # rename label
            for i in range(len(self.rename_map)):
                composed_label[composed_orig == i] = self.rename_map[i]
            composed_label = composed_label.astype('int16')

            # remove padding
            composed_label = composed_label[self.pad_w:self.pad_w+resize_dim[0], self.pad_w:self.pad_w+resize_dim[1], self.pad_w:self.pad_w+resize_dim[2]]
            print np.unique(composed_label)

            # for s in range(composed_label.shape[2]):
            #     cv2.imshow('volume_seg', np.concatenate(((vol_data_resz[:, :, s]).astype('uint8'), (composed_label[:, :, s]/4).astype('uint8')), axis=1))
            #     cv2.waitKey(30)

            # save predicted label
            composed_label_resz = resize(composed_label, vol_data.shape, order=0, preserve_range=True)
            composed_label_resz = composed_label_resz.astype('int16')

            labeling_path = os.path.join(self.labeling_dir, ('test_' + str(k) + '.nii.gz'))
            labeling_vol = nib.Nifti1Image(composed_label_resz, ref_affine)
            nib.save(labeling_vol, labeling_path)

    # test function for cross validation
    def test4crsv(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # get file list of testing dataset
        test_list = glob('{}/*.nii.gz'.format(self.testdata_dir))
        test_list.sort()

        # all dice
        all_dice = np.zeros([int(len(test_list)/2), 8])

        # test
        for k in range(2, len(test_list), 2):
            # load the volume
            vol_file = nib.load(test_list[k])
            ref_affine = vol_file.affine
            # get volume data
            vol_data = vol_file.get_data().copy()
            resize_dim = (np.array(vol_data.shape) * self.resize_r).astype('int')
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)
            # normalization
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0
            # padding
            vol_rz_pad = np.lib.pad(vol_data_resz, ((self.pad_w, self.pad_w), (self.pad_w, self.pad_w), (self.pad_w, self.pad_w)), 'constant',
                                      constant_values=np.array(((0, 0), (0, 0), (0, 0))))
            vol_pad_dim = vol_rz_pad.shape

            # decompose volume into list of cubes
            cube_list = decompose_vol2cube(vol_rz_pad, self.batch_size, self.inputI_size, self.inputI_chn, self.ovlp_ita)
            # predict on each cube
            cube_label_list = []
            for c in range(len(cube_list)):
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp

                cube_label = self.sess.run(self.fake_label, feed_dict={self.real_I: cube2test_norm})
                cube_label_list.append(cube_label)
                # print np.unique(cube_label)
            # compose cubes into a volume
            composed_orig = compose_label_cube2vol(cube_label_list, vol_pad_dim, self.inputI_size, self.ovlp_ita, self.output_chn)
            composed_label = np.zeros(composed_orig.shape, dtype='int16')
            # rename label
            for i in range(len(self.rename_map)):
                composed_label[composed_orig == i] = self.rename_map[i]
            composed_label = composed_label.astype('int16')

            # remove padding
            composed_label = composed_label[self.pad_w:self.pad_w+resize_dim[0], self.pad_w:self.pad_w+resize_dim[1], self.pad_w:self.pad_w+resize_dim[2]]
            print np.unique(composed_label)

            for s in range(composed_label.shape[2]):
                cv2.imshow('volume_seg', np.concatenate(((vol_data_resz[:, :, s]*255.0).astype('uint8'), (composed_label[:, :, s]/4).astype('uint8')), axis=1))
                cv2.waitKey(30)

            # save predicted label
            composed_label_resz = resize(composed_label, vol_data.shape, order=0, preserve_range=True)
            composed_label_resz = composed_label_resz.astype('int16')

            labeling_path = os.path.join(self.labeling_dir, ('test_' + str(k) + '.nii.gz'))
            labeling_vol = nib.Nifti1Image(composed_label_resz, ref_affine)
            nib.save(labeling_vol, labeling_path)

            # evaluation
            gt_file = nib.load(test_list[k + 1])
            gt_label = gt_file.get_data().copy()
            k_dice_c = seg_eval_metric(composed_label_resz, gt_label)
            print k_dice_c
            all_dice[int(k/2), :] = np.asarray(k_dice_c)

        mean_dice = np.mean(all_dice, axis=0)
        print "average dice: "
        print mean_dice

    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s_%s" % (self.batch_size, self.outputI_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.batch_size, self.outputI_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    # load pre-trained unet
    def initialize_unet(self):
        checkpoint_dir = '/media/xinyang/echo2/tmi17_pkg/code/GAN/FCN_cGAN_3D_full/3d_unet/outcome/model/pre-train_unet/1_80'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_unet.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))