import os
import time
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    pass


class Data:

    def __init__(self, batch_size, class_number, data_path):
        self.batch_size = batch_size
        self.class_number = class_number
        self.seed_batch_size = batch_size * self.class_number

        self._mnist = input_data.read_data_sets(data_path, reshape=False)
        self._data_train = self._mnist.train
        self._data_test = self._mnist.test

        self.number_train = self._data_train.num_examples // self.seed_batch_size
        self.number_test = self._data_test.num_examples // self.seed_batch_size
        self.data_size = 28
        pass

    def next_train_batch(self):
        train_data, _ = self._data_train.next_batch(self.seed_batch_size)
        train_data = self._deal_data(train_data, self.batch_size, self.class_number)
        return train_data

    def next_test_batch(self, index):
        start = 0 if index >= self.number_test else index * self.seed_batch_size
        end = self.seed_batch_size if index >= self.number_test else (index + 1) * self.seed_batch_size

        test_data, _ = self._data_test.images[start: end], self._data_test.labels[start: end]
        test_data = self._deal_data(test_data, self.batch_size, self.class_number)
        return test_data

    @staticmethod
    def _deal_data(train_data, batch_size, class_number):
        train_data_final = []
        for i in range(batch_size):
            now_data = np.concatenate(train_data[i * class_number: (i + 1) * class_number], axis=2)
            train_data_final.append(now_data)
            pass
        return train_data_final

    def get_batch_data(self, train_data=True, image_number=64):

        if train_data:
            images = self._data_train.images
            labels = self._data_train.labels
        else:
            images = self._data_test.images
            labels = self._data_test.labels
            pass

        if image_number is None:
            return images, labels
        return images[0: image_number], labels[0: image_number]

    def get_image_for_reconstruct(self):
        image = self._data_test.images[0: 10]
        image = np.reshape(image, newshape=[10, -1])
        return image

    pass


class Net:

    def __init__(self, batch_size, data_size, class_number):
        # 输入参数
        self.batch_size = batch_size
        self.data_size = data_size
        self.class_number = class_number

        self.graph = tf.Graph()

        # 输入
        self.x, self.x_ae, self.x_cnn, self.label_cnn = None, None, None, None
        # 网络输出
        self.hidden_ae, self.reconstruction_ae = None, None
        # 损失和训练
        self.loss, self.loss_1, self.loss_2, self.train_op = None, None, None, None

        with self.graph.as_default():
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_size,
                                                             self.data_size, self.class_number], name="x")

            self.x_ae1 = tf.split(self.x, num_or_size_splits=self.class_number, axis=3)
            self.x_ae1 = tf.squeeze(tf.concat(self.x_ae1, axis=0))
            self.x_ae = tf.reshape(self.x_ae1, shape=[-1, self.data_size * self.data_size])
            self.hidden_ae, self.reconstruction_ae = self.net_ae(self.data_size * self.data_size, 80)

            self.loss, self.loss_1, self.loss_2 = self.loss_example()
            self.train_op = self.train_op_example(learning_rate=0.001, loss_all=self.loss)
        pass

    def net_ae(self, n_input, n_hidden):
        weights = dict()
        weights["w1"] = tf.get_variable("w1", shape=[n_input, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
        weights["b1"] = tf.Variable(tf.zeros([n_hidden], dtype=tf.float32))
        weights["w2"] = tf.Variable(tf.zeros([n_hidden, n_input], dtype=tf.float32))
        weights["b2"] = tf.Variable(tf.zeros([n_input], dtype=tf.float32))

        # model
        hidden = tf.nn.softplus(tf.add(tf.matmul(self.x_ae, weights["w1"]), weights["b1"]))
        reconstruction = tf.add(tf.matmul(hidden, weights["w2"]), weights["b2"])
        return hidden, reconstruction

    # 损失
    def loss_example(self):
        # cost
        loss_1 = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction_ae, self.x_ae), 2.0))
        return loss_1, loss_1, loss_1

    # 训练节点
    @staticmethod
    def train_op_example(learning_rate, loss_all):
        return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(loss_all)

    pass


class Runner:

    def __init__(self, data, model_path="model"):
        self.data = data
        self.batch_size = self.data.batch_size
        self.class_number = self.data.class_number
        self.model_path = model_path

        # 网络
        self.net = Net(batch_size=self.batch_size, data_size=self.data.data_size, class_number=self.class_number)

        self.supervisor = tf.train.Supervisor(graph=self.net.graph, logdir=self.model_path)
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        pass

    # 训练
    def train(self, epochs=20, save_freq=2):
        with self.supervisor.managed_session(config=self.config) as sess:
            for epoch in range(epochs):
                # stop
                if self.supervisor.should_stop():
                    break
                # train
                loss_all = 0
                for step in range(self.data.number_train):
                    x = self.data.next_train_batch()

                    _, loss, x_ae, x_ae1, x_n = sess.run([self.net.train_op, self.net.loss,
                                                          self.net.x_ae, self.net.x_ae1, self.net.x],
                                                         feed_dict={self.net.x: x})
                    if step % 100 == 0:
                        # x_d = np.split(x[0], indices_or_sections=10, axis=2)
                        # x_n = np.split(x_n[0], indices_or_sections=10, axis=2)
                        #
                        # self.show_image(10, x_d[0: 10], x_n[0: 10])
                        # self.show_image(10, x_ae[0: 10], x_ae1[0: 10])
                        Tools.print_info("epoch={} step={} loss={}".format(epoch, step, loss))
                    loss_all += np.mean(loss)
                    pass
                loss_all = loss_all / self.data.number_train
                Tools.print_info("epoch={} loss={}".format(epoch, loss_all))
                # save
                if epoch % save_freq == 0:
                    self.supervisor.saver.save(sess, os.path.join(self.model_path, "model_epoch_{}".format(epoch)))
            pass
        pass

    def show_reconstruction(self):
        with self.supervisor.managed_session(config=self.config) as sess:
            image_data = self.data.get_image_for_reconstruct()
            reconstruction_ae = sess.run(self.net.reconstruction_ae, {self.net.x_ae: image_data})
            self.show_image(len(image_data), image_data, reconstruction_ae)

        pass

    def show_image(self, image_length, image_data, image_data2):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # 对比原始图片重建图片
        plt.figure(figsize=(image_length, 2))
        gs = gridspec.GridSpec(2, image_length)
        gs.update(wspace=0.05, hspace=0.05)
        for i in range(image_length):
            # 原始图片
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(np.reshape(image_data[i], (28, 28)))

            # 解码后的图
            ax = plt.subplot(gs[i + image_length])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(np.reshape(image_data2[i], (28, 28)))
            pass
        plt.show()
        pass

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="ae_just", help="name")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-class_number", type=int, default=10, help="type number")
    parser.add_argument("-data_path", type=str, default="./data/mnist", help="image data")
    args = parser.parse_args()

    output_param = "name={}batch_size={},class_number={},data_path={}"
    Tools.print_info(output_param.format(args.name, args.batch_size, args.class_number, args.data_path))

    runner = Runner(Data(batch_size=args.batch_size, class_number=args.class_number, data_path=args.data_path),
                    model_path=os.path.join("model", args.name))
    runner.show_reconstruction()
    runner.train(epochs=100)
    runner.show_reconstruction()

    pass
