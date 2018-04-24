import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

    def get_base_class(self, train_data=True):
        images = self._data_train.images if train_data else self._data_test.images
        labels = self._data_train.labels if train_data else self._data_test.labels

        # data_index = [10, 58, 76, 97, 150, 240, 121, 100, 43, 185]
        data_index = [48, 23, 13, 11, 77, 66, 260, 178, 215, 8]
        # data_index_9 = [8, 17, 24, 44, 45, 63, 93, 106, 125, 137]
        # data_index_2 = [13, 16, 65, 76, 94, 95, 103, 104, 113, 129]
        result_data = images[data_index]
        # for i in range(300):
        #     base_class_data = np.array(np.squeeze(images[i] * 255), dtype=np.uint8)
        #     Image.fromarray(base_class_data).convert("L").save(os.path.join("result",
        #                                                                     "{}_{}.bmp".format(labels[i], i)))
        #     pass

        return result_data

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
        self.logits_cnn, self.softmax_cnn, self.prediction_cnn, self.prediction_sort_cnn = None, None, None, None
        # 损失和训练
        self.loss, self.loss_1, self.loss_2, self.train_op = None, None, None, None

        with self.graph.as_default():
            # 输入，每个10通道：n * 28 * 28 * 10
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_size,
                                                             self.data_size, self.class_number], name="x")
            self.x_ae = tf.split(self.x, num_or_size_splits=self.class_number, axis=3)
            self.x_ae = tf.squeeze(tf.concat(self.x_ae, axis=0))
            # 变成1通道: 10n * 784
            self.x_ae = tf.reshape(self.x_ae, shape=[-1, self.data_size * self.data_size])
            self.hidden_ae, self.reconstruction_ae = self.net_ae(self.x_ae, self.data_size * self.data_size, 80)
            # AE结果: 10n * 28 * 28 * 1
            self.reconstruction_ae_reshape = tf.reshape(self.reconstruction_ae, [-1, self.data_size, self.data_size, 1])

            self.x_cnn_all = []
            for i in range(self.class_number):
                self.x_cnn_all.append(self.x)
            # 输入，每个10通道：10n * 28 * 28 * 10
            self.x_cnn_all = tf.concat(self.x_cnn_all, axis=0)

            # 结合.每个11通道：10n * 28 * 28 * 11
            self.x_cnn = tf.concat([self.x_cnn_all, self.reconstruction_ae_reshape], axis=3)
            self.label_cnn = tf.placeholder(dtype=tf.int32, shape=[None], name="label")

            self.logits_cnn, self.softmax_cnn, self.prediction_cnn, self.prediction_sort_cnn = self.net_cnn(self.x_cnn)

            self.loss, self.loss_1, self.loss_2 = self.loss_example()
            self.train_op = self.train_op_example(learning_rate=0.001, loss_all=self.loss)
        pass

    @staticmethod
    def net_ae(x_ae, n_input, n_hidden):
        n_hidden_2 = n_hidden * 1
        weights = dict()
        weights["w1"] = tf.get_variable("w1", shape=[n_input, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer())
        weights["b1"] = tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float32))

        # weights["w2"] = tf.get_variable("w2", shape=[n_hidden_2, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
        # weights["b2"] = tf.Variable(tf.zeros([n_hidden], dtype=tf.float32))

        # weights["w3"] = tf.Variable(tf.zeros([n_hidden, n_hidden_2], dtype=tf.float32))
        # weights["b3"] = tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float32))

        weights["w4"] = tf.Variable(tf.zeros([n_hidden_2, n_input], dtype=tf.float32))
        weights["b4"] = tf.Variable(tf.zeros([n_input], dtype=tf.float32))

        # model
        hidden1 = tf.nn.softplus(tf.add(tf.matmul(x_ae, weights["w1"]), weights["b1"]))
        # hidden2 = tf.nn.softplus(tf.add(tf.matmul(hidden1, weights["w2"]), weights["b2"]))
        hidden2 = hidden1

        # reconstruction1 = tf.nn.softplus(tf.add(tf.matmul(hidden2, weights["w3"]), weights["b3"]))
        reconstruction1 = hidden2
        reconstruction2 = tf.add(tf.matmul(reconstruction1, weights["w4"]), weights["b4"])
        return hidden2, reconstruction2

    # 网络结构
    def net_cnn(self, input_op):
        filter_number = 128
        kernel_size = 5
        weight_1 = tf.Variable(tf.truncated_normal(shape=[kernel_size, kernel_size, self.class_number + 1, filter_number], stddev=5e-2))
        kernel_1 = tf.nn.conv2d(input_op, weight_1, [1, 1, 1, 1], padding="SAME")
        bias_1 = tf.Variable(tf.constant(0.0, shape=[filter_number]))
        conv_1 = tf.nn.relu(tf.nn.bias_add(kernel_1, bias_1))
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding="SAME")
        norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        weight_2 = tf.Variable(tf.truncated_normal(shape=[kernel_size, kernel_size, filter_number, filter_number * 2], stddev=5e-2))
        kernel_2 = tf.nn.conv2d(norm_1, weight_2, [1, 1, 1, 1], padding="SAME")
        bias_2 = tf.Variable(tf.constant(0.1, shape=[filter_number * 2]))
        conv_2 = tf.nn.relu(tf.nn.bias_add(kernel_2, bias_2))
        norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        weight_23 = tf.Variable(tf.truncated_normal(shape=[kernel_size, kernel_size, filter_number * 2, filter_number * 4], stddev=5e-2))
        kernel_23 = tf.nn.conv2d(pool_2, weight_23, [1, 1, 1, 1], padding="SAME")
        bias_23 = tf.Variable(tf.constant(0.1, shape=[filter_number * 4]))
        conv_23 = tf.nn.relu(tf.nn.bias_add(kernel_23, bias_23))
        norm_23 = tf.nn.lrn(conv_23, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool_23 = tf.nn.max_pool(norm_23, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        pool_23_shape = pool_23.get_shape().as_list()
        reshape = tf.reshape(pool_23, [-1, pool_23_shape[1] * pool_23_shape[2] * filter_number * 4])
        dim = reshape.get_shape()[1].value

        weight_4 = tf.Variable(tf.truncated_normal(shape=[dim, 192], stddev=0.04))
        bias_4 = tf.Variable(tf.constant(0.1, shape=[192]))
        local_4 = tf.nn.relu(tf.matmul(reshape, weight_4) + bias_4)

        weight_5 = tf.Variable(tf.truncated_normal(shape=[192, self.class_number], stddev=1 / 192.0))
        bias_5 = tf.Variable(tf.constant(0.0, shape=[self.class_number]))
        logits = tf.add(tf.matmul(local_4, weight_5), bias_5)

        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax, 1)
        prediction_sort = tf.nn.top_k(softmax, k=self.class_number)

        return logits, softmax, prediction, prediction_sort

    # 损失
    def loss_example(self):
        # cost
        loss_1 = 5 * 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction_ae, self.x_ae), 2.0))
        loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_cnn,
                                                                               logits=self.logits_cnn))
        return tf.add(loss_1, loss_2), loss_1, loss_2
        # return loss_1, loss_1, loss_2

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
    def train(self, epochs=20, test_freq=1, save_freq=2):
        with self.supervisor.managed_session(config=self.config) as sess:
            for epoch in range(epochs):
                # stop
                if self.supervisor.should_stop():
                    break
                # train
                loss_all, loss_1_all, loss_2_all = 0, 0, 0
                for step in range(self.data.number_train):
                    x = self.data.next_train_batch()
                    label = np.reshape([[i] * len(x) for i in range(self.class_number)], newshape=[-1])

                    _, loss, loss_1, loss_2 = sess.run([self.net.train_op, self.net.loss,
                                                        self.net.loss_1, self.net.loss_2],
                                                       feed_dict={self.net.x: x, self.net.label_cnn: label})
                    if step % 100 == 0:
                        Tools.print_info("epoch={} step={} loss={} loss1={} loss2={}".format(epoch, step,
                                                                                             loss, loss_1, loss_2))
                    loss_all += np.mean(loss)
                    loss_1_all += np.mean(loss_1)
                    loss_2_all += np.mean(loss_2)
                    pass
                loss_all = loss_all / self.data.number_train
                loss_1_all = loss_1_all / self.data.number_train
                loss_2_all = loss_2_all / self.data.number_train
                Tools.print_info("epoch={} loss={} loss1={} loss2={}".format(epoch, loss_all, loss_1_all, loss_2_all))
                # test
                if epoch % test_freq == 0:
                    self._test(sess, epoch)
                # save
                if epoch % save_freq == 0:
                    self.supervisor.saver.save(sess, os.path.join(self.model_path, "model_epoch_{}".format(epoch)))
            pass
            self.show_cnn_input(sess)
            self.show_reconstruction(sess)
        pass

    # 测试
    def test(self, info="test"):
        with self.supervisor.managed_session(config=self.config) as sess:
            self._test(sess, info)
        pass

    def _test(self, sess, info):
        test_acc = 0
        for i in range(self.data.number_test):
            x = self.data.next_test_batch(i)
            label = np.reshape([[i] * len(x) for i in range(10)], newshape=[-1])
            prediction = sess.run(self.net.prediction_cnn, {self.net.x: x})
            test_acc += np.sum(np.equal(label, prediction))
        test_acc = test_acc / (self.batch_size * self.data.number_test * 10)
        Tools.print_info("{} acc={}".format(info, test_acc))
        return test_acc

    # 推理：准备几个基线
    def inference(self, result_path="result", inference_len=5000, is_save_image=False):
        Tools.new_dir(result_path)
        # 获取基准图片
        base_data = self.data.get_base_class(train_data=True)
        # 保存基准图片
        save_path = Tools.new_dir(os.path.join(result_path, "base"))
        for index, base_class_data in enumerate(base_data):
            base_class_data = np.array(np.squeeze(base_class_data * 255), dtype=np.uint8)
            Image.fromarray(base_class_data).convert("L").save(os.path.join(save_path, "{}.bmp".format(index)))
            pass

        # inference
        with self.supervisor.managed_session(config=self.config) as sess:
            base_data = np.concatenate(base_data, axis=2)
            # 获取测试图片
            save_path = Tools.new_dir(os.path.join(result_path, "judge"))
            judge_data, judge_label = self.data.get_batch_data(train_data=False, image_number=inference_len)

            prediction_result = []
            is_ok = [False] * len(judge_label)

            for index, now_data in enumerate(judge_data):
                if index % 100 == 0:
                    Tools.print_info("{}/{}".format(index, len(judge_label)))
                    pass

                # 拼接数据
                full_data = np.concatenate([base_data, now_data], axis=2)

                # 预测
                prediction, softmax, prediction_sort = sess.run(
                    [self.net.prediction_cnn, self.net.softmax_cnn, self.net.prediction_sort_cnn],
                    {self.net.x_cnn: [full_data]})
                prediction_result.append(prediction[0])

                # 保存结果
                now_data = np.array(np.squeeze(now_data) * 255, dtype=np.uint8)
                if prediction_sort[0][0][0] > 0.:
                    is_ok[index] = True
                    if is_save_image:
                        Image.fromarray(now_data).convert("L").save(
                            os.path.join(save_path, "{}_{}_{}.bmp".format(judge_label[index], prediction[0], index)))
                        Tools.print_info("{} label={} prediction={} sort={} p={}".format(
                            index, judge_label[index], prediction[0], list(prediction_sort[1][0]), list(prediction_sort[0][0])))
                        pass
                    pass

                pass

            # 打印结果
            self._print_result(judge_label, prediction_result, self.data.class_number, is_ok)

            pass

        pass

    @staticmethod
    def _print_result(judge_label, prediction_result, class_number, is_ok=None):
        judge_label = list(judge_label)
        Tools.print_info(judge_label)
        Tools.print_info(prediction_result)
        acc = np.zeros(shape=[class_number, 3], dtype=np.int)
        for index in range(len(judge_label)):
            if is_ok is not None and not is_ok[index]:
                continue
            acc[judge_label[index]][1] += 1
            acc[prediction_result[index]][2] += 1
            if judge_label[index] == prediction_result[index]:
                acc[judge_label[index]][0] += 1
            pass
        acc_all = np.sum(np.squeeze(np.split(acc, 3, 1))[0])
        count = np.sum(is_ok) if is_ok is not None else len(judge_label)
        Tools.print_info("all acc={}({}/{})".format(acc_all * 1.0 / count, acc_all, count))
        for index in range(len(acc)):
            Tools.print_info("{} acc={}({}/{}/{})".format(index, acc[index][0] * 1.0 / acc[index][1],
                                                          acc[index][0], acc[index][1], acc[index][2]))
            pass
        pass

    def show_reconstruction(self, sess):
        image_data = self.data.get_image_for_reconstruct()
        reconstruction_ae = sess.run(self.net.reconstruction_ae, {self.net.x_ae: image_data})

        self.show_image(len(image_data), image_data, reconstruction_ae)
        pass

    def show_cnn_input(self, sess):
        image_data = self.data.next_train_batch()
        x_cnn = sess.run(self.net.x_cnn, {self.net.x: image_data})

        len_data = image_data[3].shape[-1] + 1
        x_cnn = np.split(x_cnn[13], indices_or_sections=len_data, axis=2)
        self.show_image(len_data, x_cnn, x_cnn)
        pass

    def show_image(self, image_length, image_data, image_data2):
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
    parser.add_argument("-batch_size", type=int, default=5, help="batch size")
    parser.add_argument("-class_number", type=int, default=10, help="type number")
    parser.add_argument("-data_path", type=str, default="./data/mnist", help="image data")
    args = parser.parse_args()

    output_param = "name={}batch_size={},class_number={},data_path={}"
    Tools.print_info(output_param.format(args.name, args.batch_size, args.class_number, args.data_path))

    runner = Runner(Data(batch_size=args.batch_size, class_number=args.class_number, data_path=args.data_path),
                    model_path=os.path.join("model", args.name))
    runner.train(epochs=50)
    runner.test()
    runner.inference(result_path=os.path.join("result", args.name))

    pass
