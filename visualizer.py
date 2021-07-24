import time
import matplotlib.pyplot as plt
import numpy as np
import math


class Visualizer:
    """
    A tool to visualize training progress
    Usage:
        logger = visualizer.Visualizer(['train loss', 'ssim', 'Qabf'], ['r-o', 'b-o', 'y-o'], './log.txt')
        ...training code...
        logger.write_log([loss, ssim, qabf])
        logger.plot_loss()
    """
    def __init__(self, pattern: list, line_type: list, log_path: str):
        """
        :param pattern: name of loss
        :param log_path: log file name and path
        :param line_type: param to set line type when drawing, len(line_type) should be larger than len(pattern)
        """
        assert len(pattern) <= len(line_type), 'line_type should contains more elements'
        self.pattern = pattern
        self.log_path = log_path
        self.line_type = line_type
        self.loss_curve = []
        self.__log_init()

        plt.ion()
        self.fig, self.ax_1 = plt.subplots()
        self.ax_2 = self.ax_1.twinx()

    def __log_init(self):
        f = open(self.log_path, 'a+')
        f.write('---LOG AT %s---\n' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.close()

    def write_log(self, loss: list):
        self.loss_curve.append(loss)
        log_item = [self.pattern[i] + ': %.4f, ' % loss[i] for i in range(0, len(loss))]
        f = open(self.log_path, 'a+')
        f.write('epoch: %03d, ' % len(self.loss_curve) + ''.join(log_item) + '\n')
        f.close()
        print('epoch: %03d, ' % len(self.loss_curve) + ''.join(log_item))

    def plot_loss(self):
        plt.cla()
        lines = []
        label = ['', '']
        for idx, curve_name in enumerate(self.pattern):
            loss = [item[idx] for item in self.loss_curve]
            if max(loss) > 2:
                line, = self.ax_1.plot(list(range(1, len(loss) + 1)), loss, self.line_type[idx])
                label[0] = label[0] + curve_name + ' / '
            else:
                line, = self.ax_2.plot(list(range(1, len(loss) + 1)), loss, self.line_type[idx])
                label[1] = label[1] + curve_name + ' / '
            lines.append(line)
        plt.legend(lines, self.pattern)
        plt.xlabel('Epoch')
        self.ax_1.set_ylabel(label[0][0: -2])
        self.ax_2.set_ylabel(label[1][0: -2])
        plt.draw()
        plt.savefig(self.log_path[0: -4] + '.png')
        plt.pause(0.001)


class Montage:
    """
    A tool to visualize feature maps in neural networks
    """
    def __init__(self, transform=None):
        """
        transform: None or 'histeq'
        """
        self.type_tf = transform

    def __layout(self, channel_map):
        """
        generate layout of image grid
        num_map: channel of feature map
        pad: number of blocks to pad image grid as a square
        """
        col = math.ceil(math.sqrt(channel_map))
        row = col
        pad = col * row - channel_map
        return row, col, pad

    def __histeq(self, feature_map):
        feature_map = 255 * (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1)
        feature_map = feature_map.astype(np.int8)
        hist, bins = np.histogram(feature_map, 255)
        cdf = np.cumsum(hist)
        cdf = 255 * (cdf / cdf[-1])
        res = np.interp(feature_map.flatten(), bins[:-1], cdf)
        res = res.reshape(feature_map.shape)
        return res

    def __transform(self, feature_map, type_trans):
        if type_trans is None:
            trans_map = feature_map
        else:
            trans_map = self.__histeq(feature_map)
        return trans_map

    def montage(self, feature_map, batch_index=0):
        """
        turn feature_map into 2D image grid and visualize
        feature_map: feature map from neural networks, must be [B, C, H, W] and type of torch.Tensor
        batch_index: select which image in a batch to montage, must be in range of [0, B)
        """
        [num_channel, h, w] = feature_map[batch_index].shape
        row, col, pad = self.__layout(num_channel)
        feature_map = feature_map[batch_index].numpy()
        feature_map = np.concatenate((feature_map, np.zeros([pad, h, w])), axis=0)
        merge_map = np.zeros([row * h, col * w])
        channel = 0
        for r in range(0, row):
            for c in range(0, col):
                merge_map[r * h: (r + 1) * h, c * w: (c + 1) * w] = self.__transform(feature_map[channel], self.type_tf)
                channel += 1

        plt.figure()
        plt.imshow(merge_map, cmap='plasma')
        plt.axis('off')
        plt.show()
