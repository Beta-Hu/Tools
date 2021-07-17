# -*- ecoding: utf-8 -*-
# @ModuleName: visualize
# @Author: BetaHu
# @Time: 2021/7/5 16:31
import time
import matplotlib.pyplot as plt


class Visualizer:
    """
    A tool to visualize training progress
    Usage:
        logger = visualizer.Visualizer(['train loss', 'ssim', 'Qabf'], ['r-o', 'b-o', 'y-o'], './log.txt')
        ...
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
        _, self.ax_1 = plt.subplots()
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
        for idx, curve_name in enumerate(self.pattern):
            loss = [item[idx] for item in self.loss_curve]
            if max(loss) > 1:
                self.ax_1.plot(list(range(1, len(loss) + 1)), loss, self.line_type[idx])
            else:
                self.ax_2.plot(list(range(1, len(loss) + 1)), loss, self.line_type[idx])
        plt.legend(self.pattern)
        plt.xlabel('Epoch')
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    import random
    # generate a visualizer
    logger = Visualizer(['train loss', 'ssim', 'Qabf'], ['r-o', 'b-o', 'y-o'], './log.txt')
    # during training
    for i in range(0, 10):
        loss = 10 - i + random.random()
        ssim = (i + random.random()) / 10
        qabf = (i + random.random()) / 10
        # plot and write log
        logger.write_log([loss, ssim, qabf])
        logger.plot_loss()
        time.sleep(1)
