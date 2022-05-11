import os
import json

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# 验证集共1476张，其中恶性1有251张、恶性2有307张、恶性3有344张、恶性4有245、恶性5有329.
class ConfusionMatrix(object):


    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        # self.matrix = np.array([[199,49,0,0,0],
        #                         [39,242,55,2,3],
        #                         [11,21,241,20,12],
        #                         [2,1,45,205,10],
        #                         [0,0,3,18,304]])
        self.matrix = np.array([[225,22,2,3,0],
                                [20,274,25,1,2],
                                [4,10,293,12,8],
                                [2,1,23,219,7],
                                [0,0,1,10,312]])

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        # table.field_names = ["", "Precision", "Recall", "Specificity"]
        table.field_names = ["", "PPV", "TPR", "TNR"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)#从白色到蓝色进行变化

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')
        # plt.xlabel('真实标签')
        # plt.ylabel('预测标签')
        # plt.title('混淆矩阵')
        # 在图中标注数量/概率信息
        thresh = matrix.max()/ 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()#图像显示更加紧凑
        plt.show()


if __name__ == '__main__':

    # read class_indict
    json_label_path = './class_indices_9.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=5, labels=labels)


    confusion.plot()
    confusion.summary()

