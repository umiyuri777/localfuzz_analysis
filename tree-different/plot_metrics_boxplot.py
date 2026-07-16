"""
task0 / task1 / task2 の適合率・再現率・F値、および特徴量重要度を箱ひげ図で描画する。

各 task のサブプロット内に、全アルゴリズムを横並びで表示する。
評価指標図では各アルゴリズムのグループ内に適合率・再現率・F値の 3 箱を並べる。
特徴量重要度図では DT / RF / GB ごとに各特徴量の箱を並べる。

使い方:
1. export_latex_tables.py を実行し、出力された CV_FOLD_SCORES / CV_FOLD_IMPORTANCES を貼り付ける
2. ラベルなどの設定を必要に応じて変更する
3. python plot_metrics_boxplot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# ========================================================================
# 貼り付け用: plot_metrics_boxplot.py の CV_FOLD_SCORES にコピー
# ========================================================================
CV_FOLD_SCORES: dict[str, dict[str, dict[str, list[float]]]] = {
    "task0": {
        "BL": {
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "precision": [0.5769, 0.5769, 0.5769, 0.5759, 0.5759, 0.5759, 0.5759, 0.5759, 0.5759, 0.5759],
            "auc": [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            "f1": [0.7317, 0.7317, 0.7317, 0.7309, 0.7309, 0.7309, 0.7309, 0.7309, 0.7309, 0.7309]
        },
        "LR": {
            "recall": [0.8700, 0.8989, 0.8860, 0.9212, 0.9084, 0.8971, 0.8826, 0.8762, 0.8891, 0.8617],
            "precision": [0.7936, 0.7955, 0.7852, 0.8328, 0.7947, 0.7937, 0.7614, 0.7933, 0.7811, 0.8048],
            "auc": [0.8136, 0.8401, 0.8232, 0.8713, 0.8317, 0.8328, 0.7982, 0.8161, 0.8322, 0.8195],
            "f1": [0.8300, 0.8440, 0.8326, 0.8748, 0.8477, 0.8423, 0.8176, 0.8327, 0.8316, 0.8323]
        },
        "DT": {
            "recall": [0.8266, 0.8443, 0.8299, 0.8778, 0.8650, 0.8601, 0.8376, 0.8537, 0.8505, 0.8312],
            "precision": [0.8744, 0.8767, 0.8778, 0.9085, 0.8820, 0.8932, 0.8430, 0.8806, 0.8715, 0.8853],
            "auc": [0.8457, 0.8719, 0.8635, 0.9034, 0.8665, 0.8800, 0.8303, 0.8620, 0.8635, 0.8574],
            "f1": [0.8498, 0.8602, 0.8531, 0.8929, 0.8734, 0.8763, 0.8403, 0.8669, 0.8609, 0.8574]
        },
        "RF": {
            "recall": [0.8266, 0.8443, 0.8299, 0.8778, 0.8650, 0.8601, 0.8376, 0.8537, 0.8505, 0.8312],
            "precision": [0.8744, 0.8767, 0.8778, 0.9085, 0.8820, 0.8932, 0.8430, 0.8806, 0.8715, 0.8853],
            "auc": [0.8490, 0.8706, 0.8599, 0.9014, 0.8709, 0.8812, 0.8299, 0.8615, 0.8620, 0.8588],
            "f1": [0.8498, 0.8602, 0.8531, 0.8929, 0.8734, 0.8763, 0.8403, 0.8669, 0.8609, 0.8574]
        },
        "GB": {
            "recall": [0.8266, 0.8443, 0.8299, 0.8778, 0.8650, 0.8601, 0.8376, 0.8537, 0.8505, 0.8312],
            "precision": [0.8744, 0.8767, 0.8778, 0.9085, 0.8820, 0.8932, 0.8430, 0.8806, 0.8715, 0.8853],
            "auc": [0.8536, 0.8711, 0.8605, 0.9045, 0.8701, 0.8818, 0.8325, 0.8602, 0.8643, 0.8603],
            "f1": [0.8498, 0.8602, 0.8531, 0.8929, 0.8734, 0.8763, 0.8403, 0.8669, 0.8609, 0.8574]
        }
    },
    "task1": {
        "BL": {
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "precision": [0.8472, 0.8472, 0.8472, 0.8472, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426],
            "auc": [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            "f1": [0.9173, 0.9173, 0.9173, 0.9173, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146]
        },
        "LR": {
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "precision": [0.8472, 0.8472, 0.8472, 0.8472, 0.8426, 0.8426, 0.8426, 0.8465, 0.8426, 0.8426],
            "auc": [0.8243, 0.8685, 0.8208, 0.8121, 0.8422, 0.8075, 0.8169, 0.7747, 0.8336, 0.7665],
            "f1": [0.9173, 0.9173, 0.9173, 0.9173, 0.9146, 0.9146, 0.9146, 0.9169, 0.9146, 0.9146]
        },
        "DT": {
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "precision": [0.8472, 0.8472, 0.8472, 0.8472, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426],
            "auc": [0.8452, 0.8615, 0.8142, 0.8306, 0.8314, 0.7930, 0.8008, 0.8480, 0.8268, 0.8376],
            "f1": [0.9173, 0.9173, 0.9173, 0.9173, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146]
        },
        "RF": {
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "precision": [0.8472, 0.8472, 0.8472, 0.8472, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426],
            "auc": [0.8443, 0.8801, 0.8238, 0.8558, 0.8120, 0.7872, 0.8038, 0.8463, 0.8299, 0.8284],
            "f1": [0.9173, 0.9173, 0.9173, 0.9173, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146]
        },
        "GB": {
            "recall": [0.9672, 0.9781, 0.9781, 0.9672, 0.9725, 0.9670, 0.9835, 0.9835, 0.9725, 0.9725],
            "precision": [0.8429, 0.8565, 0.8443, 0.8551, 0.8469, 0.8502, 0.8443, 0.8483, 0.8469, 0.8510],
            "auc": [0.8280, 0.8766, 0.8309, 0.8796, 0.8053, 0.8106, 0.7828, 0.8405, 0.8340, 0.8297],
            "f1": [0.9008, 0.9133, 0.9063, 0.9077, 0.9054, 0.9049, 0.9086, 0.9109, 0.9054, 0.9077]
        }
    },
    "task2": {
        "BL": {
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "precision": [0.3565, 0.3565, 0.3565, 0.3519, 0.3519, 0.3519, 0.3519, 0.3519, 0.3519, 0.3519],
            "auc": [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            "f1": [0.5256, 0.5256, 0.5256, 0.5205, 0.5205, 0.5205, 0.5205, 0.5205, 0.5205, 0.5205]
        },
        "LR": {
            "recall": [0.6753, 0.6494, 0.6753, 0.5789, 0.6053, 0.6053, 0.5921, 0.5658, 0.5526, 0.7237],
            "precision": [0.6420, 0.5556, 0.6265, 0.6567, 0.5542, 0.6301, 0.6338, 0.5059, 0.5753, 0.5978],
            "auc": [0.8257, 0.8058, 0.8291, 0.8595, 0.8281, 0.8417, 0.8595, 0.7633, 0.8045, 0.8293],
            "f1": [0.6582, 0.5988, 0.6500, 0.6154, 0.5786, 0.6174, 0.6122, 0.5342, 0.5638, 0.6548]
        },
        "DT": {
            "recall": [0.8831, 1.0000, 1.0000, 1.0000, 0.8816, 1.0000, 0.8947, 0.9474, 0.8158, 0.9342],
            "precision": [0.6239, 0.6016, 0.6063, 0.6552, 0.7053, 0.5846, 0.7234, 0.6606, 0.6327, 0.6283],
            "auc": [0.8559, 0.8659, 0.8715, 0.8664, 0.8907, 0.8539, 0.9249, 0.8636, 0.8588, 0.8715],
            "f1": [0.7312, 0.7512, 0.7549, 0.7917, 0.7836, 0.7379, 0.8000, 0.7784, 0.7126, 0.7513]
        },
        "RF": {
            "recall": [0.7922, 0.8961, 0.9870, 0.7895, 0.7632, 0.9474, 0.8421, 0.8816, 0.8026, 0.9605],
            "precision": [0.6040, 0.6161, 0.6179, 0.6383, 0.6824, 0.6050, 0.7111, 0.6700, 0.6289, 0.6460],
            "auc": [0.8604, 0.8599, 0.8682, 0.8641, 0.8875, 0.8614, 0.9107, 0.8831, 0.8598, 0.8897],
            "f1": [0.6854, 0.7302, 0.7600, 0.7059, 0.7205, 0.7385, 0.7711, 0.7614, 0.7052, 0.7725]
        },
        "GB": {
            "recall": [0.8442, 0.8961, 0.9870, 0.8289, 0.7895, 0.9474, 0.8289, 0.8026, 0.8553, 0.9737],
            "precision": [0.6373, 0.6053, 0.6179, 0.6300, 0.6818, 0.6000, 0.6848, 0.6559, 0.6311, 0.6435],
            "auc": [0.8623, 0.8561, 0.8708, 0.8619, 0.8922, 0.8603, 0.9035, 0.8758, 0.8524, 0.8781],
            "f1": [0.7263, 0.7225, 0.7600, 0.7159, 0.7317, 0.7347, 0.7500, 0.7219, 0.7263, 0.7749]
        }
    }
}

# ========================================================================
# 貼り付け用: plot_metrics_boxplot.py の CV_FOLD_IMPORTANCES にコピー
# ========================================================================
CV_FOLD_IMPORTANCES: dict[str, dict[str, dict[str, list[float]]]] = {
    "task0": {
        "DT": {
            "tree": [0.0032, 0.0019, 0.0024, 0.0032, 0.0029, 0.0024, 0.0034, 0.0028, 0.0025, 0.0023],
            "cpNum": [0.3067, 0.3122, 0.3072, 0.3166, 0.3090, 0.3054, 0.3042, 0.3069, 0.3057, 0.3086],
            "cpNum_range": [0.6900, 0.6858, 0.6904, 0.6802, 0.6881, 0.6922, 0.6925, 0.6903, 0.6915, 0.6891],
            "cpNum_dir_2": [0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            "cpNum_dir_3": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0003, 0.0000],
            "cpNum_dir_4": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        },
        "RF": {
            "tree": [0.0050, 0.0040, 0.0041, 0.0044, 0.0044, 0.0044, 0.0050, 0.0054, 0.0049, 0.0049],
            "cpNum": [0.2481, 0.2446, 0.2453, 0.2498, 0.2396, 0.2570, 0.2488, 0.2495, 0.2444, 0.2576],
            "cpNum_range": [0.7436, 0.7481, 0.7472, 0.7425, 0.7526, 0.7352, 0.7431, 0.7419, 0.7476, 0.7341],
            "cpNum_dir_2": [0.0012, 0.0010, 0.0011, 0.0009, 0.0010, 0.0011, 0.0011, 0.0008, 0.0011, 0.0011],
            "cpNum_dir_3": [0.0008, 0.0010, 0.0011, 0.0011, 0.0012, 0.0010, 0.0009, 0.0011, 0.0010, 0.0010],
            "cpNum_dir_4": [0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0011, 0.0014, 0.0010, 0.0013]
        },
        "GB": {
            "tree": [0.0066, 0.0061, 0.0052, 0.0055, 0.0060, 0.0060, 0.0071, 0.0075, 0.0062, 0.0061],
            "cpNum": [0.3055, 0.3108, 0.3062, 0.3156, 0.3087, 0.3045, 0.3030, 0.3055, 0.3049, 0.3071],
            "cpNum_range": [0.6856, 0.6802, 0.6854, 0.6763, 0.6830, 0.6873, 0.6871, 0.6845, 0.6858, 0.6843],
            "cpNum_dir_2": [0.0008, 0.0004, 0.0007, 0.0008, 0.0006, 0.0006, 0.0004, 0.0005, 0.0006, 0.0009],
            "cpNum_dir_3": [0.0003, 0.0008, 0.0005, 0.0006, 0.0006, 0.0004, 0.0012, 0.0008, 0.0014, 0.0005],
            "cpNum_dir_4": [0.0011, 0.0015, 0.0020, 0.0012, 0.0011, 0.0012, 0.0011, 0.0012, 0.0011, 0.0010]
        }
    },
    "task1": {
        "DT": {
            "tree": [0.0000, 0.0045, 0.0000, 0.0045, 0.0031, 0.0000, 0.0001, 0.0034, 0.0271, 0.0045],
            "cpNum": [0.2248, 0.2487, 0.2301, 0.2313, 0.2198, 0.2100, 0.2272, 0.2268, 0.1911, 0.2068],
            "cpNum_range": [0.7572, 0.7466, 0.7649, 0.7640, 0.7693, 0.7721, 0.7570, 0.7637, 0.7816, 0.7885],
            "cpNum_dir_2": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0000, 0.0000, 0.0000],
            "cpNum_dir_3": [0.0028, 0.0000, 0.0000, 0.0000, 0.0000, 0.0035, 0.0054, 0.0000, 0.0000, 0.0000],
            "cpNum_dir_4": [0.0152, 0.0002, 0.0049, 0.0002, 0.0078, 0.0144, 0.0102, 0.0060, 0.0002, 0.0002]
        },
        "RF": {
            "tree": [0.0367, 0.0301, 0.0275, 0.0289, 0.0345, 0.0325, 0.0332, 0.0292, 0.0363, 0.0314],
            "cpNum": [0.2257, 0.2503, 0.2385, 0.2339, 0.2317, 0.2109, 0.2371, 0.2497, 0.1980, 0.2283],
            "cpNum_range": [0.7009, 0.6851, 0.6963, 0.7043, 0.6989, 0.7159, 0.6875, 0.6796, 0.7296, 0.7066],
            "cpNum_dir_2": [0.0102, 0.0130, 0.0116, 0.0124, 0.0128, 0.0140, 0.0129, 0.0129, 0.0107, 0.0093],
            "cpNum_dir_3": [0.0123, 0.0096, 0.0128, 0.0076, 0.0112, 0.0132, 0.0127, 0.0128, 0.0114, 0.0122],
            "cpNum_dir_4": [0.0142, 0.0118, 0.0132, 0.0129, 0.0109, 0.0134, 0.0165, 0.0159, 0.0140, 0.0122]
        },
        "GB": {
            "tree": [0.0507, 0.0410, 0.0491, 0.0417, 0.0588, 0.0534, 0.0545, 0.0512, 0.0569, 0.0508],
            "cpNum": [0.2250, 0.2458, 0.2255, 0.2299, 0.2200, 0.2085, 0.2410, 0.2213, 0.1942, 0.2107],
            "cpNum_range": [0.6801, 0.6869, 0.6927, 0.7027, 0.6855, 0.7045, 0.6712, 0.6867, 0.7199, 0.7089],
            "cpNum_dir_2": [0.0065, 0.0067, 0.0123, 0.0137, 0.0126, 0.0139, 0.0064, 0.0091, 0.0082, 0.0096],
            "cpNum_dir_3": [0.0181, 0.0081, 0.0123, 0.0028, 0.0099, 0.0059, 0.0141, 0.0162, 0.0062, 0.0123],
            "cpNum_dir_4": [0.0195, 0.0115, 0.0082, 0.0092, 0.0132, 0.0138, 0.0128, 0.0156, 0.0146, 0.0077]
        }
    },
    "task2": {
        "DT": {
            "tree": [0.0213, 0.0184, 0.0203, 0.0302, 0.0209, 0.0232, 0.0168, 0.0195, 0.0207, 0.0212],
            "cpNum": [0.3653, 0.3497, 0.3591, 0.3642, 0.3572, 0.3686, 0.3546, 0.3495, 0.3526, 0.3603],
            "cpNum_range": [0.6135, 0.6319, 0.6206, 0.6056, 0.6219, 0.6067, 0.6286, 0.6310, 0.6267, 0.6184],
            "cpNum_dir_2": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0014, 0.0000, 0.0000, 0.0000, 0.0000],
            "cpNum_dir_3": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            "cpNum_dir_4": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        },
        "RF": {
            "tree": [0.0257, 0.0220, 0.0242, 0.0295, 0.0247, 0.0238, 0.0240, 0.0255, 0.0256, 0.0230],
            "cpNum": [0.2950, 0.2873, 0.2878, 0.3086, 0.2910, 0.2972, 0.2912, 0.2896, 0.2864, 0.2950],
            "cpNum_range": [0.6616, 0.6749, 0.6709, 0.6443, 0.6627, 0.6609, 0.6638, 0.6666, 0.6702, 0.6655],
            "cpNum_dir_2": [0.0045, 0.0050, 0.0044, 0.0045, 0.0063, 0.0050, 0.0055, 0.0058, 0.0042, 0.0047],
            "cpNum_dir_3": [0.0047, 0.0044, 0.0048, 0.0056, 0.0052, 0.0044, 0.0062, 0.0044, 0.0050, 0.0043],
            "cpNum_dir_4": [0.0085, 0.0064, 0.0078, 0.0074, 0.0101, 0.0087, 0.0093, 0.0082, 0.0087, 0.0075]
        },
        "GB": {
            "tree": [0.0275, 0.0209, 0.0237, 0.0317, 0.0235, 0.0254, 0.0211, 0.0213, 0.0240, 0.0248],
            "cpNum": [0.3583, 0.3481, 0.3541, 0.3608, 0.3504, 0.3639, 0.3470, 0.3468, 0.3454, 0.3547],
            "cpNum_range": [0.6030, 0.6223, 0.6126, 0.5934, 0.6132, 0.6004, 0.6146, 0.6213, 0.6181, 0.6102],
            "cpNum_dir_2": [0.0010, 0.0018, 0.0014, 0.0024, 0.0021, 0.0014, 0.0016, 0.0026, 0.0018, 0.0011],
            "cpNum_dir_3": [0.0025, 0.0017, 0.0007, 0.0016, 0.0005, 0.0005, 0.0019, 0.0006, 0.0009, 0.0008],
            "cpNum_dir_4": [0.0079, 0.0052, 0.0075, 0.0103, 0.0104, 0.0084, 0.0139, 0.0074, 0.0098, 0.0084]
        }
    }
}

# =============================================================================
# 描画設定（文言はここで変更）
# =============================================================================
MODEL_ORDER = ["BL", "LR", "DT", "RF", "GB"]
MODEL_LABELS = {
    "BL": "BL",
    "LR": "LR",
    "DT": "DT",
    "RF": "RF",
    "GB": "GB",
}

TASK_ORDER = ["task0", "task1", "task2"]
TASK_LABELS = {
    "task0": "Single",
    "task1": "Partial",
    "task2": "All",
}

METRIC_ORDER = ["precision", "recall", "f1"]
METRIC_LABELS = {
    "precision": "適合率",
    "recall": "再現率",
    "f1": "F値",
}

FIGURE_TITLE = "各モデルの評価結果(モデル構築プロセス)"
YLABEL = "スコア"
YLIM = (0.3, 1.02)  # 1.0 ちょうどの箱ひげが上端で見えなくなるのを防ぐ

FIGURE_SIZE = (15, 5)
# 白黒印刷向けグレースケール（明→暗）
METRIC_COLORS = ["#C8C8C8", "#909090", "#585858"]
BOX_ALPHA = 0.85
BOX_EDGE_COLOR = "black"
GROUP_GAP = 0.4
METRIC_SPACING = 0.3
BOX_WIDTH = 0.2
BOX_LABEL_FONTSIZE = 8
BOX_LABEL_ROTATION = 90
BOX_LABEL_Y = -0.02
MODEL_LABEL_FONTSIZE = 15
MODEL_LABEL_Y = -0.15

GUIDE_LINE_STYLE = "--"
GUIDE_LINE_COLOR = "#A0A0A0"
GUIDE_LINE_ALPHA = 0.55
GUIDE_LINE_WIDTH = 0.7

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_BASENAME = "metrics_model"
IMPORTANCE_OUTPUT_BASENAME = "importance_model"
OUTPUT_FORMATS = ("png",)  # 日本語ラベル利用時は pdf はフォント設定が必要

IMPORTANCE_MODEL_ORDER = ["DT", "RF", "GB"]
IMPORTANCE_MODEL_LABELS = {
    "DT": "DT",
    "RF": "RF",
    "GB": "GB",
}

FEATURE_ORDER = [
    "tree",
    "cpNum",
    "cpNum_range",
    "cpNum_dir_2",
    "cpNum_dir_3",
    "cpNum_dir_4",
]
FEATURE_LABELS = {
    "tree": "tree",
    "cpNum": "C",
    "cpNum_range": "D",
    "cpNum_dir_2": "E=2",
    "cpNum_dir_3": "E=3",
    "cpNum_dir_4": "E=4",
}

IMPORTANCE_FIGURE_TITLE = "各モデルの特徴量重要度(モデル構築プロセス)"
IMPORTANCE_YLABEL = "重要度"
IMPORTANCE_YLIM = (0.0, 1.02)
IMPORTANCE_FIGURE_SIZE = (12, 5)
FEATURE_COLORS = [
    "#D8D8D8",
    "#B8B8B8",
    "#989898",
    "#787878",
    "#585858",
    "#383838",
]
IMPORTANCE_GROUP_GAP = 0.5
IMPORTANCE_FEATURE_SPACING = 0.22
IMPORTANCE_BOX_WIDTH = 0.16
IMPORTANCE_BOX_LABEL_FONTSIZE = 7
IMPORTANCE_BOX_LABEL_ROTATION = 90
IMPORTANCE_BOX_LABEL_Y = -0.02
IMPORTANCE_MODEL_LABEL_FONTSIZE = 15
IMPORTANCE_MODEL_LABEL_Y = -0.18

JAPANESE_FONTS = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic Pro",
    "Yu Gothic",
    "Meiryo",
    "Noto Sans CJK JP",
]


def configure_matplotlib() -> None:
    """日本語表示とマイナス記号の設定。"""
    plt.rcParams["axes.unicode_minus"] = False
    for font in JAPANESE_FONTS:
        try:
            plt.rcParams["font.family"] = font
            break
        except OSError:
            continue


def _validate_scores(
    cv_fold_scores: dict[str, dict[str, dict[str, list[float]]]],
) -> None:
    missing_tasks = [task_id for task_id in TASK_ORDER if task_id not in cv_fold_scores]
    if missing_tasks:
        raise ValueError(f"CV_FOLD_SCORES に task が不足しています: {missing_tasks}")

    for task_id in TASK_ORDER:
        task_scores = cv_fold_scores[task_id]
        for model_name in MODEL_ORDER:
            if model_name not in task_scores:
                available = ", ".join(sorted(task_scores))
                raise ValueError(
                    f"{task_id} にモデル '{model_name}' がありません。"
                    f" 利用可能: {available}"
                )
            for metric_key in METRIC_ORDER:
                if metric_key not in task_scores[model_name]:
                    raise ValueError(
                        f"{task_id} / {model_name} に指標 '{metric_key}' がありません"
                    )


def _validate_importances(
    cv_fold_importances: dict[str, dict[str, dict[str, list[float]]]],
) -> None:
    missing_tasks = [
        task_id for task_id in TASK_ORDER if task_id not in cv_fold_importances
    ]
    if missing_tasks:
        raise ValueError(f"CV_FOLD_IMPORTANCES に task が不足しています: {missing_tasks}")

    for task_id in TASK_ORDER:
        task_importances = cv_fold_importances[task_id]
        for model_name in IMPORTANCE_MODEL_ORDER:
            if model_name not in task_importances:
                available = ", ".join(sorted(task_importances))
                raise ValueError(
                    f"{task_id} にモデル '{model_name}' がありません。"
                    f" 利用可能: {available}"
                )
            for feature_name in FEATURE_ORDER:
                if feature_name not in task_importances[model_name]:
                    raise ValueError(
                        f"{task_id} / {model_name} に特徴量 '{feature_name}' がありません"
                    )


def _build_grouped_boxplot_data(
    task_scores: dict[str, dict[str, list[float]]],
    *,
    model_order: list[str],
    model_labels: dict[str, str],
    item_order: list[str],
    item_colors: list[str],
    group_gap: float,
    item_spacing: float,
) -> tuple[
    list[list[float]],
    list[float],
    list[str],
    list[float],
    list[str],
    list[str],
]:
    """1 task 分のグループ化箱ひげ図用データを生成する。"""
    n_items = len(item_order)
    group_stride = (n_items - 1) * item_spacing + group_gap + item_spacing

    box_data: list[list[float]] = []
    positions: list[float] = []
    box_colors: list[str] = []
    box_item_keys: list[str] = []
    group_centers: list[float] = []

    for group_idx, model_name in enumerate(model_order):
        group_start = group_idx * group_stride
        group_centers.append(group_start + (n_items - 1) * item_spacing / 2)

        for item_idx, item_key in enumerate(item_order):
            positions.append(group_start + item_idx * item_spacing)
            box_data.append(task_scores[model_name][item_key])
            box_colors.append(item_colors[item_idx % len(item_colors)])
            box_item_keys.append(item_key)

    tick_labels = [model_labels.get(model, model) for model in model_order]
    return box_data, positions, box_colors, group_centers, tick_labels, box_item_keys


def _build_metrics_boxplot_data(
    task_scores: dict[str, dict[str, list[float]]],
) -> tuple[
    list[list[float]],
    list[float],
    list[str],
    list[float],
    list[str],
    list[str],
]:
    return _build_grouped_boxplot_data(
        task_scores,
        model_order=MODEL_ORDER,
        model_labels=MODEL_LABELS,
        item_order=METRIC_ORDER,
        item_colors=METRIC_COLORS,
        group_gap=GROUP_GAP,
        item_spacing=METRIC_SPACING,
    )


def _build_importance_boxplot_data(
    task_importances: dict[str, dict[str, list[float]]],
) -> tuple[
    list[list[float]],
    list[float],
    list[str],
    list[float],
    list[str],
    list[str],
]:
    return _build_grouped_boxplot_data(
        task_importances,
        model_order=IMPORTANCE_MODEL_ORDER,
        model_labels=IMPORTANCE_MODEL_LABELS,
        item_order=FEATURE_ORDER,
        item_colors=FEATURE_COLORS,
        group_gap=IMPORTANCE_GROUP_GAP,
        item_spacing=IMPORTANCE_FEATURE_SPACING,
    )


def _add_box_guide_lines(
    ax: plt.Axes,
    positions: list[float],
    *,
    label_y: float,
    color: str = GUIDE_LINE_COLOR,
    linestyle: str = GUIDE_LINE_STYLE,
    alpha: float = GUIDE_LINE_ALPHA,
    linewidth: float = GUIDE_LINE_WIDTH,
) -> None:
    """各箱ひげ図の中心を通る垂直点線を描き、ラベルとの対応を示す。"""
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for pos in positions:
        ax.plot(
            [pos, pos],
            [1.0, label_y],
            transform=transform,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
            zorder=0,
            clip_on=False,
        )


def _add_box_item_labels(
    ax: plt.Axes,
    positions: list[float],
    item_keys: list[str],
    item_labels: dict[str, str],
    *,
    fontsize: int,
    rotation: int,
    label_y: float,
) -> None:
    """各箱ひげ図の直下に項目ラベルを付ける（白黒印刷でも区別可能にする）。"""
    label_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for pos, item_key in zip(positions, item_keys):
        ax.text(
            pos,
            label_y,
            item_labels[item_key],
            ha="right",
            va="top",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=fontsize,
            transform=label_transform,
        )


def _add_box_metric_labels(
    ax: plt.Axes,
    positions: list[float],
    metric_keys: list[str],
) -> None:
    _add_box_item_labels(
        ax,
        positions,
        metric_keys,
        METRIC_LABELS,
        fontsize=BOX_LABEL_FONTSIZE,
        rotation=BOX_LABEL_ROTATION,
        label_y=BOX_LABEL_Y,
    )


def _add_feature_labels(
    ax: plt.Axes,
    positions: list[float],
    feature_keys: list[str],
) -> None:
    _add_box_item_labels(
        ax,
        positions,
        feature_keys,
        FEATURE_LABELS,
        fontsize=IMPORTANCE_BOX_LABEL_FONTSIZE,
        rotation=IMPORTANCE_BOX_LABEL_ROTATION,
        label_y=IMPORTANCE_BOX_LABEL_Y,
    )


def _add_model_labels(
    ax: plt.Axes,
    group_centers: list[float],
    model_labels: list[str],
    *,
    fontsize: int,
    label_y: float,
) -> None:
    """アルゴリズム名を項目ラベルの下に表示する。"""
    label_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for pos, label in zip(group_centers, model_labels):
        ax.text(
            pos,
            label_y,
            label,
            ha="center",
            va="top",
            fontsize=fontsize,
            transform=label_transform,
        )


def plot_metrics_boxplot(
    cv_fold_scores: dict[str, dict[str, dict[str, list[float]]]],
    *,
    output_dir: Path = OUTPUT_DIR,
    show: bool = False,
) -> Path:
    """全アルゴリズムを各 task サブプロット内に横並びで描画する。"""
    _validate_scores(cv_fold_scores)

    configure_matplotlib()
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=FIGURE_SIZE, sharey=True)
    if len(TASK_ORDER) == 1:
        axes = [axes]

    for ax, task_id in zip(axes, TASK_ORDER):
        box_data, positions, box_colors, group_centers, tick_labels, box_metric_keys = (
            _build_metrics_boxplot_data(cv_fold_scores[task_id])
        )

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(BOX_EDGE_COLOR)
            patch.set_alpha(BOX_ALPHA)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color(BOX_EDGE_COLOR)

        ax.set_xticks([])
        ax.set_ylim(*YLIM)
        _add_box_guide_lines(ax, positions, label_y=MODEL_LABEL_Y + 0.15)
        _add_box_metric_labels(ax, positions, box_metric_keys)
        _add_model_labels(
            ax,
            group_centers,
            tick_labels,
            fontsize=MODEL_LABEL_FONTSIZE,
            label_y=MODEL_LABEL_Y,
        )
        task_label = TASK_LABELS.get(task_id, task_id)
        ax.set_title(task_label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel(YLABEL, fontsize=12)
    fig.suptitle(FIGURE_TITLE, fontsize=14, fontweight="bold")

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in OUTPUT_FORMATS:
        output_path = output_dir / f"{OUTPUT_BASENAME}.{fmt}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        saved_paths.append(output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_paths[0]


def plot_importance_boxplot(
    cv_fold_importances: dict[str, dict[str, dict[str, list[float]]]],
    *,
    output_dir: Path = OUTPUT_DIR,
    show: bool = False,
) -> Path:
    """DT / RF / GB の特徴量重要度を各 task サブプロット内に横並びで描画する。"""
    _validate_importances(cv_fold_importances)

    configure_matplotlib()
    fig, axes = plt.subplots(
        1,
        len(TASK_ORDER),
        figsize=IMPORTANCE_FIGURE_SIZE,
        sharey=True,
    )
    if len(TASK_ORDER) == 1:
        axes = [axes]

    for ax, task_id in zip(axes, TASK_ORDER):
        box_data, positions, box_colors, group_centers, tick_labels, box_feature_keys = (
            _build_importance_boxplot_data(cv_fold_importances[task_id])
        )

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=IMPORTANCE_BOX_WIDTH,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(BOX_EDGE_COLOR)
            patch.set_alpha(BOX_ALPHA)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color(BOX_EDGE_COLOR)

        ax.set_xticks([])
        ax.set_ylim(*IMPORTANCE_YLIM)
        _add_box_guide_lines(ax, positions, label_y=IMPORTANCE_MODEL_LABEL_Y+ 0.18)
        _add_feature_labels(ax, positions, box_feature_keys)
        _add_model_labels(
            ax,
            group_centers,
            tick_labels,
            fontsize=IMPORTANCE_MODEL_LABEL_FONTSIZE,
            label_y=IMPORTANCE_MODEL_LABEL_Y,
        )
        task_label = TASK_LABELS.get(task_id, task_id)
        ax.set_title(task_label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel(IMPORTANCE_YLABEL, fontsize=12)
    fig.suptitle(IMPORTANCE_FIGURE_TITLE, fontsize=14, fontweight="bold")

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in OUTPUT_FORMATS:
        output_path = output_dir / f"{IMPORTANCE_OUTPUT_BASENAME}.{fmt}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        saved_paths.append(output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_paths[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="task0〜task2 の評価指標と特徴量重要度を箱ひげ図で描画する",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="画像の保存先ディレクトリ",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存後にウィンドウで表示する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not CV_FOLD_SCORES and not CV_FOLD_IMPORTANCES:
        raise SystemExit(
            "CV_FOLD_SCORES と CV_FOLD_IMPORTANCES がどちらも空です。"
            " export_latex_tables.py の出力を貼り付けてから実行してください。"
        )

    if CV_FOLD_SCORES:
        output_path = plot_metrics_boxplot(
            CV_FOLD_SCORES,
            output_dir=args.output_dir,
            show=args.show,
        )
        print(f"保存しました: {output_path}")

    if CV_FOLD_IMPORTANCES:
        output_path = plot_importance_boxplot(
            CV_FOLD_IMPORTANCES,
            output_dir=args.output_dir,
            show=args.show,
        )
        print(f"保存しました: {output_path}")


if __name__ == "__main__":
    main()
