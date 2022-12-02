from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn import metrics

# Torch
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

from imblearn.over_sampling import ADASYN
from sklearn import preprocessing
from sklearn.preprocessing import (LabelBinarizer)

from utils import parse_data


def read_dataframe(train_path, test_path):
    feature = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
               "urgent", "hot",
               "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
               "num_file_creations", "num_shells",
               "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
               "serror_rate", "srv_serror_rate",
               "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
               "dst_host_count", "dst_host_srv_count",
               "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
               "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]
    train = pd.read_csv(train_path, names=feature)
    test = pd.read_csv(test_path, names=feature)

    train.drop(['difficulty', 'num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['difficulty', 'num_outbound_cmds'], axis=1, inplace=True)

    return train, test


def label_mapping(df):
    """
    this function specificly is used for NSL-KDD dataset
    """

    df.label.replace(
        ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'apache2', 'processtable', 'udpstorm'],
        'Dos', inplace=True)
    df.label.replace(
        ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 'sendmail',
         'named', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm'], 'R2L', inplace=True)
    df.label.replace(['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'], 'Probe', inplace=True)
    df.label.replace(['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm', 'httptunnel'],
                     'U2R', inplace=True)

    return df


def normalization(train, test):
    numeric_col_list = train.select_dtypes(include='number').columns
    train_df_numeric_col = train[numeric_col_list].values
    min_max_scaler = preprocessing.MinMaxScaler()
    train_df_numeric_col_scaled = min_max_scaler.fit_transform(train_df_numeric_col)
    train_df_numeric_col_scaled = pd.DataFrame(train_df_numeric_col_scaled, columns=numeric_col_list)
    train[numeric_col_list] = train_df_numeric_col_scaled

    test_df_numeric_col = test[numeric_col_list].values
    test_df_numeric_col_scaled = min_max_scaler.transform(test_df_numeric_col)
    test_df_numeric_col_scaled = pd.DataFrame(test_df_numeric_col_scaled, columns=numeric_col_list)
    test[numeric_col_list] = test_df_numeric_col_scaled

    return train, test


def onehot_encoding(train, test):
    label_feature_name = 'label'

    categorical_column_list = list(train.select_dtypes(include='object').columns)
    categorical_column_list.remove(label_feature_name)
    print(categorical_column_list)

    one_hot_train = pd.get_dummies(train, columns=categorical_column_list)
    one_hot_test = pd.get_dummies(test, columns=categorical_column_list)

    print('train columns:', len(one_hot_train.columns))
    print('test columns:', len(one_hot_test.columns))

    diff = list(set(one_hot_train.columns) - set(one_hot_test.columns))
    print('difference between train and test that must be rescale:', diff)

    # check if test and train don't have same features after onehot-encoding, so upcoming lines of code of this function
    # rescale test and train dataset
    df_test = pd.DataFrame(data=None, columns=one_hot_train.columns)
    zero_values = [0] * len(one_hot_test)
    for c in df_test.columns:
        if c not in list(one_hot_test.columns):
            df_test[str(c)] = zero_values
        else:
            df_test[str(c)] = one_hot_test[str(c)]

    train_label_col = one_hot_train.pop(label_feature_name)
    one_hot_train.insert(len(one_hot_train.columns), label_feature_name,
                         train_label_col)  # position parameter for insert function starts with 0 so we dont need to len(one_hot_train.columns)+1 (after pop function)

    test_label_col = df_test.pop(label_feature_name)
    df_test.insert(len(df_test.columns), label_feature_name, test_label_col)

    train = one_hot_train
    test = df_test

    print('difference between train and test after re-scaling', len(list(set(train.columns) - set(test.columns))))

    print('train columns:', len(train.columns))
    print('test columns:', len(test.columns))

    return train, test


def data_augmentation(train):
    X_train_adasyn = train.loc[:, :'flag_SH']
    y_train_adasyn = train['label']

    labels_frequency = y_train_adasyn.value_counts()
    print('frequency of the labels before augmentation', labels_frequency)

    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train_adasyn, y_train_adasyn)
    train_adasyn = pd.concat([X_adasyn, y_adasyn], axis=1)

    return train_adasyn


def label_binarize(train, test):
    # create an object of label binarizer, then fit on train labels
    LabelBinarizerObject_fittedOnTrainLabel = LabelBinarizer().fit(train['label'])

    # transform train labels with that object
    TrainBinarizedLabel = LabelBinarizerObject_fittedOnTrainLabel.transform(train['label'])

    # convert transformed labels to dataframe
    TrainBinarizedLabelDataFrame = pd.DataFrame(TrainBinarizedLabel,
                                                columns=LabelBinarizerObject_fittedOnTrainLabel.classes_)

    # concatenate training set after drop 'label' with created dataframe of binarized labels
    train = pd.concat([train.drop(['label'], axis=1), TrainBinarizedLabelDataFrame], axis=1)

    TestBinarizedLabel = LabelBinarizerObject_fittedOnTrainLabel.transform(test['label'])
    TestBinarizedLabelDataFrame = pd.DataFrame(TestBinarizedLabel,
                                               columns=LabelBinarizerObject_fittedOnTrainLabel.classes_)
    test = pd.concat([test.drop(['label'], axis=1), TestBinarizedLabelDataFrame], axis=1)

    return train, test


def evaluate_v1(true_label, pred_label) -> dict:
    conf_mat = metrics.confusion_matrix(true_label, pred_label)
    print(conf_mat)

    cls_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    metric_param = {
        'total_acc': metrics.accuracy_score(true_label, pred_label),
        'accuracy': cls_acc,

        'micro_f1': metrics.f1_score(true_label, pred_label, average='micro'),
        'macro_f1': metrics.f1_score(true_label, pred_label, average='macro'),
        'weighted_f1': metrics.f1_score(true_label, pred_label, average='weighted'),
        'f1': metrics.f1_score(true_label, pred_label, average=None),

        'micro_recall': metrics.recall_score(true_label, pred_label, average='micro'),
        'macro_recall': metrics.recall_score(true_label, pred_label, average='macro'),
        'weighted_recall': metrics.recall_score(true_label, pred_label, average='weighted'),
        'recall': metrics.recall_score(true_label, pred_label, average=None),

        'micro_precision': metrics.precision_score(true_label, pred_label, average='micro'),
        'macro_precision': metrics.precision_score(true_label, pred_label, average='macro'),
        'weighted_precision': metrics.precision_score(true_label, pred_label, average='weighted'),
        'precision': metrics.precision_score(true_label, pred_label, average=None),
    }

    false_positive = conf_mat.sum(axis=0) - np.diag(conf_mat)
    false_negative = conf_mat.sum(axis=1) - np.diag(conf_mat)
    true_positive = np.diag(conf_mat)
    true_negative = conf_mat.sum() - (false_positive + false_negative + true_positive)

    metric_param['false_alarm_rate'] = false_positive / (false_positive + true_negative)
    metric_param['detection_rate'] = true_positive / (true_positive + false_negative)

    print(metrics.classification_report(true_label, pred_label))
    return metric_param


def main(args: Namespace) -> None:
    train_path_ = args.dir.joinpath('KDDTrain+.txt')
    test_path_ = args.dir.joinpath('KDDTest+.txt')

    # Start, data processing
    train_df, test_df = read_dataframe(train_path=train_path_, test_path=test_path_)
    train_df = label_mapping(train_df)
    test_df = label_mapping(test_df)
    train_norm, test_norm = normalization(train_df, test_df)
    train_onehot, test_onehot = onehot_encoding(train_norm, test_norm)
    train_adasyn = data_augmentation(train_onehot)
    _, test_df = label_binarize(train_adasyn, test_onehot)
    # End, data processing
    classes = ['Dos', 'Probe', 'R2L', 'U2R', 'normal']

    x_test, y_test = parse_data(test_df, classes)

    # Convert 1D to 2D
    x_test = x_test.reshape((-1, 11, 11))

    test_loader = DataLoader(
        data_utils.TensorDataset(torch.tensor(x_test.reshape((-1, 1, 11, 11))), torch.tensor(y_test)),
        batch_size=args.batch_size,
        num_workers=args.n_worker,
        shuffle=False)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    for m_p in args.pretrain.glob('*.pth'):
        model = torch.load(str(m_p))
        # Test Phase
        val_true_labels = []
        val_pred_labels = []
        model = model.train(False)
        for payload, pay_label in test_loader:
            payload = payload.float()
            pay_label = pay_label.float()

            payload = payload.to(device)
            pay_label = pay_label.to(device)

            output = model(payload)
            val_true_labels.append(np.argmax(pay_label.cpu().detach().numpy(), axis=1))
            val_pred_labels.append(np.argmax(output.cpu().detach().numpy(), axis=1))

        val_true_labels = np.concatenate(val_true_labels)
        val_pred_labels = np.concatenate(val_pred_labels)

        cfm = metrics.confusion_matrix(val_true_labels, val_pred_labels, labels=list(range(5)))
        eval_metrics = evaluate_v1(val_true_labels, val_pred_labels)

        for key, value in eval_metrics.items():
            if isinstance(value, np.float64):
                line = f"{key} :\t{value}"
                print(line)
            else:
                head = key.capitalize()
                print(head)
                for idx, class_name in enumerate(classes):
                    line = f'\t{class_name}:\t{value[idx]}'
                    print(line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', help='Dataset directory', type=Path, default='./Data')
    parser.add_argument('--pretrain', help='pretrained directory', type=Path, default='./Pretrains')
    main(parser.parse_args())
