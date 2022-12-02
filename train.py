from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import shutil

# Torch
import torch
from torch import nn,optim
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

from imblearn.over_sampling import ADASYN
from sklearn import preprocessing
from sklearn.preprocessing import (LabelBinarizer)

# Utils
from utils import parse_data, random_sample

# Models
from models import SPConv2D

np.random.seed(0)


def get_model(classes):
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=1),
        nn.ReLU(True),
        SPConv2D(in_channels=8, out_channels=16, alpha=0.8),
        SPConv2D(in_channels=16, out_channels=32, alpha=0.8),
        nn.ReLU(True),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(in_features=3200, out_features=len(classes)),
        nn.Softmax(dim=1)
    )


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
    train_df, test_df = label_binarize(train_adasyn, test_onehot)
    # End, data processing

    work_dir: Path = args.out
    work_dir.mkdir(parents=True, exist_ok=True)

    plot_save = work_dir.joinpath('plots')
    plot_save.mkdir(parents=True, exist_ok=True)

    classes = ['Dos', 'Probe', 'R2L', 'U2R', 'normal']

    x_train, y_train = parse_data(train_df, classes)
    x_test, y_test = parse_data(test_df, classes)

    # Convert 1D to 2D
    x_train = x_train.reshape((-1, 11, 11))
    x_test = x_test.reshape((-1, 11, 11))

    plt.rcParams.update({'font.size': 100})
    fig, axes = plt.subplots(2, 5, figsize=(200, 80), gridspec_kw={'height_ratios': [1, 1]})

    for i in range(2):
        for j in range(5):
            rnd = np.random.choice(range(len(x_test)), replace=False)
            axes[i, j].imshow(x_test[rnd])
            axes[i, j].grid(False)
            axes[i, j].axis('off')
            axes[i, j].set_title(classes[np.argmax(y_test[rnd])])
    fig.savefig(str(plot_save.joinpath("test_samples.png")))

    test_loader = DataLoader(
        data_utils.TensorDataset(torch.tensor(x_test.reshape((-1, 1, 11, 11))), torch.tensor(y_test)),
        batch_size=args.batch_size,
        num_workers=args.n_worker,
        shuffle=False)

    # Train Phase
    for class_idx in range(args.n_classiers):
        print(f'Classifier {class_idx + 1}')
        tmp_work_dir = work_dir.joinpath(f'classifier_{class_idx + 1}')
        best_models = tmp_work_dir.joinpath('best')
        best_models.mkdir(parents=True, exist_ok=True)

        x_prime, y_prime = random_sample(x_train, y_train, 52)
        #     x_prime2,y_prime2 = random_sample(x_train,y_train,52)

        train_loader = DataLoader(
            data_utils.TensorDataset(torch.tensor(x_prime.reshape((-1, 1, 11, 11))), torch.tensor(y_prime)),
            batch_size=args.batch_size,
            num_workers=args.n_worker,
            shuffle=True)

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        model = get_model()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

        train_loss_per_epoch = []
        val_loss_per_epoch = []

        train_acc_per_epoch = []
        val_acc_per_epoch = []

        last_acc = 0.

        max_val_acc = float('-inf')
        for epoch in range(args.epochs):
            train_loss = 0.0
            val_loss = 0.0

            # Train Phase
            train_true_labels = []
            train_pred_labels = []
            model = model.train(True)
            for idx, (payload, pay_label) in enumerate(train_loader):
                #         for idx, ((data1,lb1), (data2,lb2)) in enumerate(zip(train_loader,train_loader2)):

                #             lam = np.random.beta(ALPHA,ALPHA)
                #             payload = data1 * lam + (1 - lam) * data2
                #             pay_label = lb1 * lam + (1 - lam) * lb2

                payload = payload.float()
                pay_label = pay_label.float()

                payload = payload.to(device)
                pay_label = pay_label.to(device)

                opt.zero_grad()
                output = model(payload)
                loss = criterion(pay_label, output)
                loss.backward()
                opt.step()
                train_loss += loss.item()
                train_true_labels.append(np.argmax(pay_label.cpu().detach().numpy(), axis=1))
                train_pred_labels.append(np.argmax(output.cpu().detach().numpy(), axis=1))

            scheduler.step()

            train_true_labels = np.concatenate(train_true_labels)
            train_pred_labels = np.concatenate(train_pred_labels)
            train_acc = accuracy_score(train_true_labels, train_pred_labels)
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
                loss = criterion(pay_label, output)
                val_loss += loss.item()
                val_true_labels.append(np.argmax(pay_label.cpu().detach().numpy(), axis=1))
                val_pred_labels.append(np.argmax(output.cpu().detach().numpy(), axis=1))

            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(test_loader)

            val_true_labels = np.concatenate(val_true_labels)
            val_pred_labels = np.concatenate(val_pred_labels)

            val_acc = accuracy_score(val_true_labels, val_pred_labels)
            print(
                'Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss: {:.6f}\t Train Accuracy: {:.6f},\tValidation Accuracy: {:.6f}'.format(
                    epoch,
                    train_loss,
                    val_loss,
                    train_acc,
                    val_acc
                ))
            train_loss_per_epoch.append(train_loss)
            val_loss_per_epoch.append(val_loss)

            train_acc_per_epoch.append(train_acc)
            val_acc_per_epoch.append(val_acc)

            if val_acc > max_val_acc:
                print(f"[*] Validation {max_val_acc} to {val_acc}, model saved at {str(best_models)}")
                max_val = val_loss_per_epoch[-1]
                max_val_acc = val_acc
                last_acc = val_acc
                torch.save(model, best_models.joinpath(f'classifier.pth'))

        if (last_acc >= args.th):
            weight_path = tmp_work_dir.joinpath('weights')
            weight_path.mkdir(parents=True, exist_ok=True)
            torch.save(model, weight_path.joinpath(f'classifier.pth'))

            # Plot the results
            plt.rcParams.update({'font.size': 140})
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(200, 80))
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            ax_loss.plot(list(range(args.epochs)), train_loss_per_epoch, color=colors[0], label="Train Loss")
            ax_loss.plot(list(range(args.epochs)), val_loss_per_epoch, color=colors[1], label="Valid Loss", linestyle="-.")
            ax_acc.plot(list(range(args.epochs)), train_acc_per_epoch, color=colors[0], label="Train Accuracy")
            ax_acc.plot(list(range(args.epochs)), val_acc_per_epoch, color=colors[1], label="Valid Accuracy", linestyle="-.")
            ax_loss.grid(False)
            ax_loss.legend()
            ax_loss.set_title("Loss")
            ax_acc.grid(False)
            ax_acc.set_title("accuracy")
            ax_acc.legend()
            fig.savefig(tmp_work_dir.joinpath('result.jpg'))
            plt.clf()

        else:
            shutil.rmtree(str(tmp_work_dir))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', help='Dataset directory', type=Path, default='./Data')
    parser.add_argument('--out', help='Output path to save results', type=Path, default='./results/xp1')

    # Hyperparameter
    parser.add_argument('--lr', help='Learning rate.', type=float, default=.1)
    parser.add_argument('--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=50)
    parser.add_argument('--n_sample', help='number of samples', type=int, default=52)
    parser.add_argument('--n_classifiers', help='number of classifiers', type=int, default=33)
    parser.add_argument('--n_worker', help='dataloader workers', type=int, default=2)
    parser.add_argument('--th', help='save best classifiers threshold', type=float, default=.36)

    main(parser.parse_args())
