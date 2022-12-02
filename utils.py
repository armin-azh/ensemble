import numpy as np


def parse_data(df, classes):
    """
    Convert Dataframe to NumPy
    :param df:
    :param classes:
    :return:
    """
    glob_cl = set(range(len(df.columns)))
    cl_idx = set([df.columns.get_loc(cl) for cl in classes])
    target_feature_idx = list(glob_cl.difference(cl_idx))
    cl_idx = list(cl_idx)
    dt = df.iloc[:, target_feature_idx]
    lb = df.iloc[:, cl_idx]
    assert len(dt) == len(lb), 'Something Wrong!!\nnumber of data is not equal to labels'
    return dt.to_numpy(), lb.to_numpy()


def random_sample(x_data, x_label, n_sample):
    lb_repo = []
    dt_repo = []
    for idy in range(x_label.shape[1]):
        class_idx = np.where(x_label[:, idy] == 1)[0]
        tmp_lb = x_label[class_idx, :]
        tmp_dt = x_data[class_idx, :]
        if len(class_idx) <= n_sample:
            lb_repo.append(tmp_lb)
            dt_repo.append(tmp_dt)
            continue

        choices = np.random.choice(range(len(class_idx)), n_sample)
        lb_repo.append(tmp_lb[choices, :])
        dt_repo.append(tmp_dt[choices, :])

    dt = np.concatenate(dt_repo)
    lb = np.concatenate(lb_repo)
    return dt, lb
