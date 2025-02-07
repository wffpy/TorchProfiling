from textwrap import fill

import h5py
import prettytable as pt
import torch
import torch.nn.functional as F


def get_keys(h5f):
    keys = []

    def add_key(name):
        keys.append(name)

    h5f.visit(add_key)
    return keys


def load_data(h5f, key):

    dest = h5f[key]
    if dest.shape == ():
        return None
    return torch.tensor(h5f[key][:])


def compute_max_diff(tensor1, tensor2):
    diff = tensor1 - tensor2

    # Step 2: Compute the absolute value of the differences
    abs_diff = torch.abs(diff)

    # Step 3: Find the maximum value from the resulting tensor
    max_diff = torch.max(abs_diff)
    return max_diff.item()


def cosin(tensor1, tensor2):
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)
    cosine_similarity = F.cosine_similarity(
        tensor1_flat.unsqueeze(0).to(torch.float32),
        tensor2_flat.unsqueeze(0).to(torch.float32),
    )
    return cosine_similarity.item()


def compute_rmse(tensor1, tensor2):
    error = tensor1 - tensor2
    squared_error = error**2
    mean_squared_error = torch.mean(squared_error)
    rmse = torch.sqrt(mean_squared_error.to(torch.float32))
    return rmse.item()


def compute_mape(tensor1, tensor2):
    epsilon = 1e-10
    absolute_percentage_error = torch.abs((tensor1 - tensor2) / (tensor1 + epsilon)) * 100
    mape = torch.mean(absolute_percentage_error)
    return mape.item()


def compare(path1, path2):
    lhs_h5f = h5py.File(path1, "r")
    rhs_h5f = h5py.File(path2, "r")

    lhs_keys = get_keys(lhs_h5f)
    rhs_keys = get_keys(rhs_h5f)

    table = pt.PrettyTable(
        [
            fill("Status", width=10),
            fill("op name", width=100),
            fill("cosin", width=10),
            fill("rmse", width=10),
            fill("mape", width=10),
            fill("max_diff", width=10),
            fill("max_lhs", width=10),
            fill("max_rhs", width=10),
            fill("mean_lhs", width=10),
            fill("mean_rhs", width=10),
        ]
    )

    lhs_tensor_num = len(lhs_keys)
    rhs_tensor_num = len(rhs_keys)

    assert lhs_tensor_num == rhs_tensor_num, "lhs and rhs tensor number not equal, {} vs {}.".format(
        lhs_tensor_num, rhs_tensor_num
    )

    for key in rhs_keys:
        if not key in lhs_keys:
            print("not found: {} in lhs keys".format(key))
            continue

        lhs = load_data(lhs_h5f, key)
        rhs = load_data(rhs_h5f, key)

        if lhs is None or rhs is None:
            continue

        flat_lhs = lhs.to(torch.float32).view(-1)
        flat_rhs = rhs.to(torch.float32).view(-1)

        if flat_rhs.shape != flat_lhs.shape:
            print("skipping shape different op: {}, lhs shape: {}, rhs shape: {}".format(key, lhs.shape, rhs.shape))
            continue

        cos = cosin(flat_lhs, flat_rhs)
        rmse = compute_rmse(flat_lhs, flat_rhs)
        mape = compute_mape(flat_lhs, flat_rhs)
        max_diff = compute_max_diff(flat_lhs, flat_rhs)
        lhs_mean = flat_lhs.mean().item()
        rhs_mean = flat_rhs.mean().item()
        lhs_max = torch.max(flat_lhs).item()
        rhs_max = torch.max(flat_rhs).item()

        if abs(abs(cos) - 1) > 0.05:
            table.add_row(
                [
                    "F",
                    key,
                    cos,
                    rmse,
                    mape,
                    max_diff,
                    lhs_max,
                    rhs_max,
                    lhs_mean,
                    rhs_mean,
                ]
            )
        else:
            table.add_row(
                [
                    " ",
                    key,
                    cos,
                    rmse,
                    mape,
                    max_diff,
                    lhs_max,
                    rhs_max,
                    lhs_mean,
                    rhs_mean,
                ]
            )

    return table


# if __name__ == "__main__":
#     compare(sys.argv[1], sys.argv[2])
