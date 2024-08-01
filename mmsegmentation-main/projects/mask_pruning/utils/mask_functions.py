import torch


def logistic_function(x, k, b, epsilon=5e-5):
    """

    Args:
        x: the input tensor
        k: the exponent
        b: the shift on the x-axis
        epsilon:

    Returns:

    """
    # use clip to make it numerically stable
    e = torch.clip(-k * (x - b), max=10.0, min=-10.0)
    res = 1 / (1 + torch.exp(e))
    res[res != torch.clamp(res, epsilon)] = 0
    return res


def get_index_matrix(rows, colums, device="cuda"):
    """
    creates a 2D Tensor of the following form
    [[0, 0, 0, 0, ...]
    [1, 1, 1, 1, ...]
    [2, 2, 2, 2, ...]
    ...
    [n, n, n, n, ...]]

    Args:
        rows: number of rows
        colums: number of columns
        device: the device to save the created tensor to

    Returns:

    """
    arr = torch.zeros((rows, colums), requires_grad=False, device=device, dtype=torch.float)
    for i in range(rows):
        arr[i] = i
    return arr


def get_weighting_matrix(b, rows, columns, k=7, device="cuda"):
    """
    creates a matrix of the following form:
        [[0, 0, 0, 0, ...]
        [1, 1, 1, 1, ...]
        [2, 2, 2, 2, ...]
        ...
        [n, n, n, n, ...]]
    and applies to every element x in the matrix a sigmoid/logistic function of the form:
        f(x) = 1 / (1 + e^(-k * (x - b)))

    Args:
        b: the shift on the x-axis
        rows: number of rows
        columns: number of columns
        k: parameter that defines slope of the sigmoid/logistic function
        device:

    Returns:

    """
    res = logistic_function(get_index_matrix(rows, 1, device), k, b).expand(rows, columns)
    return res


def get_permutation_vector(weight):
    """
    gets the permutation vector by sorting the number of neurons in ascending order.
    The weight matrix is summed over in_size (dim=1) and ordered over out_size (dim=0)

    The function is the simplest estimation of an importance ranking of the neurons in a linear layer

    Args:
        weight: the weight matrix of size (out_size x in_size)

    Returns:
        the permutation vector in ascending order

    """
    return torch.argsort(torch.sum(torch.abs(weight), dim=1))