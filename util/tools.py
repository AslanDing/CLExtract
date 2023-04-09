import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from torch.utils.data import Dataset, TensorDataset

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        print("SEED ",seed)
        seed += 1
        np.random.seed(seed)
        print("SEED ",seed)
        seed += 1
        torch.manual_seed(seed)
        print("SEED ",seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
                print("SEED ", seed)

    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]


def npcutout(x, max_ratio=0.2):
    max_leng = int(len(x) * max_ratio)
    leng = random.randint(0, max_leng)
    start = random.randint(0, len(x) - leng)
    end = min(len(x), start + leng)
    res = x.copy()
    res[start:end] = 0
    return res


def torchcutout(x, ratio=0.2):
    oleng = x.shape[0]
    max_leng = int(oleng * ratio)
    leng = random.randint(0, max_leng)
    start = random.randint(0, oleng - 1)
    end = min(oleng, start + leng)
    res = x.clone()
    res[start:end] = 0
    return res


def npflip(x, ratio=-1):
    sample = np.random.rand(len(x))
    mask = (np.sign(sample - ratio) + 1) / 2
    mask = mask.astype(int)
    return mask * x + (1 - x) * (1 - mask)


def torchflip(x, ratio=0.2, device='cpu'):
    sample = torch.rand(len(x)).to(device)
    mask = (torch.sign(sample - ratio) + 1) / 2
    # print(mask.device)
    mask = mask.type(torch.int)
    # print(mask.device)
    mask.to(x.device)
    # print(x.device)
    # print(mask.device)
    # temp = mask * x
    # temp1 = (1 - x) * (1 - mask)
    
    return mask * x + (1 - x) * (1 - mask)


def batch_flip(x, ratio=0.2, device='cpu'):
    x_flat = x.view(-1)
    x_flip_flat = torchflip(x_flat, ratio, device)
    return x_flip_flat.view(x.shape)


def np_bit_loss(x, ratio=-1):
    sample = np.random.rand(len(x))
    mask = (np.sign(sample - ratio) + 1) / 2
    mask = mask.astype(int)
    return mask * x


def torch_bit_loss(x, ratio=0.2, device='cpu'):
    sample = torch.rand(len(x)).to(device)
    mask = (torch.sign(sample - ratio) + 1) / 2
    mask = mask.type(torch.int)
    return mask * x


def batch_bit_loss(x, ratio=0.2, device='cpu'):
    x_flat = x.view(-1)
    x_flip_flat = torch_bit_loss(x_flat, ratio, device)
    return x_flip_flat.view(x.shape)


def np_bit_aug(x, ratio=-1, flip_to_corrupt=0.5):
    flip_x = npflip(x, ratio)
    loss_x = np_bit_loss(x, ratio)
    sample = np.random.rand(len(x))
    mask = (np.sign(sample - flip_to_corrupt) + 1) / 2  # 0 for flip
    return flip_x * (1 - mask) + loss_x * mask


def torch_bit_aug(x, ratio=0.2, flip_to_corrupt=0.5, device='cpu'):
    flip_x = torchflip(x, ratio, device)
    loss_x = torch_bit_loss(x, ratio, device)
    sample = torch.rand(len(x)).to(device)
    mask = (torch.sign(sample - flip_to_corrupt) + 1) / 2
    return flip_x * (1 - mask) + loss_x * mask


def batch_bit_aug(x, ratio=0.2, flip_to_lost=0.5, device='cpu'):
    #     x_flat = x.view(-1)
    x_flat = torch.flatten(x)
    x_flip_flat = torch_bit_aug(x_flat, ratio, flip_to_lost, device)
    return x_flip_flat.view(x.shape)


def binariz(v, thres=0.5):
    if isinstance(v,torch.Tensor):
        result = torch.where(v<thres,torch.zeros_like(v),torch.ones_like(v))
    # if v >= thres:
    #     return 1
    # else:
    #     return 0
    return result

def accConfusion(y_test, y_pred_test):
    acc = accuracy_score(y_test, y_pred_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    return acc, tn, fp, fn, tp