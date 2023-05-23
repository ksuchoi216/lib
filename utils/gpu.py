import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device('mps:0')
    else:
        device = torch.device('cpu')
    return device


def get_device222(gpu_num=0, model=None):
    cuda_available = torch.cuda.is_available()
    print(f'cuda is available: {cuda_available}')  # Number of gpus

    device = torch.device(
        f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    if cuda_available:
        model = model.to('cuda')
    if multi_gpu:
        model = nn.DataParallel(model)
