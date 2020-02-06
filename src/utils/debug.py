def print_info(image, prefix: str = ''):
    print(f'{prefix}\t\t{image.shape}\t{image.min()}\t{image.max()}\t{image.dtype}')
