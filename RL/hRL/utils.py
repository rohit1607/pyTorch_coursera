import os

def make_dir(path, raise_error=False):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print(f"Created Dir: {path}")
    else:
        print(f"The path '{path}' already exists")
        if raise_error:
            raise NameError('path already exists')


# make_dir('/home/rohit/Documents/Research/pyTorch_coursera/RL/hRL/test')

