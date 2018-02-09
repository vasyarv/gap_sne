import numpy as np
from numpy.linalg import norm
from functools import lru_cache
from tqdm import tqdm

def ex(arr, j, i):
    return np.exp(-norm(arr[i] - arr[j])**2)

def p(arr, j, i):
    a = ex(arr, j, i)
    b = sum(ex(arr, k, i) for k in range(len(arr)) if k!=i)
    return a / b

def d(arr, i, j, i2):
    return np.abs(arr[i, i2] - arr[j, i2])

def norm1(i, j):
    return norm(arr1[i] - arr1[j])**2

def cost(arr1, arr2):
    @lru_cache(maxsize=None)
    def norm1(i, j):
        return norm(arr1[i] - arr1[j])**2
    @lru_cache(maxsize=None)
    def ex1(i, j):
        return np.exp(-norm1(i, j))
    @lru_cache(maxsize=None)
    def p1(j, i):
        a = ex1(j, i)
        b = sum(ex1(k, i) for k in range(len(arr1)) if k!=i)
        return a / b
    @lru_cache(maxsize=None)
    def norm2(i, j):
        return norm(arr2[i] - arr2[j])**2
    @lru_cache(maxsize=None)
    def ex2(i, j):
        return np.exp(-norm2(i, j))
    @lru_cache(maxsize=None)
    def p2(j, i):
        a = ex2(j, i)
        b = sum(ex2(k, i) for k in range(len(arr2)) if k!=i)
        return a / b
    s = 0
    for i in range(len(arr1)):
        for j in range(len(arr1)):
            s += p1(j, i) * np.log(p1(j, i) / p2(j, i))
    return s


def get_grad(arr1, arr2, i1, i2):
    '''
    arr1 - массив без пропусков(укороченный)
    arr2 - массив с прочерками(удлиенный)
    i1, i2 -  координаты nan
    '''
    @lru_cache(maxsize=None)
    def norm1(i, j):
        return norm(arr1[i] - arr1[j])

    @lru_cache(maxsize=None)
    def ex1(i, j):
        return np.exp(-norm1(i, j))

    @lru_cache(maxsize=None)
    def p1(j, i):
        a = ex1(j, i)
        b = sum(ex1(k, i) for k in range(len(arr1)) if k!=i)
        return a / b

    @lru_cache(maxsize=None)
    def norm2(i, j):
        return norm(arr2[i] - arr2[j])
    @lru_cache(maxsize=None)
    def ex2(i, j):
        return np.exp(-norm2(i, j))
    @lru_cache(maxsize=None)
    def p2(j, i):
        a = ex2(j, i)
        b = sum(ex2(k, i) for k in range(len(arr2)) if k!=i)
        return a / b
    
    @lru_cache(maxsize=None)
    def d(i, j):
        '''
        "Дистанция после дифференцирования" - то же самое, только arr == arr2 и i2 == i2
        '''
        a = np.abs(arr2[i, i2] - arr2[j, i2])
        return a
    def get_i_part(i):
        '''
        считаем i часть суммы
        '''
        s = 0
        s += p1(i1, i) + p1(i, i1)
        s -= p2(i1, i)*(1 + p1(i, i))
        s -= p2(i, i1)*(1 + p1(i1, i1))
        return s * d(i, i1)
    # if verbose:
    #     grad = sum(get_i_part(i) for i in tqdm(range(len(arr1))) if i!=i1)
    # else:
    grad = sum(get_i_part(i) for i in range(len(arr1)) if i!=i1)
    return grad

def get_full_grad(arr1, arr2, nan_coords, verbose=False):
    '''
    arr1 - массив без пропусков(укороченный)
    arr2 - массив с прочерками(удлиенный)
    i1, i2 -  координаты nan
    '''
    grads = []
    if verbose:
        for i1, i2 in tqdm(nan_coords):
            grads.append(get_grad(arr1, arr2, i1, i2))
    else:
        for i1, i2 in nan_coords:
            grads.append(get_grad(arr1, arr2, i1, i2))
    return np.array(grads)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def get_mae(arr1, arr2, nan_coords):
    vec1 = []
    vec2 = []
    for j, (x,y) in enumerate(nan_coords):
        vec1.append(arr1[x, y])
        vec2.append(arr2[x, y])
    return mean_absolute_error(vec1, vec2)

def get_msqe(arr1, arr2, nan_coords):
    vec1 = []
    vec2 = []
    for j, (x,y) in enumerate(nan_coords):
        vec1.append(arr1[x, y])
        vec2.append(arr2[x, y])
    return mean_squared_error(vec1, vec2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
def get_acc(arr2, target):
    return 0
#     df_acc = pd.DataFrame(arr2)
#     df_acc['target'] = target
    forest = RandomForestClassifier()
    return cross_val_score(forest, arr2, target, scoring='accuracy', cv=7).mean()