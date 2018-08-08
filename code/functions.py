import numpy as np
from numpy.linalg import norm
from functools import lru_cache
from tqdm import tqdm
from scipy.optimize import linprog

def get_weights_gap(code_matrix, dich_classifiers=None, weights_type=None):
    l, N = code_matrix.shape
    c = np.zeros(N+1)
    c[-1] = -1
    # размер A Nx (l*(l-1)/2)
    A_ub = []
    b_ub = np.zeros(l*(l-1)//2)
    for nu in range(l):
        for mu in range(nu+1, l):
            A_arr = []
            for j in range(N): # кол-во дихотомий
                diff_munu = code_matrix[nu][j] - code_matrix[mu][j]
                if weights_type is not None:
                    if weights_type == 'confusion_list':
                        score = dich_classifiers[j][weights_type][mu]#, nu].mean() #maybe dirty hack
                    else:
                        score = dich_classifiers[j][weights_type]
                    if diff_munu == 1:
                        diff_munu = score
                    else:
                        diff_munu = 1-score
                A_arr.append(-np.abs(diff_munu))
            A_arr.append(1)
            A_ub.append(A_arr)
    A_ub = np.array(A_ub)
    A_ub = np.vstack([A_ub, -np.eye(N+1)[:-1]]) # x_i >= 0
    b_ub = np.append(b_ub, np.zeros(N))
    A_eq = np.ones(N+1).reshape((1, -1))
    A_eq[0][-1] = 0
    b_eq = np.array(N).reshape((-1))
    opt_result = linprog(c, A_ub, b_ub, A_eq, b_eq, options={'disp': False})
    return opt_result['x'][:-1] # last value is gap

def ex(arr, j, i):
    return np.exp(-norm(arr[i] - arr[j])**2)

def p(arr, j, i):
    a = ex(arr, j, i)
    b = sum(ex(arr, k, i) for k in range(len(arr)) if k!=i)
    return a / b

def d(arr, i, i1, i2):
    # return np.abs(arr[i, i2] - arr[j, i2])
    return 2*(arr[i1, i2] - arr[i, i2])

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
    def d(i, i1):
        '''
        "Дистанция после дифференцирования" - то же самое, только arr == arr2 и i2 == i2
        '''
        dist = 2*(arr2[i1, i2] - arr2[i, i2])
        return dist
    
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

def get_mape(arr1, arr2, nan_coords):
    vec1 = []
    vec2 = []
    for j, (x,y) in enumerate(nan_coords):
        vec1.append(arr1[x, y])
        vec2.append(arr2[x, y])
    return mean_absolute_percentage_error(np.array(vec1), np.array(vec2))

def get_rmse(arr1, arr2, nan_coords):
    vec1 = []
    vec2 = []
    for j, (x,y) in enumerate(nan_coords):
        vec1.append(arr1[x, y])
        vec2.append(arr2[x, y])
    return np.sqrt(mean_squared_error(vec1, vec2))

def get_rmspe(arr1, arr2, nan_coords):
    vec1 = []
    vec2 = []
    for j, (x,y) in enumerate(nan_coords):
        vec1.append(arr1[x, y])
        vec2.append(arr2[x, y])
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    pi = np.abs(vec1-vec2) / vec1
    return np.mean(100*pi)

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
def get_acc(arr2, target):
    return 0
#     df_acc = pd.DataFrame(arr2)
#     df_acc['target'] = target
    forest = RandomForestClassifier()
    return cross_val_score(forest, arr2, target, scoring='accuracy', cv=7).mean()

def set_nans(df0, seed, num_nan_cols, nan_fraction):
    df = df0.copy()
    np.random.seed(seed)
    if num_nan_cols >= 0:
        nan_cols = np.random.random_integers(0, df.shape[1] - 1, num_nan_cols)
        for col in set(nan_cols):
            df.loc[df.sample(int(nan_fraction * len(df))).index, col] = np.nan
        nan_coords = np.array(np.where(df.isnull().values)).T
    else:
        all_pairs = np.array([[i,j] for i in range(df.shape[0]) for j in range(df.shape[1])])
        nan_places = np.random.choice(np.arange(0, df.size), size=int(nan_fraction*df.size), replace=False)
        nan_coords = all_pairs[nan_places]
        # df.iloc[nan_coors[:,0], nan_coors[:,1]] = None
        for x,y in nan_coords:
            df.iloc[x, y] = np.nan
            
    print('Num nan places: {}'.format(nan_coords.shape[0]))    
    df1 = df.loc[:, df.isnull().sum() == 0]
    df2 = df.fillna(df.mean())
    print(df1.shape, df2.shape)
    arr_nan = df.values # с пропусками
    arr_raw = df0.values # исходные
    arr_known = df1.values # суженные до известных признаков
    arr_pred = df2.values # текущие предсказанные 
    return df, df1, df2, arr_nan, arr_raw, arr_known, arr_pred, nan_coords

def Cnk(n, k):
    a = b = c = tmp = 1
    for i in range(1, n+1):
        tmp *= i
        if i == n-k:
            a = tmp
        if i == k:
            b = tmp
        if i == n:
            c = tmp
    return c / (a*b)

def predict_codeword(x, dich_classifiers):
    codeword = []
    for dich in dich_classifiers:
        clf = dich['model']
        codeword.append(clf.predict(x.reshape(1, -1)))
    return np.array(codeword).flatten()

def hamming(arr1, arr2, scores=None, weights=1):
#     print(arr1, arr2, scores)
    if scores is None:
        return (arr1 != arr2).sum()
    return ((arr1 != arr2)*scores*weights).sum() + ((arr1 == arr2)*(1-scores)*weights).sum()
    
def predict_class(x, dich_classifiers, code_matrix, score_type=None, weights=1, verbose=False):
    codeword = predict_codeword(x, dich_classifiers)
    if not score_type:
        hammings = np.array([hamming(codeword, class_code, weights=weights) for class_code in code_matrix])
    else:
        scores = np.array([d[score_type] for d in dich_classifiers])
        if score_type == 'confusion_list':
            # ПРОВЕРИТЬ ВЕРНО ЛИ ФОРМИРУЮТСЯ ОЦЕНКИ ТУТ
            hammings = np.array([hamming(codeword, class_code, scores.T[i], weights=weights) \
                                 for i, class_code in enumerate(code_matrix)])
        else:
            hammings = np.array([hamming(codeword, class_code, scores) for class_code in code_matrix])
    if verbose:
        print(hammings)
    indices = np.where(hammings == hammings.min())
    return np.random.choice(indices[0])

def predict_all(X_test, dich_classifiers, code_matrix, score_type=None, weight_type=None):
    if weight_type is None:
        weights = 1
    else:
        weights = get_weights_gap(code_matrix, dich_classifiers, score_type)
    preds = [predict_class(x, dich_classifiers, code_matrix, score_type, weights) for x in X_test]
    preds = np.array(preds)
    return preds

def make_random_dichs(l, N):
    code_matrix = None
    for i in tqdm(range(N), desc='Adding dich'):
        code_matrix = add_random_dich(l, code_matrix)
    
def add_random_dich(l=10, code_matrix=None):
    if code_matrix is None:
        # матрица пуста
        dich = np.random.randint(0, 2, l)
        while np.unique(dich).size == 1:
            dich = np.random.randint(0, 2, l)
        return dich.reshape((-1, 1))
    # матрица непуста
    dich = np.random.randint(0, 2, l)
    def does_dich_exist(dich, code_matrix):
        diff = (cm == dich).sum(axis=0)
        if diff.max() == l or diff.min() == 0:
            return True
        return False
    while np.unique(dich).size == 1 and not does_dich_exist(dich, code_matrix):
        dich = np.random.randint(0, 2, l)
#     print(code_matrix.shape, dich.shape)
    return np.hstack([code_matrix, dich.reshape((-1, 1))])