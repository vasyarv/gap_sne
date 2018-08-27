import numpy as np
from numpy.linalg import norm
from functools import lru_cache
from tqdm import tqdm
from scipy.optimize import linprog
from sklearn.metrics import accuracy_score, f1_score

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'errorbar.capsize': 2})

def sq(a):
    return np.dot(a, a)

def cluster_score(data, target, score_type='trace_w'):
    # target 0...max
    num_class = target.max() + 1
    score = 0
    for i in range(num_class):
        s = 0
        sub_data = data[target==i]
        mean_vector = sub_data.mean(axis=0)
        for x in sub_data:
            s += sq(x-mean_vector)
        if score_type != 'trace_w':
            s /= len(sub_data)
        score += s
    return score

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
    if len(indices[0]) == 0:
        print(hammings, hammings.min(), score_type, scores)
    return np.random.choice(indices[0])

def predict_all(X_test, dich_classifiers, code_matrix, score_type=None, weight_type=None):
    if weight_type is None:
        weights = np.array([1]*len(dich_classifiers))
    elif weight_type == -1:
        weights = get_weights_gap(code_matrix, dich_classifiers, None)
    else:
        weights = get_weights_gap(code_matrix, dich_classifiers, weight_type)
    num_real_dich = (weights > np.median(weights)/100).sum()
#     print('Num dich = {}/{}'.format(num_real_dich, len(weights)))
#     print(weights)
    preds = [predict_class(x, dich_classifiers, code_matrix, score_type, weights) for x in X_test]
    preds = np.array(preds)
    return preds, num_real_dich

def int2bin(val, l):
    res = np.zeros(l)
    i = 0
    while val>0:
        res[i] = val&1
        val = val>>1     # val=val/2
        i += 1
    return res[::-1] 

def add_dich(dich, code_matrix=None):
    if code_matrix is None:
        return dich.reshape((-1, 1))
    return np.hstack([code_matrix, dich.reshape((-1, 1))])


def make_random_dichs(l, N):
    if N > 2**(l-1) - 1:
        N = 2**(l-1) - 1
        print('Dich Num reduced to max={}'.format(N))
    code_matrix = None
    binary_dich_numbers = np.random.choice(np.arange(0, 2**(l-1) - 1), N, replace=False)
    for dich in tqdm(binary_dich_numbers, desc='Adding dich'):
        binary_dich = int2bin(dich+1, l)
        code_matrix = add_dich(binary_dich, code_matrix)
    return code_matrix

def make_random_dichs_old(l, N):
    code_matrix = None
    for i in tqdm(range(N), desc='Adding dich'):
        code_matrix = add_random_dich(l, code_matrix)
    return code_matrix

def make_local_optimal_dichotomy(cur_dich, code_matrix, score_function, verbose=0):
    cur_score = score_function(cur_dich, code_matrix)
    next_score = cur_score
    while True:
        next_dich = cur_dich.copy()
        next_scores = np.zeros(len(cur_dich)) - 1
        for i in range(len(cur_dich)):
            next_dich = cur_dich.copy()
            next_dich[i] = 1 - next_dich[i]
            if not does_dich_exist(next_dich, code_matrix): #дихотомия нормальная
                next_scores[i] = score_function(next_dich, code_matrix)
        next_scores = np.array(next_scores)
        next_score = next_scores.max()
        #print(next_scores)
        if next_score <= cur_score: #идем только на повышение, но можно скор сделать поменьше
            break
        cur_score = next_score
        best_index = np.random.choice(np.flatnonzero(next_scores == next_score)) # it is random of the best
        if verbose > 0:
            print(cur_dich)
        if verbose > 1:
            print(next_score, best_index)
        cur_dich[best_index] = 1 - cur_dich[best_index]
#     if cur_dich.max() == cur_dich.min():
#         print(next_scores)
    return cur_dich

def make_code_matrix_local(l, N, score_function, verbose=1):
    code_matrix = None
    for i in tqdm(range(N)):
        new_dich = np.random.randint(0, 2, l)
        new_dich = make_local_optimal_dichotomy(new_dich.copy(), code_matrix, score_function, verbose)
        code_matrix = add_dich(new_dich, code_matrix)
    return code_matrix
    
def add_random_dich(l=10, code_matrix=None):
    if code_matrix is None:
        # матрица пуста
        dich = np.random.randint(0, 2, l)
        while np.unique(dich).size == 1:
            dich = np.random.randint(0, 2, l)
        return dich.reshape((-1, 1))
    # матрица непуста
    dich = np.random.randint(0, 2, l)
    
    while does_dich_exist(dich, code_matrix):
        dich = np.random.randint(0, 2, l)
#     print(code_matrix.shape, dich.shape)
    return np.hstack([code_matrix, dich.reshape((-1, 1))])

def does_dich_exist(dich, code_matrix):
    if code_matrix is None:
        return False
    l = code_matrix.shape[0]
    if dich.max() == 0 or dich.min() == 1:
        return True # trivial dich
    diff = (code_matrix.T == dich).sum(axis=1)
    if diff.max() == l or diff.min() == 0:
        return True
    return False
    
def train_dichs(code_matrix, X_train, y_train, X_test, y_test, BaseClassifier, params=None):
    dich_classifiers = []
    l, N = code_matrix.shape
    for i in tqdm(range(N), desc='Training dich classifiers'):
        if params is None:
            clf = BaseClassifier()
        else:
            clf = BaseClassifier(**params)
        X = X_train
        y_classes = code_matrix.T[i]
        y = np.array([y_classes[i] for i in y_train])
        clf.fit(X, y)
        y_pred = clf.predict(X_test)
        y_true = np.array([y_classes[i] for i in y_test])
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        true_mask = (y_pred == y_true)
        confusion_list = np.array([np.sum(true_mask*(y_test==i))/np.sum(y_test==i) for i in range(l)])
        dich_classifiers.append({'model': clf, 'accuracy': accuracy, 
                                 'f1': f1, 'confusion_list': confusion_list})
    return dich_classifiers


def plot_approach(df_, dataset='digits', 
                approach='random', dich_range=[20,200],
                xticks=np.arange(20, 210, 10),
                yticks=np.arange(0, 1., 0.005),
                title='Сравнение точности при взвешивании по F-мере',
                clf='linearsvc'):
    df = df_.copy()
    df.sort_values(by=['dataset', 'num_real_dich'], inplace=True)
    df.drop_duplicates(subset=['dataset', 'num_real_dich', 'approach'], inplace=True)
    df = df[(df['num_real_dich'] > dich_range[0]) & (df['num_real_dich'] <= dich_range[1])]
    df = df[df['clf'] == clf]
    sub_df = df[(df['dataset'] == dataset) & (df['approach'] == approach)]
    if len(sub_df) == 0:
        return None
    
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    x = sub_df['num_real_dich'].values
    y = sub_df['ecoc_mean'].values
    error = sub_df['ecoc_std'].values
    plt.errorbar(x, y, yerr=error, fmt='-o', label='Стандартное расстояние Хемминга')

    y = sub_df['accuracy_mean'].values
    error = sub_df['accuracy_std'].values
    plt.errorbar(x, y, yerr=error, fmt='-o', label='Взвешенное по вероятности классификации')

    y = sub_df['f1_mean'].values
    error = sub_df['f1_std'].values
    plt.errorbar(x, y, yerr=error, fmt='-o', label='Взвешенное по F-мере')

    y = sub_df['confusion_list_mean'].values
    error = sub_df['confusion_list_std'].values
    plt.errorbar(x, y, yerr=error, fmt='-o', label='Взвешенное по спискам неточностей')

    plt.legend(loc='lower right', fontsize=16)
    plt.title('Случайное построение дихотомической матрицы', fontsize=16)

    plt.xlabel('Количество дихотомий в матрице', fontsize=14)
    plt.ylabel('Точность (accuracy)', fontsize=14)

    plt.grid()
    return plt
    
    
def plot_score(df_, 
                dataset='digits', 
                score_type='f1', 
                xticks=np.arange(20, 60, 5), 
                yticks=np.arange(0, 1., 0.01), 
                approaches=['random'],
                legends=['Случайное построение дихотомической матрицы'],
                dich_range=[20,60],
                title='Сравнение точности при взвешивании по F-мере',
                clf='linearsvc'):
    df = df_.copy()
    df.sort_values(by=['dataset', 'num_real_dich'], inplace=True)
    df.drop_duplicates(subset=['dataset', 'num_real_dich', 'approach'], inplace=True)
    df = df[(df['num_real_dich'] > dich_range[0]) & (df['num_real_dich'] <= dich_range[1])]
    df = df[df['clf'] == clf]
    if len(df) == 0:
        return None
    assert len(approaches) == len(legends)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    for i in range(len(approaches)):
        approach = approaches[i]
        legend = legends[i]
        sub_df = df[(df['dataset'] == dataset) & (df['approach'] == approach)]
        x = sub_df['num_real_dich'].values
        y = sub_df[score_type+'_mean'].values
        error = sub_df[score_type+'_std'].values
        plt.errorbar(x, y, yerr=error, fmt='-o', label=legend)
    plt.legend(loc='lower right', fontsize=16)
    plt.title(title, fontsize=16)
    plt.xlabel('Количество дихотомий в матрице', fontsize=14)
    plt.ylabel('Точность', fontsize=14)
    plt.grid()
    return plt