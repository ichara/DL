
x_train, y_train, x_test = load_fashionmnist()

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# logの中身が0になるのを防ぐ
def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=x))

def softmax(x):
    x -= x.max(axis=1, keepdims=True) # expのunderflow & overflowを防ぐ
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

# weights
W = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype('float32')
b = np.zeros(shape=(10,)).astype('float32')

# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

def train(x, t, eps=1.0):
    """
    :param x: np.ndarray, 入力データ, shape=(batch_size, 入力の次元数)
    :param t: np.ndarray, 教師ラベル, shape=(batch_size, 出力の次元数)
    :param eps: float, 学習率
    """
    global W, b
    
    batch_size = x.shape[0]
    
    # 順伝播
    y = softmax(np.matmul(x, W) + b) # shape: (batch_size, 出力の次元数)
    
    # 逆伝播
    cost = (- t * np_log(y)).sum(axis=1).mean()
    delta = y - t # shape: (batch_size, 出力の次元数)
    
    # パラメータの更新
    dW = np.matmul(x.T, delta) / batch_size # shape: (入力の次元数, 出力の次元数)
    db = np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size # shape: (出力の次元数,)
    W -= eps * dW
    b -= eps * db

    return cost

def valid(x, t):
    y = softmax(np.matmul(x, W) + b)
    cost = (- t * np_log(y)).sum(axis=1).mean()
    
    return cost, y

for epoch in range(10):
    # オンライン学習
    x_train, y_train = shuffle(x_train, y_train)
    for x, y in zip(x_train, y_train):
        cost = train(x[None, :], y[None, :], 1 / (epoch + 1))
    cost, y_pred = valid(x_valid, y_valid)
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
        epoch + 1,
        cost,
        accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
    ))
    if accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1)) >= 0.830:
        break;

y_pred = softmax(np.matmul(x_test, W) + b).argmax(axis=1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/submission_pred.csv', header=True, index_label='id')