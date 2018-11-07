
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(1234)
random_state = 42

### レイヤー定義 ###

class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], padding='VALID'):
        # Heの初期値
        fan_in = np.prod(filter_shape[:3]) # filter_shape: (縦の次元数)x(横の次元数)x(入力チャンネル数)x(出力チャンネル数)
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/fan_in),
                        high=np.sqrt(6/fan_in),
                        size=filter_shape
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b') # バイアスはフィルタごとなので, 出力フィルタ数と同じ次元数
        self.function = function
        self.strides = strides
        self.padding = padding

    def __call__(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)    
    
class Pooling:
    def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
    
    def __call__(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)    
    
class Flatten:
    def __call__(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))    

class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # He Initialization
        # in_dim: 入力の次元数、out_dim: 出力の次元数
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/in_dim),
                        high=np.sqrt(6/in_dim),
                        size=(in_dim, out_dim)
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def __call__(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)    
    
def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))    

### ネットワーク ###

x_train, x_test, t_train = load_mnist()
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.1, random_state=random_state)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])

# (縦の次元数)x(横の次元数)x(チャネル数)
h = Conv((3, 3, 1, 30), tf.nn.relu)(x)  # 28x28x 1 -> 26x26x30
h = Pooling((1, 2, 2, 1))(h)            # 26x26x30 -> 13x13x30
h = Conv((4, 4, 30, 60), tf.nn.relu)(h) # 13x13x30 -> 10x10x60
h = Pooling((1, 2, 2, 1))(h)            # 10x10x60 ->  5x 5x60
h = Flatten()(h)
y = Dense(5*5*60, 300, tf.nn.relu)(h)
y = Dense(300, 10, tf.nn.softmax)(y)

cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

### 学習 ###

n_epochs = 10
batch_size = 100
n_batches = x_train.shape[0]//batch_size

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        x_train, t_train = shuffle(x_train, t_train, random_state=random_state)
        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: x_train[start:end], t: t_train[start:end]})
        y_pred, cost_valid = sess.run([y, cost], feed_dict={x: x_valid, t: t_valid})
        print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
            epoch,
            cost_valid,
            accuracy_score(t_valid.argmax(axis=1), y_pred.argmax(axis=1))
        ))
    y_pred = sess.run(y, feed_dict={x: x_test}).argmax(axis=1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/chap05/materials/submission_pred.csv', header=True, index_label='id')