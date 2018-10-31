
import math
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

tf.reset_default_graph() # グラフのリセット

x_train, y_train, x_test = load_fashionmnist()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1000)

x = tf.placeholder(tf.float32, (None, 784)) # 入力データ
t = tf.placeholder(tf.float32, (None, 10)) # 教師データ

class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        self.W = tf.Variable(tf.random_uniform(shape=(in_dim, out_dim), minval=-0.08, maxval=0.08), name='W')
        self.b = tf.Variable(tf.zeros(out_dim), name='b')
        self.function = function
        
        self.params = [self.W, self.b]
    
    def __call__(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)

def sgd(cost, params, eps=0.1):
    grads = tf.gradients(cost, params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append(param.assign_sub(eps * grad))
    return updates

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

layers = [
    Dense(784, 200, tf.nn.relu),
    Dense(200, 200, tf.nn.relu),
    Dense(200, 10, tf.nn.softmax)
]

# params = []
# h = x
# for layer in layers:
#     h = layer(h)
#     params += layer.params
# y = h

def get_params(layers):
    params_all = []
    for layer in layers:
        params = layer.params
        params_all.extend(params)
    return params_all

def f_props(layers, h):
    for layer in layers:
        h = layer(h)
    return h

y = f_props(layers, x)
params_all = get_params(layers)

cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))

updates = sgd(cost, params_all)
train = tf.group(*updates)

n_epochs = 20
batch_size = 100
n_batches = math.ceil(len(x_train) / batch_size)

sess = tf.Session()
# show_graph(sess.graph)

sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):
    x_train, y_train = shuffle(x_train, y_train)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        sess.run(train, feed_dict={x: x_train[start:end], t: y_train[start:end]}) # set placeholders
    y_pred, cost_valid_ = sess.run([y, cost], feed_dict={x: x_valid, t: y_valid})
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
        epoch + 1,
        cost_valid_,
        accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
    ))

y_pred = sess.run(y, feed_dict={x: x_test}).argmax(axis=1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/chap04/materials/submission_pred.csv', header=True, index_label='id')