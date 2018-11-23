import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

### レイヤー定義 ###
class Embedding:
    def __init__(self, vocab_size, emb_dim, scale=0.08):
        self.V = tf.Variable(tf.random_normal([vocab_size, emb_dim], stddev=scale), name='V')

    def __call__(self, x):
        return tf.nn.embedding_lookup(self.V, x)

# class RNN:
#     def __init__(self, in_dim, hid_dim, seq_len=None, scale=0.08):
#         self.in_dim = in_dim
#         self.hid_dim = hid_dim
        
#         glorot = tf.cast(tf.sqrt(6/(in_dim + hid_dim*2)), tf.float32)
#         self.W = tf.Variable(tf.random_uniform([in_dim+hid_dim, hid_dim], minval=-glorot, maxval=glorot), name='W')
#         self.b = tf.Variable(tf.zeros([hid_dim]), name='b')
        
#         self.seq_len = seq_len
#         self.initial_state = None

#     def __call__(self, x):
#         def fn(h_prev, x_and_m):
#             x_t, m_t = x_and_m
#             inputs = tf.concat([x_t, h_prev], -1)
#             # RNN
#             h_t = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
#             # マスクの適用
#             h_t = m_t * h_t + (1 - m_t) * h_prev
            
#             return h_t

#         # 入力の時間順化
#         # shape: [batch_size, max_seqence_length, in_dim] -> [max_seqence_length, batch_size, in_dim]
#         x_tmaj = tf.transpose(x, perm=[1, 0, 2])
        
#         # マスクの生成＆時間順化
#         mask = tf.cast(tf.sequence_mask(self.seq_len, tf.shape(x)[1]), tf.float32)
#         mask_tmaj = tf.transpose(tf.expand_dims(mask, axis=-1), perm=[1, 0, 2])
        
#         if self.initial_state is None:
#             batch_size = tf.shape(x)[0]
#             self.initial_state = tf.zeros([batch_size, self.hid_dim])
        
#         h = tf.scan(fn=fn, elems=[x_tmaj, mask_tmaj], initializer=self.initial_state)
        
#         return h[-1]

class RNN:
    def __init__(self, hid_dim, seq_len = None, initial_state = None):
        self.cell = tf.nn.rnn_cell.BasicRNNCell(hid_dim)
        self.initial_state = initial_state
        self.seq_len = seq_len
    
    def __call__(self, x):
        if self.initial_state is None:
            self.initial_state = self.cell.zero_state(tf.shape(x)[0], tf.float32)
            
        # outputsは各系列長分以降は0になるので注意
        outputs, state = tf.nn.dynamic_rnn(self.cell, x, self.seq_len, self.initial_state)
        return tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(x)[0]), self.seq_len-1], axis=1))

### グラフ構築 ###
tf.reset_default_graph()

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

emb_dim = 20
hid_dim = 10
num_words = max([max(s) for s in np.hstack((x_train, x_test))])
pad_index = 0

x = tf.placeholder(tf.int32, [None, None], name='x')
t = tf.placeholder(tf.float32, [None, None], name='t')

seq_len = tf.reduce_sum(tf.cast(tf.not_equal(x, pad_index), tf.int32), axis=1)

h = Embedding(num_words, emb_dim)(x)
# h = RNN(emb_dim, hid_dim, seq_len)(h)
h = RNN(hid_dim, seq_len)(h)
y = tf.layers.Dense(1, tf.nn.sigmoid)(h)

cost = -tf.reduce_mean(t*tf_log(y) + (1 - t)*tf_log(1 - y))

# train = tf.train.AdamOptimizer().minimize(cost)
optimizer = tf.train.AdamOptimizer()
grads = optimizer.compute_gradients(cost)
clipped_grads = [(tf.clip_by_value(grad_val, -1., 1.), var) for grad_val, var in grads]
train = optimizer.apply_gradients(clipped_grads)

test = tf.round(y)

### データの準備 ###
x_all = x_train
t_all = t_train
x_train, x_valid, t_train, t_valid = train_test_split(x_all, t_all, test_size=0.25, random_state=42)

### 学習 ###
n_epochs = 20
batch_size = 100 # バッチサイズが大きいと、ResourceExhaustedErrorになることがあります

n_batches_train = len(x_train) // batch_size
n_batches_valid = len(x_valid) // batch_size
n_batches_test = len(x_test) // batch_size

y_pred_test = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        # Train
        train_costs = []
        for i in range(n_batches_train):
            start = i * batch_size
            end = start + batch_size
            
            x_train_batch = np.array(pad_sequences(x_train[start:end], padding='post', value=pad_index)) # バッチ毎のPadding
            t_train_batch = np.array(t_train[start:end])[:, None]

            _, train_cost = sess.run([train, cost], feed_dict={x: x_train_batch, t: t_train_batch})
            train_costs.append(train_cost)
            
            # print('Train: EPOCH: %i, Training Cost: %.3f' % (epoch+1, np.mean(train_costs)))
        
        # Valid
        valid_costs = []
        y_pred = []
        for i in range(n_batches_valid):
            start = i * batch_size
            end = start + batch_size
            
            x_valid_pad = np.array(pad_sequences(x_valid[start:end], padding='post', value=pad_index)) # バッチ毎のPadding
            t_valid_pad = np.array(t_valid[start:end])[:, None]
            
            pred, valid_cost = sess.run([test, cost], feed_dict={x: x_valid_pad, t: t_valid_pad})
            y_pred += pred.flatten().tolist()
            valid_costs.append(valid_cost)

            # print('Valid: EPOCH: %i, Validation: %.3f' % (epoch+1, np.mean(valid_costs)))

        print('EPOCH: %i, Training Cost: %.3f, Validation Cost: %.3f, Validation F1: %.3f' % (epoch+1, np.mean(train_costs), np.mean(valid_costs), f1_score(t_valid, y_pred, average='macro')))

        # Test
        x_test_pad = np.array(pad_sequences(x_test, padding='post', value=pad_index))
        y_pred_test = sess.run(test, feed_dict={x: x_test_pad})

        ### 出力 ###
        print(len(y_pred_test.T[0]))
        submission = pd.Series(y_pred_test.T[0], name='label')
        submission.to_csv('./submission_pred.csv', header=True, index_label='id')

        if f1_score(t_valid, y_pred, average='macro') >= 0.920:
            break;

        x_train, x_valid, t_train, t_valid = train_test_split(x_all, t_all, test_size=0.25, random_state=42+epoch+1)
        
tf.Session().close()