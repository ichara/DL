import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import csv

pad_index = 0
en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1
print(len(tokenizer_en.word_index))
print(len(tokenizer_ja.word_index))

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

### レイヤー定義 ###
class Embedding:
    def __init__(self, vocab_size, emb_dim, scale=0.08):
        self.V = tf.Variable(tf.random_normal([vocab_size, emb_dim], stddev=scale), name='V')

    def __call__(self, x):
        return tf.nn.embedding_lookup(self.V, x)

# Encoder-Decoderモデルでは系列対系列の関係を扱うため、新たに状態の出力、系列での出力に対応させる必要があります。
# また、生成を行う際には1ステップずつ逐次的にLSTMを実行したいので、状態を保持する機能も持たせましょう。

class LSTM:
    def __init__(self, hid_dim, seq_len, initial_state, return_state = False, return_sequences = False, hold_state = False, name = None):
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(hid_dim)
        self.initial_state = initial_state
        self.seq_len = seq_len
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.hold_state = hold_state
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name):
            outputs, state = tf.nn.dynamic_rnn(self.cell, x, self.seq_len, self.initial_state)
        
        if self.hold_state:
            self.initial_state = state
        
        if not self.return_sequences:
            outputs = state.h
            
        if not self.return_state:
            return outputs
        
        return outputs, state

class Attention:
    def __init__(self, hid_dim, out_dim, enc_out, seq_len):
        e_hid_dim, d_hid_dim = hid_dim, hid_dim
        
        self.enc_out = enc_out
        self.seq_len = seq_len
        glorot_a = tf.cast(tf.sqrt(6/(e_hid_dim + d_hid_dim)), tf.float32)
        self.W_a  = tf.Variable(tf.random_uniform([e_hid_dim, d_hid_dim], minval=-glorot_a, maxval=glorot_a), name='W_a')
        
        glorot_c = tf.cast(tf.sqrt(6/(e_hid_dim + d_hid_dim + out_dim)), tf.float32)
        self.W_c = tf.Variable(tf.random_uniform([e_hid_dim+d_hid_dim, out_dim], minval=-glorot_c, maxval=glorot_c), name='W_c')
        self.b    = tf.Variable(np.zeros([out_dim]).astype('float32'), name='b')
        
    def __call__(self, dec_out):
        # self.enc_out: [batch_size, enc_length, e_hid_dim]
        # self.W_a  : [e_hid_dim, d_hid_dim]
        # -> enc_out: [batch_size, enc_length, d_hid_dim]
        W_a_broadcasted = tf.tile(tf.expand_dims(self.W_a, axis=0), [tf.shape(self.enc_out)[0],1,1])
        enc_out = tf.matmul(self.enc_out, W_a_broadcasted)
        
        # dec_out: [batch_size, dec_length, d_hid_dim]
        # enc_out: [batch_size, enc_length, d_hid_dim]
        # -> score: [batch_size, dec_length, enc_length]
        score = tf.matmul(dec_out, tf.transpose(enc_out, perm=[0,2,1])) # Attention score
        
        # encoderのステップにそって正規化する
        score = score - tf.reduce_max(score, axis=-1, keep_dims=True) # for numerically stable softmax
        mask = tf.cast(tf.sequence_mask(self.seq_len, tf.shape(score)[-1]), tf.float32) # encoder mask
        exp_score = tf.exp(score) * tf.expand_dims(mask, axis=1)
        self.a = exp_score / tf.reduce_sum(exp_score, axis=-1, keep_dims=True) # softmax

        # self.a  : [batch_size, dec_length, enc_length]
        # self.enc_out: [batch_size, enc_length, e_hid_dim]
        # -> c: [batch_size, dec_length, e_hid_dim]
        c = tf.matmul(self.a, self.enc_out) # Context vector

        W_c_broadcasted = tf.tile(tf.expand_dims(self.W_c, axis=0), [tf.shape(c)[0],1,1])

        return tf.nn.tanh(tf.matmul(tf.concat([c, dec_out], -1), W_c_broadcasted) + self.b)

# EncoderについてはLSTM内でマスク処理を行います。
# またDecoderでは、padding部分についてはコストが0になるようにします。
# paddingの部分の教師ラベルdの要素をすべて0になるようにします。

### グラフ構築 ###
tf.reset_default_graph()

emb_dim = 512
hid_dim = 256
att_dim = 128

x = tf.placeholder(tf.int32, [None, None], name='x')
t = tf.placeholder(tf.int32, [None, None], name='t')

seq_len = tf.reduce_sum(tf.cast(tf.not_equal(x, pad_index), tf.int32), axis=1)
seq_len_t_in = tf.reduce_sum(tf.cast(tf.not_equal(t, pad_index), tf.int32), axis=1) - 1

t_out = tf.one_hot(t[:, 1:], depth=ja_vocab_size, dtype=tf.float32)
t_out = t_out * tf.expand_dims(tf.cast(tf.not_equal(t[:, 1:], pad_index), tf.float32), axis=-1)

initial_state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([tf.shape(x)[0], hid_dim]), tf.zeros([tf.shape(x)[0], hid_dim]))

# Encoder
h_e = Embedding(en_vocab_size, emb_dim)(x)
# _, encoded_state = LSTM(hid_dim, seq_len, initial_state, return_state=True, name='encoder_lstm')(h_e)
encoded_outputs, encoded_state = LSTM(hid_dim, seq_len, initial_state, return_sequences=True, return_state=True, name='encoder_lstm_a')(h_e)

# Decoder
decoder = [
    Embedding(ja_vocab_size, emb_dim),
    LSTM(hid_dim, seq_len_t_in, encoded_state, return_sequences=True, name='decoder_lstm_a'),
    Attention(hid_dim, att_dim, encoded_outputs, seq_len),
    tf.layers.Dense(ja_vocab_size, tf.nn.softmax)
] # 生成時に再利用するためにモデルの各レイヤーを配列で確保

# Decoderに変数を通す
h_d = decoder[0](t)
h_d = decoder[1](h_d)
h_d = decoder[2](h_d)
y = decoder[3](h_d)

cost = -tf.reduce_mean(tf.reduce_sum(t_out * tf_log(y[:, :-1]), axis=[1, 2]))

train = tf.train.AdamOptimizer().minimize(cost)

### データの準備 ###
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train)

x_train_lens = [len(com) for com in x_train]
sorted_train_indexes = sorted(range(len(x_train_lens)), key=lambda x: -x_train_lens[x])

x_train = [x_train[ind] for ind in sorted_train_indexes]
t_train = [t_train[ind] for ind in sorted_train_indexes]

### 学習 ###
n_epochs = 120
n_div = 10
batch_size = 64
train_size = len(x_train) // n_div
valid_size = len(x_valid) // n_div
print(train_size)
print(valid_size)

# run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
run_options = tf.RunOptions()

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):
    # train
    train_costs = []

    # データ縮小(6.81GiB from one_hot)
    e = epoch % n_div
    x_train_sub = x_train[e * train_size : (e+1) * train_size]
    t_train_sub = t_train[e * train_size : (e+1) * train_size]
    x_valid_sub = x_valid[e * valid_size : (e+1) * valid_size]
    t_valid_sub = t_valid[e * valid_size : (e+1) * valid_size]
    n_batches = len(x_train_sub) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        x_train_batch = np.array(pad_sequences(x_train_sub[start:end], padding='post', value=pad_index))
        t_train_batch = np.array(pad_sequences(t_train_sub[start:end], padding='post', value=pad_index))

        _, train_cost = sess.run([train, cost], feed_dict={x: x_train_batch, t: t_train_batch}, options=run_options)
        train_costs.append(train_cost)

    # valid
    x_valid_pad = np.array(pad_sequences(x_valid_sub, padding='post', value=pad_index))
    t_valid_pad = np.array(pad_sequences(t_valid_sub, padding='post', value=pad_index))

    valid_cost = sess.run(cost, feed_dict={x: x_valid_pad, t: t_valid_pad}, options=run_options)
    print('EPOCH: %i, Training Cost: %.3f, Validation Cost: %.3f' % (epoch+1, np.mean(train_costs), valid_cost))
    if valid_cost <= 22.100:
        break

### 生成用グラフ構築 ###
bos_eos = tf.placeholder(tf.int32, [2], name='bos_eos')
max_len = tf.placeholder(tf.int32, name='max_len') # iterationの繰り返し回数の限度

def cond(t, continue_flag, init_state, seq_last, seq, att):
    unfinished = tf.not_equal(tf.reduce_sum(tf.cast(continue_flag, tf.int32)), 0)
    return tf.logical_and(t < max_len, unfinished)

def body(t, prev_continue_flag, init_state, seq_last, seq, att):
    decoder[1].initial_state = init_state
    
    # Decoderの再構築
    h = decoder[0](tf.expand_dims(seq_last, -1))
    h = decoder[1](h)
    h = decoder[2](h)
    y = decoder[3](h)
    
    seq_t = tf.reshape(tf.cast(tf.argmax(y, axis=2), tf.int32), shape=[-1])
    next_state = decoder[1].initial_state
    
    continue_flag = tf.logical_and(prev_continue_flag, tf.not_equal(seq_t, bos_eos[1])) # flagの更新

    # return [t+1, continue_flag, next_state, seq_t, seq.write(t, seq_t)]
    return [t+1, continue_flag, next_state, seq_t, seq.write(t, seq_t), att.write(t-1, tf.squeeze(decoder[2].a))]

decoder[1].hold_state = True
decoder[1].seq_len = None

seq_0 = tf.ones([tf.shape(x)[0]], tf.int32)*bos_eos[0]

t_0 = tf.constant(1)
f_0 = tf.cast(tf.ones_like(seq_0), dtype=tf.bool) # バッチ内の各系列で</s>が出たかどうかの未了flag(0:出た, 1:出てない)
seq_array = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True).write(0, seq_0)
att_array = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

# *_, seq = tf.while_loop(cond, body, loop_vars=[t_0, f_0, encoded_state, seq_0, seq_array])
*_, seq, att = tf.while_loop(cond, body, loop_vars=[t_0, f_0, encoded_state, seq_0, seq_array, att_array])

# res = tf.transpose(seq.stack())
res = (tf.transpose(seq.stack()), tf.transpose(att.stack(), perm=[1, 0, 2]))

### 生成 ###
bos_id_ja, eos_id_ja = tokenizer_ja.texts_to_sequences(['<s> </s>'])[0]
# y_pred = sess.run(res, feed_dict={
y_pred, att_weights = sess.run(res, feed_dict={
    x: pad_sequences(x_test, padding='post', value=pad_index),
    bos_eos: np.array([bos_id_ja, eos_id_ja]),
    max_len: 100
}, options=run_options)

### 出力 ###
def get_raw_contents(dataset, num, bos_id, eos_id):
    result = []
    for index in dataset[num]:
        if index == eos_id:
            break
            
        result.append(index)
        
        if index == bos_id:
            result = []
            
    return result

output = [get_raw_contents(y_pred, i, bos_id_ja, eos_id_ja) for i in range(len(y_pred))]

with open('./materials/submission_gen.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(output)