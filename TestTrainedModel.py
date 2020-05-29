import tensorlayer as tl
from tensorlayer.layers import *

import tensorflow as tf
import time

###============= prepare data
from data.twitter import data
metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')                   # Twitter
# from data.cornell_corpus import data
# metadata, idx_q, idx_a = data.load_data(PATH='data/cornell_corpus/')          # Cornell Moive
(trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)


testX = testX.tolist()
testY = testY.tolist()


testX = tl.prepro.remove_pad_sequences(testX)
testY = tl.prepro.remove_pad_sequences(testY)


###============= parameters
xseq_len = len(trainX)#.shape[-1]
yseq_len = len(trainY)#.shape[-1]
assert xseq_len == yseq_len
batch_size = 32
n_step = int(xseq_len/batch_size)
xvocab_size = len(metadata['idx2w']) # 8002 (0~8001)
emb_dim = 1024

w2idx = metadata['w2idx']   # dict  word 2 index
idx2w = metadata['idx2w']   # list index 2 word

unk_id = w2idx['unk']   # 1
pad_id = w2idx['_']     # 0

start_id = xvocab_size  # 8002
end_id = xvocab_size+1  # 8003

w2idx.update({'start_id': start_id})
w2idx.update({'end_id': end_id})
idx2w = idx2w + ['start_id', 'end_id']

xvocab_size = yvocab_size = xvocab_size + 2

###============= model
def model(encode_seqs, decode_seqs, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        # for chatbot, you can use the same embedding layer,
        # for translation, you may want to use 2 seperated embedding layers
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True) # remove if TL version == 1.8.0+
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
        net_rnn = Seq2Seq(net_encode, net_decode,
                cell_fn = tf.contrib.rnn.BasicLSTMCell,
                n_hidden = emb_dim,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                initial_state_encode = None,
                dropout = (0.5 if is_train else None),
                n_layer = 3,
                return_seq_2d = True,
                name = 'seq2seq')
        net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')
    return net_out, net_rnn

encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
net, net_rnn = model(encode_seqs2, decode_seqs2, is_train=False, reuse=True)
y = tf.nn.softmax(net.outputs)


# sess = tf.InteractiveSession()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
temp_net = tl.files.load_and_assign_npz(sess=sess, name='n.npz', network=net)
print(temp_net.shape)
seeds = ["happy birthday have a nice day",
         "donald trump won last nights presidential debate according to snap online polls"]
for seed in seeds:
    print("Query >", seed)
    seed_id = [w2idx[w] for w in seed.split(" ")]
    for _ in range(5):  # 1 Query --> 5 Reply
        # 1. encode, get state
        state = sess.run(net_rnn.final_state_encode,
                         {encode_seqs2: [seed_id]})
        # 2. decode, feed start_id, get first word
        #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
        o, state = sess.run([y, net_rnn.final_state_decode],
                            {net_rnn.initial_state_decode: state,
                             decode_seqs2: [[start_id]]})
        w_id = tl.nlp.sample_top(o[0], top_k=3)
        w = idx2w[w_id]
        # 3. decode, feed state iteratively
        sentence = [w]
        for _ in range(30):  # max sentence length
            o, state = sess.run([y, net_rnn.final_state_decode],
                                {net_rnn.initial_state_decode: state,
                                 decode_seqs2: [[w_id]]})
            w_id = tl.nlp.sample_top(o[0], top_k=2)
            w = idx2w[w_id]
            if w_id == end_id:
                break
            sentence = sentence + [w]
        print(" >", ' '.join(sentence))