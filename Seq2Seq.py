"""References
----------
http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/
https://github.com/tensorlayer/seq2seq-chatbot

"""
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
from sklearn.utils import shuffle
from data.twitter import data

cluster_size = 32
hidden_units = 1024
learning_rate = 0.0001
epochs = 50
QTrain, ATrain, QTest, ATest = [], [], [], []
total_error = 0


def removeZeros():
    for i in QuesTrain:
        QTrain.append(np.trim_zeros(i))
    for i in QuesTrain:
        ATrain.append(np.trim_zeros(i))
    for i in QuesTrain:
        QTest.append(np.trim_zeros(i))
    for i in QuesTrain:
        ATest.append(np.trim_zeros(i))
    return True


# Seq2Seq syntax taken from recurrent.py documentation file of RNN
def create_model(encode_q, decode_a, is_train, reuse):
    with tf.variable_scope("create_model", reuse=reuse):
        with tf.variable_scope("inserting") as temp:
            ques_layer = EmbeddingInputlayer(encode_q, size_of_dict, hidden_units, name='seq_embedding')
            temp.reuse_variables()
            tl.layers.set_name_reuse(True)
            ans_layer = EmbeddingInputlayer(decode_a, size_of_dict, hidden_units, name='seq_embedding')
        seq2seq_rnn = Seq2Seq(ques_layer, ans_layer, tf.contrib.rnn.BasicLSTMCell,
                              n_hidden=hidden_units,
                              initializer=tf.random_uniform_initializer(-0.1, 0.1),
                              encode_sequence_length=retrieve_seq_length_op2(encode_q),
                              decode_sequence_length=retrieve_seq_length_op2(decode_a),
                              dropout=(0.3 if is_train else None),
                              n_layer=3,
                              return_seq_2d=True)
        dense_rnn = DenseLayer(seq2seq_rnn, size_of_dict)
    return dense_rnn, seq2seq_rnn


class TrainRNN:
    def __init__(self):
        self.out = ""
        self.loss = ''
        self.optimzed_runn = ''
        self.total_error = 0
        self.configure_model()

    def configure_model(self):
        self.out = tf.nn.softmax(test_rnn.outputs)
        loss = tl.cost.cross_entropy_seq_with_mask(train_rnn.outputs, label_a, label_mask)
        self.optimized_rnn = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def run_epochs(self):
        for epoch in range(0, epochs):
            print("Starting epoch: ", epoch)
            for ques, ans in tl.iterate.minibatches(QTrain, ATrain, cluster_size):
                ques = tl.prepro.pad_sequences(ques)
                label_ans = tl.prepro.sequences_add_end_id(ans, word2id['<EOS>'])
                label_ans = tl.prepro.pad_sequences(label_ans)

                decode_ans = tl.prepro.sequences_add_start_id(ans, word2id['<GO>'])
                decode_ans = tl.prepro.pad_sequences(decode_ans)
                mask = tl.prepro.sequences_get_mask(label_ans)

                _, error = session.run([self.optimized_rnn, self.loss], {encode_q: ques, decode_a: decode_ans,
                                                                         label_a: label_ans, label_mask: mask})
                self.total_error += error
            tl.files.save_npz(test_rnn.all_params, 'ChatbotRNN.npz', session);


if __name__ == "__main__":
    frequentWords, questions, answers = data.load_data(PATH='data/twitter/')
    word2id = frequentWords['w2idx']
    id2word = frequentWords['idx2w']

    word2id.update({'<GO>': len(id2word)})
    word2id.update({'<EOS>': len(id2word) + 1})
    size_of_dict = len(id2word) + 2

    QuesTrain, AnsTrain, QuesTest, AnsTest = data.split_dataset(questions, answers)
    removeTrail = removeZeros()
    assert removeTrail == True
    QTrain, ATrain = shuffle(QTrain, ATrain, random_state=0)

    encode_q = tf.placeholder(tf.int64, [cluster_size, None])
    decode_a = tf.placeholder(tf.int64, [cluster_size, None])
    label_a = tf.placeholder(tf.int64, [cluster_size, None])
    label_mask = tf.placeholder(tf.int64, [cluster_size, None])
    encode_test_q = tf.placeholder(tf.int64, [1, None])
    decode_test_a = tf.placeholder(tf.int64, [1, None])
    train_rnn, _ = create_model(encode_q, decode_a, True, False)
    test_rnn, seq2seq_rnn = create_model(encode_test_q, decode_test_a, False, True)

    # Configuring the prototype to have soft placements without logging them
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(session)
    tl.files.load_and_assign_npz(session, 'Chatbot_RNN.npz', test_rnn)
