import tensorflow as tf
from dis_model import DIS
from gen_model import GEN
import pickle
import numpy as np
import utils as ut
import multiprocessing
import time

cores = multiprocessing.cpu_count()
ratio = '0.001'

#########################################################################################
# Hyper-parameters
#########################################################################################
EMB_DIM = 200
BATCH_SIZE = 16
INIT_DELTA = 0.05

workdir = 'douban/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'

#########################################################################################
# Load data
#########################################################################################
user_pos_train = {}
userls = set()
itemls = set()
with open('train' + ratio + '.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(eval(line[0]))
        iid = int(eval(line[1]))
        userls.add(uid)
        itemls.add(iid)
        r = float(eval(line[2]))
        if r > 0.5:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {}
test_dic = {}
with open('test' + ratio + '.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(eval(line[0]))
        iid = int(eval(line[1]))
        userls.add(uid)
        itemls.add(iid)
        r = float(eval(line[2]))

        if uid in test_dic:
            test_dic[uid].append(iid)
        else:
            test_dic[uid] = [iid]

        if r > 0.5:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

all_users = user_pos_train.keys()
# all_users.sort()
userls = list(userls)
itemls = list(itemls)
USER_NUM = len(userls)
ITEM_NUM = len(itemls)
all_items = set(range(ITEM_NUM))


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def ap_at_k(r, k):
    ts = 0
    for s in range(k):
        if r[s] == 1:
            ts += np.mean(r[:(s + 1)])
    ts = ts / k
    return ts


def simple_test_one_user(x):
    rating = x[0]
    u = x[1]

    # test_items = list(all_items - set(user_pos_train[u]))
    test_items = test_dic[u]
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_8 = np.mean(r[:8])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_8 = ndcg_at_k(r, 8)
    ap_3 = ap_at_k(r, 3)
    ap_5 = ap_at_k(r, 5)
    ap_8 = ap_at_k(r, 8)

    return np.array([p_3, p_5, p_8, ndcg_3, ndcg_5, ndcg_8, ap_3, ap_5, ap_8])


def simple_test(sess, model):
    result = np.array([0.] * 9)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = user_pos_test.keys()
    test_users = list(test_users)
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def generate_for_d(sess, model, filename):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]

        rating = sess.run(model.all_rating, {model.u: [u]})
        rating = np.array(rating[0]) / 0.2  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        neg = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=prob)
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


print("load model...")

print(USER_NUM, ITEM_NUM)
# param: param = [user_matrix, item_matrix, bias]    matrix is constructed from graph
param = pickle.load(open('embeddings_' + ratio + '.pkl', 'rb'))
generator = GEN(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.0 / BATCH_SIZE, param=param, initdelta=INIT_DELTA,
                learning_rate=0.002)
discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1 / BATCH_SIZE, param=None, initdelta=INIT_DELTA,
                    learning_rate=0.001)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

print("gen ", simple_test(sess, generator))
print("dis ", simple_test(sess, discriminator))

dis_log = open(workdir + 'dis_log.txt', 'w')
gen_log = open(workdir + 'gen_log.txt', 'w')

t0 = time.time()

# minimax training
best = 0.
for epoch in range(50):
    if epoch >= 0:
        for d_epoch in range(50):
            if d_epoch % 5 == 0:
                generate_for_d(sess, generator, DIS_TRAIN_FILE)
                train_size = ut.file_len(DIS_TRAIN_FILE)
            index = 1
            while True:
                if index > train_size:
                    break
                if index + BATCH_SIZE <= train_size + 1:
                    input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                else:
                    input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                            train_size - index + 1)
                index += BATCH_SIZE

                _ = sess.run(discriminator.d_updates,
                             feed_dict={discriminator.u: input_user, discriminator.i: input_item,
                                        discriminator.label: input_label})

        # Train G
        for g_epoch in range(25):  # 50
            for u in user_pos_train:
                sample_lambda = 0.2
                pos = user_pos_train[u]
                pos = pos

                rating = sess.run(generator.all_logits, {generator.u: [u]})
                exp_rating = np.exp(rating)
                prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

                pn = (1 - sample_lambda) * prob
                pn[pos] += sample_lambda * 1.0 / len(pos)
                # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

                sample = np.random.choice(np.arange(ITEM_NUM), 2 * len(pos), p=pn)
                ###########################################################################
                # Get reward and adapt it with importance sampling
                ###########################################################################
                reward = sess.run(discriminator.reward, {discriminator.u: u, discriminator.i: sample})
                reward = reward * prob[sample] / pn[sample]
                ###########################################################################
                # Update G
                ###########################################################################
                _ = sess.run(generator.gan_updates,
                             {generator.u: u, generator.i: sample, generator.reward: reward})

            if g_epoch % 5 == 0:
                result = simple_test(sess, generator)
                print("epoch ", epoch, "gen: ", result)
                buf = '\t'.join([str(x) for x in result])
                gen_log.write(str(epoch) + '\t' + buf + '\n')
                gen_log.flush()

                p_5 = result[2]
                if p_5 > best:
                    best_result = result
                    iterations = epoch * 25 + g_epoch
                    print('best: ', result)
                    best = p_5
                    # generator.save_model(sess, "ml-100k/gan_generator_1.pkl")
                    print('Iteration:', epoch * 25 + g_epoch)
                    t1 = time.time()
gen_log.close()
dis_log.close()

print('best result:', best_result)
print('Iteration:', iterations)
print('Seconds:', t1 - t0)
