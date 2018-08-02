import os
import sys
import time
import ipdb
import random
import cPickle as pickle
import numpy as np
import tensorflow as tf

from vocab import Vocabulary, build_vocab
from accumulator import Accumulator
from options import load_arguments
from file_io import load_sent, write_sent
from style_tranfer_model import Model
from utils import *
from nn import *
import beam_search, greedy_decoding        


def transfer(model, decoder, sess, args, vocab, data0, data1, out_path):
    batches, order0, order1 = get_batches(data0, data1,
        vocab.word2id, args.batch_size)

    #data0_rec, data1_rec = [], []
    data0_tsf, data1_tsf = [], []
    losses = Accumulator(len(batches), ['loss', 'rec', 'adv', 'd0', 'd1'])
    for batch in batches:
        rec, tsf = decoder.rewrite(batch)
        half = batch['size'] / 2
        #data0_rec += rec[:half]
        #data1_rec += rec[half:]
        data0_tsf += tsf[:half]
        data1_tsf += tsf[half:]

        loss, loss_rec, loss_adv, loss_d0, loss_d1 = sess.run([model.loss,
            model.loss_rec, model.loss_adv, model.loss_d0, model.loss_d1],
            feed_dict=feed_dictionary(model, batch, args.rho, args.gamma_min))
        losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1])

    n0, n1 = len(data0), len(data1)
    #data0_rec = reorder(order0, data0_rec)[:n0]
    #data1_rec = reorder(order1, data1_rec)[:n1]
    data0_tsf = reorder(order0, data0_tsf)[:n0]
    data1_tsf = reorder(order1, data1_tsf)[:n1]

    if out_path:
        #write_sent(data0_rec, out_path+'.0'+'.rec')
        #write_sent(data1_rec, out_path+'.1'+'.rec')
        write_sent(data0_tsf, out_path+'.0'+'.tsf')
        write_sent(data1_tsf, out_path+'.1'+'.tsf')

    return losses


def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        #print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        #print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    return model


if __name__ == '__main__':
    args = load_arguments()

    #####   data preparation   #####
    if args.train:
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        print '#sents of training file 0:', len(train0)
        print '#sents of training file 1:', len(train1)

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print 'vocabulary size:', vocab.size

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)

        if args.beam > 1:
            decoder = beam_search.Decoder(sess, args, vocab, model)
        else:
            decoder = greedy_decoding.Decoder(sess, args, vocab, model)

        if args.train:
            batches, _, _ = get_batches(train0, train1, vocab.word2id,
                args.batch_size, noisy=True)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            losses = Accumulator(args.steps_per_checkpoint,
                ['loss', 'rec', 'adv', 'd0', 'd1'])
            best_dev = float('inf')
            learning_rate = args.learning_rate
            rho = args.rho
            gamma = args.gamma_init
            dropout = args.dropout_keep_prob

            #gradients = Accumulator(args.steps_per_checkpoint,
            #    ['|grad_rec|', '|grad_adv|', '|grad|'])

            for epoch in range(1, 1+args.max_epochs):
                print '--------------------epoch %d--------------------' % epoch
                print 'learning_rate:', learning_rate, '  gamma:', gamma

                for batch in batches:
                    feed_dict = feed_dictionary(model, batch, rho, gamma,
                        dropout, learning_rate)

                    loss_d0, _ = sess.run([model.loss_d0, model.optimize_d0],
                        feed_dict=feed_dict)
                    loss_d1, _ = sess.run([model.loss_d1, model.optimize_d1],
                        feed_dict=feed_dict)

                    # do not back-propagate from the discriminator
                    # when it is too poor
                    if loss_d0 < 1.2 and loss_d1 < 1.2:
                        optimize = model.optimize_tot
                    else:
                        optimize = model.optimize_rec

                    loss, loss_rec, loss_adv, _ = sess.run([model.loss,
                        model.loss_rec, model.loss_adv, optimize],
                        feed_dict=feed_dict)
                    losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1])

                    #grad_rec, grad_adv, grad = sess.run([model.grad_rec_norm,
                    #    model.grad_adv_norm, model.grad_norm],
                    #    feed_dict=feed_dict)
                    #gradients.add([grad_rec, grad_adv, grad])

                    step += 1
                    if step % args.steps_per_checkpoint == 0:
                        losses.output('step %d, time %.0fs,'
                            % (step, time.time() - start_time))
                        losses.clear()

                        #gradients.output()
                        #gradients.clear()

                if args.dev:
                    dev_losses = transfer(model, decoder, sess, args, vocab,
                        dev0, dev1, args.output + '.epoch%d' % epoch)
                    dev_losses.output('dev')
                    if dev_losses.values[0] < best_dev:
                        best_dev = dev_losses.values[0]
                        print 'saving model...'
                        model.saver.save(sess, args.model)

                gamma = max(args.gamma_min, gamma * args.gamma_decay)

        if args.test:
            test_losses = transfer(model, decoder, sess, args, vocab,
                test0, test1, args.output)
            test_losses.output('test')

        if args.online_testing:
            while True:
                sys.stdout.write('> ')
                sys.stdout.flush()
                inp = sys.stdin.readline().rstrip()
                if inp == 'quit' or inp == 'exit':
                    break
                inp = inp.split()
                y = int(inp[0])
                sent = inp[1:]

                batch = get_batch([sent], [y], vocab.word2id)
                ori, tsf = decoder.rewrite(batch)
                print 'original:', ' '.join(w for w in ori[0])
                print 'transfer:', ' '.join(w for w in tsf[0])
