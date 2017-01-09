from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
import data_util

import tcl_seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 774, "English vocabulary size.")
tf.app.flags.DEFINE_integer("ner_vocab_size", 15, "NER vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "../data/format_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "../data/checkpoints", "Training directory.")
tf.app.flags.DEFINE_string("vocab_dir", "../data/vocabulary", "Vocabulary directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", True,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(10, 10), (15, 15), (25, 25), (50, 50)]


def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_util.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def read_eval_data(source_path, target_path):
    data_eval = ([], [])
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            while source and target:
                data_eval[0].append([int(x) for x in source.split()])
                data_eval[1].append([int(x) for x in target.split()])
                source, target = source_file.readline(), target_file.readline()
    return data_eval


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = tcl_seq2seq_model.TCLSeq2SeqModel(
        FLAGS.in_vocab_size,
        FLAGS.ner_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    print("Preparing TCL data in %s" % FLAGS.data_dir)
    in_train, ner_train, in_test, ner_test, _, _ = data_util.prepare_tcl_data(FLAGS.data_dir, FLAGS.vocab_dir)

    with tf.Session() as sess:
        # create model
        print("creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # Read data into buckets
        test_set = read_data(in_test, ner_test)
        train_set = read_data(in_train, ner_train)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # Traing step
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_lossses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a tep
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                preplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f preplexity %.2f" %
                      (model.global_step.eval(), model.learning_rate.eval(), step_time, preplexity))

                if len(previous_lossses) > 2 and loss > max(previous_lossses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_lossses.append(loss)

                # Save checkpoint and zero timer and loss
                checkpoint_path = os.path.join(FLAGS.train_dir, "tcl_ner.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                # Run testing and print perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(test_set[bucket_id]) == 0:
                        print(" eval:empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def custom_accuracy_eval(is_training_data=False, target_token=6):
    in_train, ner_train, in_test, ner_test, _, _ = data_util.prepare_tcl_data(FLAGS.data_dir, FLAGS.vocab_dir)
    if is_training_data:
        eval_set = read_eval_data(in_train, ner_train)
    else:
        eval_set = read_eval_data(in_test, ner_test)
    with tf.Session() as sess:
        model = create_model(sess, True)
        model.batch_size = 1

        print(eval_set)
        sentence_counter = 0
        false_sentence_counter = 0
        for token_ids, ground_truth_ids in zip(eval_set[0], eval_set[1]):
            sentence_counter += 1

            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_util.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_util.EOS_ID)]

            # For general accuracy
            # words_error_counter += len([x for x, y in zip(outputs, ground_truth_ids) if x != y])

            # For only location accuracy evaluation:
            pre_index = 0
            if target_token not in ground_truth_ids:
                sentence_counter -= 1
                continue
            if len(outputs) != len(ground_truth_ids):
                false_sentence_counter += 1
            else:
                while pre_index < len(ground_truth_ids):
                    if ground_truth_ids[pre_index] == target_token:
                        if outputs[pre_index] != target_token:
                            false_sentence_counter += 1
                            break
                        else:
                            pass
                    elif outputs[pre_index] == target_token:
                        false_sentence_counter += 1
                        break
                    pre_index += 1

            print("processing %d line" % sentence_counter)

    print(
        "%d sentences with %d accurate tagging, final accuracy %.2f" % (sentence_counter,
                                                                        sentence_counter - false_sentence_counter,
                                                                        1 - false_sentence_counter / sentence_counter))
    return 1 - false_sentence_counter/sentence_counter


def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        in_vocab_path = os.path.join(FLAGS.vocab_dir,
                                     "vocab.in")
        ner_vocab_path = os.path.join(FLAGS.vocab_dir,
                                      "vocab.ner")
        en_vocab, _ = data_util.initialize_vocabulary(in_vocab_path)
        _, rev_fr_vocab = data_util.initialize_vocabulary(ner_vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_util.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            # Which bucket does it belong to?
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_util.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_util.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def main(_):
    if FLAGS.self_test:
        print("=================self_test==================")
        custom_accuracy_eval(is_training_data=True)
    elif FLAGS.decode:
        decode()
    else:
        print("===============train================")
        train()


if __name__ == "__main__":
    tf.app.run()
