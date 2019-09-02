#!/usr/bin/env python3

import fire
import json
import os
#import numpy as np // no longer needed because seed'ing will be handled by the input text file
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    models_dir='models',
):
    
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        #np.random.seed(seed) // use pi for now
        tf.set_random_seed(31415926)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Pull this from a text file later, manual for now >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 80)
                    print(text)
                    #send output to file for 
                    f = open("jangledOutput.txt", "a+")                    
                    f.write("\n\n\n")
                    f.write("========================================\n")
                    f.write("==========USER INPUT SEPERATOR==========\n")
                    f.write("========================================\n")
                    f.write("\n\n\n")
                    f.write(text)
                    f.close()
            print("=" * 80)
            break

if __name__ == '__main__':
    fire.Fire(interact_model)

