# Paraphrase generation based on LSTM-VAE by wanzeyu
# Overview
Keras implementation for 
`A Deep Generative Framework for Paraphrase Generation`.

Resource Used:

1. MSRP paraphrase corpus

2. Fasttext's pretrained vector

Requirements:
1. Keras
2. Numpy

In this project I try to implement novel VAE-LSTM architecture mentioned in `A Deep Generative Framework for 
Paraphrase Generation`.

# QuickStart
Use `python example.py`. The code will train on the corpus and 
print predicted result after every epoch.

By default ,the program will be in training mode and will save model 
to three files. 

The default epoch is 200.

If you want to change
the parameters. You have to change according lines.

# Resource
* `test_source.txt` is the original file
* `test_target.txt` is the paraphrase file
* `wiki.simple.vec` is fasttext's pretrained vector on simple 
wiki

# Model Architecture
encoder model:

![encoder](https://github.com/paulx3/keras_generative_pg/raw/master/encoder.png)

decoder model:

![decoder](https://github.com/paulx3/keras_generative_pg/raw/master/generator.png)

vae overview:

![overview](https://github.com/paulx3/keras_generative_pg/raw/master/vae.png)

# Problems
1. I didn't write a proper test function or use BLEU to 
evaluate. This has to be done after I found out what's wrong 
with my implementation.


# Progress
- [x] Implement the basic framework of the thesis
- [ ] Write the evaluation code
- [x] Refactor and clean up the messy code




# References
- [Keras implementation of LSTM Variational Autoencoder](https://github.com/twairball/keras_lstm_vae)
- [Toni-Antonova/VAE-Text-Generation](https://github.com/Toni-Antonova/VAE-Text-Generation)
- [A Deep Generative Framework for Paraphrase Generation](https://arxiv.org/pdf/1709.05074.pdf)
- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
