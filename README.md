# Paraphrase generation based on LSTM-VAE
# Overview
Keras implementation for 
`A Deep Generative Framework for Paraphrase Generation`.

Warning:

This project is still in its primitive stage. 

I use part of MSRP data as the training data.


In this project I try to implement novel VAE-LSTM architecture mentioned in `A Deep Generative Framework for 
Paraphrase Generation`.

There are still many problems within. 
If you have a better idea, please open an issue.

# QuickStart
Use `python example.py`.By default ,the program will be in training mode and 
will save model to three files. The default epoch is 20.If you want to change
the parameters. You have to change according lines.

# Problems
1. The result is really bad, I have to find what I did wrong in 
implementing the model
2. I didn't write a proper test function or use BLEU to 
evaluate. This has to be done after I found out what's wrong 
with my implementation.

# Progress
- [x] Implement the basic framework of the thesis
- [ ] Write the test code
- [ ] Refactor and clean up the messy code




##### References
    - [Toni-Antonova/VAE-Text-Generation](https://github.com/Toni-Antonova/VAE-Text-Generation)
    - [A Deep Generative Framework for Paraphrase Generation](https://arxiv.org/pdf/1709.05074.pdf)
    - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
    - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
