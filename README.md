# CS 224N Default Final Project - Multitask BERT

This repository contains the code for the default final project for the Stanford CS 224N and some further optimizing extensions. The project focuses on implementing core components of the BERT model, understanding its architecture, and leveraging its embeddings for three key downstream NLP tasks:

1. Sentiment Classification
2. Paraphrase Detection
3. Semantic Similarity
4. 
After completing the BERT implementation, you will have a multitask model capable of performing these tasks simultaneously. The project also includes opportunities to implement extensions and optimizations to improve the baseline model's performance.

## Project Overview

The project is structured to help you:
* Gain a deep understanding of the BERT architecture by implementing key components.
* Explore multitask learning by training a single model to perform multiple tasks.
* Experiment with extensions to enhance the baseline model’s performance.

## 📈 Extensions
After implementing the baseline model, explore these extensions to improve its performance:

* Optimizer and Learning Rate Scheduling: Experiment with optimizers like Adam, AdamW, or SGD, and use techniques like warmup and decay.
* Regularization Techniques: Add dropout, weight decay, or layer normalization to prevent overfitting.
* Task-Specific Modifications: Create custom task heads for better task-specific performance.

### Acknowledgement
This project is for educational purposes and is restricted to use within the Stanford CS 224N course. For external use, please contact the course administrators.

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
