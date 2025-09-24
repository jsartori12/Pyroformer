# Pyroformer 🔥

![PyroGPT](pyrogpt.png)

An experimental GPT-style language model built from scratch in PyTorch to generate novel, heat-stable thermophilic protein sequences.

## ✨ Features

- Decoder-only Transformer architecture.
- Custom Byte-Pair Encoding (BPE) tokenizer trained on a curated dataset of thermophilic proteins.
- Simple, educational codebase for training and generation.

## 🚀 Usage

1.  **Prepare the dataset** and place it in the root folder.

2.  **Train the BPE tokenizer:**
    ```bash
    python bpe_vocab.py
    ```

3.  **Train the generative model:**
    ```bash
    python main.py
    ```
    The script will print generated sequences upon completion.

## Acknowledgements

This project is heavily inspired by Andrej Karpathy's educational work on [nanoGPT](https://github.com/karpathy/nanoGPT).
