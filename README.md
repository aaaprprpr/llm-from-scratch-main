# llm-from-scratch

A high-performance, transformer-based language model implemented from scratch. This project focuses on the granular implementation of modern LLM architectures, moving away from high-level `PyTorch` abstractions to explicitly demostrate the underlying mechanics of large language models. Also, it is trainable.

## Overview

This repository contains a complete implementation of a Transformer-based LLM. The core philosophy of this project is **zero reliance on high-level `torch.nn` layers** (like `nn.Linear`, `nn.LayerNorm`, or `nn.Transformer`).

### Key Technical Features

* **Model Architecture:**
    * **Pre-normalization:** Improved training stability by normalizing before blocks.
    * **RMSNorm:** Root Mean Square Layer Normalization for faster computation.
    * **SwiGLU Activation:** Implementation of the Gated Linear Unit variant used in Llama architectures.
    * **Rotary Positional Embeddings (RoPE):** Features an auto-expanding implementation for flexible sequence lengths.
* **Training & Optimization:**
    * **Custom AdamW:** Built from the `torch.optim.Optimizer` base class.
    * **Learning Rate Schedule:** Cosine annealing for optimal convergence.
    * **Gradient Clipping:** To prevent exploding gradients during intense training phases.

## 🏗 Architecture

The model strictly follows the architectural flow illustrated below:

![Architecture](img/architecture.png)


## 📂 Project Structure

```text
├── main/
│   ├── model.py                # Core Transformer component implementations
│   ├── tokenizer_optimized.py   # Custom BPE/Tokenization
│   ├── train_model.py          # Training utils and optimizer
│   ├── run_train_model.py      # Entry point for training
│   └── play_model.ipynb        # Play with the trained model
├── tokenized_data/             # Pre-processed datasets for training
├── trained_tokenizer/          # Saved tokenizer states
├── img/                        # Architectural diagrams
├── run.sh                      # Shell script for one-touch execution
└── generate_tree.py            # Utility for project visualization
```

## 🛠 Usage (to train it urself)

This implementation is optimized for efficiency and is fully trainable on consumer hardware, including MacBook Pro (M-series) chips.

### Quick Start

To begin training on whatever dataset you like:

1. Clone the repository.
2. Ensure you have PyTorch installed.
3. Implement the tokenizer to fill in `trained_tokenizer` and `tokenized_data`.
4. Execute the training script (you may need to adjust some params a little before running):
    ```sh
    ./run.sh
    ```

## 🧪 Implementation Constraints

To demonstrate technical rigor, this project intentionally avoids `torch.nn` high-level definitions. The only components used from `torch` are:

- `torch.nn.Parameter`: For weight initialization.
- Container classes: (`Module`, `ModuleList`, `Sequential`) for organizing the model graph.
- `torch.optim.Optimizer`: Used only as a base class for a ground-up AdamW implementation.

## Lessons Learned

Building a Transformer without the safety nets of `torch.nn` high-level modules revealed several non-obvious engineering challenges:

#### 1. Numerical Stability in Custom Layers
Implementing RMSNorm and Softmax from scratch highlighted the importance of numerical stability. Without `torch.nn.LayerNorm`, the key thing is to ensure that the epsilon placement was precise to avoid division-by-zero or overflow during the reciprocal square root calculation.

#### 2. The Nuance of RoPE (Rotary Positional Embeddings)
Implementing RoPE required a deep dive into complex number rotations. A more challenging part is to create an auto-expanding cache for the rotation frequencies so the model can handle sequence lengths beyond the initial training window without re-calculating the rotation matrix from scratch every time.

#### 3. Manual Weight Management
In `nn.Linear`, the actually multiply in the forward pass is $y = xW^\top$ rather than $y = Wx$ because the row-major memory ordering in PyTorch. This reinforced my understanding of how PyTorch manages memory and tensor layouts under the hood.

#### 4. Optimizer State Tracking
Implementing **AdamW** from the base `Optimizer` class was a masterclass in state management. I had to manually track the first and second moments ($m_t$ and $v_t$) for every parameter and ensure the decoupled weight decay was applied correctly ie distinct from the gradient update, to maintain the regularization benefits that standard Adam loses.

## 🙏 Ackowledgements

- Stanford University CS336: A profound thank you to the course instructors and material for the guidance and motivation required to implement these complex systems from the ground up.
- Xuying Li: For the excellent recommendation of the CS336 curriculum.

License: MIT
