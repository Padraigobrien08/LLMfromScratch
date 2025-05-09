# LLM from Scratch

This repository contains an experimental project dedicated to building and training language models from scratch using PyTorch. The code in this repository covers several aspects—from a simple bigram model to a scaled-down GPT implementation capable of generating coherent text outputs. While the code provided here trains on smaller text files and toy datasets, I also experimented with training the LLM from scratch on the [Open Web Text Corpus](https://blog.openai.com/better-language-models/) (roughly 45 GB in size). Note that the large dataset is not included in this repository due to its size.

Publication of this project on ReadyTensor [Building a Transformer Based LLM from Scratch using PyTorch | ReadyTensor](https://app.readytensor.ai/publications/building-a-transformer-based-llm-from-scratch-using-pytorch-HMEzasyetWey)

## Project Structure

- **bigram.ipynb**  
  A Jupyter Notebook implementing a simple bigram language model. This notebook demonstrates:
  - Loading text data (in this case, from "wizard_of_oz.txt").
  - Mapping characters to integers (and vice versa).
  - Training a basic model using an embedding table that functions as a bigram.
  - Generating text based on a seed context.

- **chatbot.py**  
  A Python script that loads a pre-trained GPT-like language model (via a pickle file) and provides an interactive command-line chatbot interface. It:
  - Accepts a batch size parameter via command-line arguments.
  - Loads model parameters from `model-01.pkl`.
  - Utilizes a text generation function with temperature adjustment for sampling completions.

- **gpt-v1.ipynb**  
  A Jupyter Notebook that demonstrates a more complex GPT-like model:
  - Incorporates self-attention layers, feedforward blocks, and positional embeddings.
  - Shows how to load and preprocess text data.
  - Trains the GPT model on the provided data and generates sample outputs.
  
- **torch-examples.ipynb**  
  A collection of PyTorch examples:
  - Demonstrates creating various tensors (e.g., zeros, ones, random tensors).
  - Shows basic tensor operations such as matrix multiplications, stacking, transposing, and applying activation functions.
  - Provides code examples for using functions like `torch.multinomial`, `torch.softmax`, and more.
  
- **training.py**  
  A full training script for a GPT language model:
  - Uses a subword tokenizer to preprocess text data.
  - Employs memory-mapped files to efficiently load large text datasets.
  - Defines a transformer-based GPT model with multiple attention heads and transformer blocks.
  - Includes training routines with evaluation, TensorBoard logging, and learning rate scheduling.
  - Saves model checkpoints during training.

## Requirements

Make sure you have Python 3.7+ installed and the following packages (you can install them via pip):

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib pylzma ipykernel jupyter tensorboard tokenizers
```

For running specific files, check the installation commands in the notebooks (e.g., the pip install commands at the top of `bigram.ipynb`).

## Setup & Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Padraigobrien08/LLMfromScratch.git
   cd LLMfromScratch
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows, run: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt  # If you create a requirements file listing the above packages
   ```

3. **Configure Jupyter Kernel:**

   If you plan to use the notebooks, install the kernel from within the virtual environment:

   ```bash
   python -m ipykernel install --user --name=.venv --display-name="venv-gpt"
   ```

## Usage

### Running the Notebooks

- **bigram.ipynb**, **gpt-v1.ipynb**, and **torch-examples.ipynb**  
  Open these notebooks in Jupyter Notebook or Jupyter Lab:

  ```bash
  jupyter notebook
  ```

  Then navigate to and open the desired notebook.

### Running the Chatbot Script

- **chatbot.py**  
  Run the chatbot script via the command line. It requires a batch size argument:

  ```bash
  python chatbot.py -batch_size 32
  ```

  Once running, the chatbot will prompt you for input, and it will generate completions using the pre-trained model loaded from `model-01.pkl`.

### Training a GPT Model

- **training.py**  
  This script trains the GPT language model using a subword tokenizer and text data loaded from memory-mapped files. You can adjust training parameters via command-line arguments. For example:

  ```bash
  python training.py -batch_size 64 -epochs 10 -save_dir checkpoints
  ```

  The script will:
  - Log training and validation loss to TensorBoard.
  - Save model checkpoints in the specified directory.
  - Utilize gradient clipping and a learning rate scheduler for stable training.

## Experiment with the Open Web Text Corpus

I have also experimented with training this LLM from scratch on the [Open Web Text Corpus](https://openwebtext.com/), which is approximately 45 GB in size. Due to its size, I have not uploaded it to the repository. However, the `training.py` script is designed to handle larger datasets by using memory mapping and batch sampling methods. If you decide to use a large-scale dataset, make sure to adjust paths and potentially tweak hyperparameters (such as `block_size` and `max_iters`) to accommodate the dataset’s scale.

## Acknowledgments

- Special thanks to the open source community and research papers that have inspired this project.
- The project was developed by [Padraig O'Brien](https://github.com/Padraigobrien08) as a learning and research tool in building language models from scratch.
- Project was completed following a tutorial at [Youtube Video](https://youtu.be/UU1WVnMk4E8?si=7SPWXPRwDNepAqKF)
