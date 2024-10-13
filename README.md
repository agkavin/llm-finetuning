

# Fine-tuning Phi-3 Mini LLM on ArXiv Math Dataset Using QLoRA 

### QLoRA Fine-Tuning Method

QLoRA (Quantized Low-Rank Adaptation) is a fine-tuning technique that reduces the computational and memory overhead typically associated with training large language models. It works by applying two key optimizations:

1. **Quantization**: The model's weights are reduced in precision (e.g., from 32-bit to 8-bit), which significantly decreases the memory usage without drastically compromising model performance. This makes it possible to run large models on consumer-grade hardware.

2. **Low-Rank Adaptation (LoRA)**: Instead of updating all the model's parameters during fine-tuning, LoRA modifies a smaller, low-rank subset of the model’s weight matrices. This allows for more efficient training, as only the essential parts of the model are adapted to the new data.

--- 

## 1. Project Overview

This project focuses on fine-tuning the [Phi-3-mini-4k-instruct](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct) Large Language Model (LLM) using a dataset specialized in mathematical queries from ArXiv. The aim is to enhance the model’s ability to handle domain-specific tasks efficiently, particularly within the field of mathematics. The fine-tuned model, available on Hugging Face, is optimized for answering technical and mathematical questions. The model is deployed locally using OpenWebUI for user-friendly interaction.

## 2. Technologies Used

- **Model Architecture**: [Phi-3-mini-4k-instruct](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct) based on the Transformer architecture.
- **Fine-Tuning Method**: QLoRA (Quantized Low-Rank Adaptation).
- **Deployment**: Ollama for local hosting and OpenWebUI for browser-based interaction.
- **Dataset**: [arxiv-math-instruct-Unsloth-50k](https://huggingface.co/datasets/0xZee/arxiv-math-instruct-Unsloth-50k).

## 3. How to Use

### 3.1 Dataset

The dataset used for fine-tuning the model consists of mathematical queries and answers from ArXiv:

- **Dataset Link**: [arxiv-math-instruct-Unsloth-50k](https://huggingface.co/datasets/0xZee/arxiv-math-instruct-Unsloth-50k).
- **Format**: The dataset is preprocessed into instruction-response pairs to train the model on how to generate answers to technical math problems.

### 3.2 Base Model

The base model, [Phi-3-mini-4k-instruct](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct), is a transformer-based model optimized for instruction-following tasks. It serves as the foundation for the fine-tuning process.

### 3.3 Fine-Tuned Model

After fine-tuning, the final model is hosted on Hugging Face:

- **Fine-Tuned Model**: [Phi-3-mini_ft_arxiv-math-Q8_0-GGUF](https://huggingface.co/agkavin/Phi-3-mini_ft_arxiv-math-Q8_0-GGUF).
- **Model Use Case**: This fine-tuned version can answer complex mathematical questions with improved precision and relevance, optimized for math-specific applications.

### 3.4 Running the Model Locally

To interact with the model locally using a web interface, you can use OpenWebUI and Ollama. Below are the steps to set up and run the model locally:

#### Step 1: Install OpenWebUI and Dependencies
To install OpenWebUI for hosting the model, first, set up your Python environment. Run the following commands:

```bash
pip install openwebui ollama transformers
```

#### Step 2: Download the Fine-Tuned Model
You can directly download the fine-tuned model from Hugging Face using the `transformers` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "agkavin/Phi-3-mini_ft_arxiv-math-Q8_0-GGUF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

#### Step 3: Start OpenWebUI

OpenWebUI enables browser-based interaction with the model. Once the model is downloaded and loaded into memory, run OpenWebUI:

```bash
openwebui serve 
```

This will launch the UI on your local server at `http://localhost:5000`.

### 3.5 Querying the Model

Once the UI is up and running, you can use it to interact with the model by entering math-related queries. For example, ask the model to solve equations or explain mathematical concepts, and it will generate answers in real time.
