This project is for fine-tuning and deploying a large language model. It uses a Mistral base model and QLoRA for memory-efficient training.

### Project Overview

The project is designed to fine-tune a large language model on a custom dataset of Markdown files. The process involves several steps:

1.  **Data Preparation:** The `prepare_data.py` script processes Markdown files into a JSONL file of question/answer pairs. This is done by chunking the text and using a language model to generate questions and answers.
2.  **Training:** The `train_qlora.py` script fine-tunes the base model using the generated JSONL file. It uses QLoRA to reduce memory usage during training.
3.  **Inference:** The `infer.py` script uses the trained model to answer questions.
4.  **Merging:** The `merge_lora.py` script merges the trained LoRA adapter with the base model to create a new, fine-tuned model.
5.  **Quantization:** The `Modelfile` is used to create a GGUF model from the merged model, which is a quantized format suitable for running on a CPU.

### Building and Running

**1. Prepare the data:**

```bash
python prepare_data.py --md_dir data/md --out_jsonl data/train.jsonl
```

**2. Train the model:**

```bash
python train_qlora.py \
    --base_model "mistralai/Mistral-7B-Instruct-v0.3" \
    --train_file "data/train.jsonl" \
    --output_dir "outputs/mistral7b-md-qlora"
```

**3. Merge the LoRA adapter:**

```bash
python merge_lora.py \
    "mistralai/Mistral-7B-Instruct-v0.3" \
    "outputs/mistral7b-md-qlora" \
    "outputs/mistral7b-merged"
```

**4. Run inference:**

```bash
python infer.py \
    --base_model "mistralai/Mistral-7B-Instruct-v0.3" \
    --adapter_dir "outputs/mistral7b-md-qlora" \
    --question "What is the capital of France?"
```

### Development Conventions

*   The project uses Python 3.
*   The main dependencies are `transformers`, `datasets`, `peft`, and `trl`.
*   The code is organized into separate scripts for each step of the process.
*   The scripts use command-line arguments for configuration.
*   The output of the training process is saved to the `outputs` directory.
