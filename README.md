# Sarcasm Detection with Various Language Models

This repository contains code and datasets for detecting sarcasm in text using different pre-trained models fine-tuned on a sarcasm detection dataset. The models used in this project include `OPT_LoRA`, `DistilGPT2_LoRA`, `Cerebras`, and `BLOOM`. 

## Repository Structure

- **`electronics_dataset.py`**: Converts a JSON dataset (`electronics.json`) into a CSV file and uses a fine-tuned `OPT_LoRA` model to predict sarcasm in product reviews. It tokenizes the text, passes it through the model, and prints predictions of whether each review is sarcastic or not.
  
- **`opt_lora.py`**: Fine-tunes the `OPT_LoRA` model on a balanced sarcasm dataset (`train-balanced-sarcasm.csv`) and saves the model. The comments are labeled as sarcastic or not sarcastic, and the model is trained using the `xturing` library.

- **`bloom.py`**: Fine-tunes the `BLOOM` model on the sarcasm dataset using the same approach as `opt_lora.py`, with labels for sarcasm classification.

- **`cerebras.py`**: Fine-tunes the `Cerebras` model in a similar manner, training it to detect sarcasm from text.

- **`distil_gpt.py`**: Fine-tunes the `DistilGPT2_LoRA` model on the sarcasm dataset for sarcasm detection.

- **Jupyter Notebooks**:
  - `load_dataset.ipynb`: A notebook for loading and preprocessing the dataset used for sarcasm detection.
  - `preprocessing.ipynb`: A notebook containing code for preprocessing the dataset.
  - `BERT.ipynb`: A notebook demonstrating how to use BERT for sarcasm detection.

## Dataset

The dataset used for fine-tuning the models is the `train-balanced-sarcasm.csv` file, which consists of labeled sarcastic and non-sarcastic comments. Preprocessing steps include removing missing values and cleaning the text.

## Models

We use the `xturing` library to fine-tune the following pre-trained models:
- **OPT_LoRA**: A lightweight version of the OPT model optimized with Low-Rank Adaptation (LoRA).
- **DistilGPT2_LoRA**: A distilled version of GPT-2 optimized with LoRA.
- **BLOOM**: A large language model trained on multiple languages.
- **Cerebras**: A model developed by Cerebras for NLP tasks.

## How to Run

1. Install required libraries:
   ```bash
   pip install transformers torch pandas xturing
2. Fine-tune the model of your choice by running one of the Python scripts:
  ```bash
  python opt_lora.py
```
This will fine-tune the OPT_LoRA model and save it in the repository.
3. Use the model for sarcasm detection:
```bash
python electronics_dataset.py
```
## Results
After fine-tuning the models, they are used to classify comments as sarcastic or not. The predictions are outputted to the console, indicating whether each text is labeled sarcastic or not.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
Thanks to the authors of the xturing library for making model fine-tuning accessible.
Special thanks to the creators of the sarcasm dataset for their contribution to this project.
csharp
