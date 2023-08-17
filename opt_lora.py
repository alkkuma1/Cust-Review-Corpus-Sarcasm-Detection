import pandas as pd
df = pd.read_csv('./data/train-balanced-sarcasm.csv')
df.head()
df = df[df['comment'].notna()]
df = df[df['comment'] != '']
train_text = df['comment'].tolist()
train_labels = df['label'].tolist()
train_labels = ["sarcastic" if x == 0 else "not sarcastic" for x in train_labels]
from xturing.datasets.text_dataset import TextDataset
dataset = TextDataset({
    "text": train_text,
    "target": train_labels
})
from xturing.models import BaseModel
model = BaseModel.create("opt_lora")
model.finetune(dataset=dataset)
model.save("./opt_lora")