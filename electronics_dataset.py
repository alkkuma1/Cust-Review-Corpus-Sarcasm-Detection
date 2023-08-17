import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

with open("electronics.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

with open("electronics.csv", "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["text"])
    for item in data:
        csv_writer.writerow([item["text"]])

model = AutoModelForSequenceClassification.from_pretrained("opt_lora.bin")
tokenizer = AutoTokenizer.from_pretrained("opt_lora.bin")

csv_data = []
with open("electronics.csv", "r", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header
    for row in csv_reader:
        csv_data.append(row[0])

results = []
for text in csv_data:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits)
    results.append(prediction.item())

# Print the results
for i, result in enumerate(results):
    print(f"Row {i+1}: {'Sarcastic' if result == 0 else 'Not Sarcastic'} ({result})")
