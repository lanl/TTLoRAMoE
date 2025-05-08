from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

save_directory = "./llama3.2-1b"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)