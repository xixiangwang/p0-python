from transformers import pipeline

p = pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english")

print(p("I love learning Python"))
print(p("This is terrible"))