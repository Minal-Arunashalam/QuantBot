from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 
device = "cpu" #"cuda:0" if torch.cuda.is_available() else 
#load finBERT NLP model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#move model to CPU
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
#labels for sentiment
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        #tokenize news (seperate words/chars into meaningful parts to feed into NLP model)
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)
        #pass tokens to NLP model and get predicted sentiment probabilities
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        #compute the softmax probability (0<=p<=1) of the sentiment, meaning how sure the model is of its prediction
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        #Extract the predicted sentiment label and probability
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        #return the sentiment and its probability (prediction strength) value
        return probability, sentiment
    else:
        #default return if no news
        return 0, labels[-1]


# if __name__ == "__main__":
#     tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!','traders were displeased!'])
#     print(tensor, sentiment)
#     print(torch.cuda.is_available())