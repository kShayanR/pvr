import pandas as pd

df = pd.read_csv('datasets/details.csv')

unique_phrases = set()

def add_phrases_to_set(phrases, phrase_set):
    if pd.notnull(phrases):
        phrase_list = phrases.split(', ')
        for phrase in phrase_list:
            phrase_set.add(phrase.lower())

df['amenities'].apply(lambda x: add_phrases_to_set(x, unique_phrases))
# df['keywords'].apply(lambda x: add_phrases_to_set(x, unique_phrases))

general_list = list(unique_phrases)

# print(general_list)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
model_dir = "zero-shot classification model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

def zero_shot_classify(text, candidate_labels):
    result = classifier(text, candidate_labels, multi_label=True)
    
    labels = result['labels']
    scores = result['scores']
    
    labeled_scores = list(zip(labels, scores))
    labeled_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Text: {text}")
    print("Classification Results:")
    for idx, (label, score) in enumerate(labeled_scores):
        if score >= 0.70:
            print(f" {idx+1}- {label}: {score:.4f}")

text = "The most relaxing Monday! My sister and I attended the pool at the Plazzo Versace Hotel today and we had the most relaxing time on the comfiest sunbeds ever! For lunch we ordered the avocado and prawn toast and a ceasar salad, both of which were excellent. As it was super hot today, the bar tender suggested he make us frozen daquris, which were so tasty we both had two! Overall a lovely experience and we can't wait to come back."
candidate_labels = general_list

zero_shot_classify(text, candidate_labels)