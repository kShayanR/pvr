from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
model_name = "zero-shot classification model"

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
    for i, (label, score) in enumerate(labeled_scores):
        print(f" {i+1}- {label}: {score:.4f}")

text = "I love playing football on sunny days."
candidate_labels = ["sports", "weather", "hobbies", "music"]

zero_shot_classify(text, candidate_labels)

