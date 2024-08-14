import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk
# nltk.download('punkt')
import os

parser = argparse.ArgumentParser(description="Extract user preferences within a specified range.")
parser.add_argument('--start', type=int, default=0, help='Start index for the range of users')
parser.add_argument('--stop', type=int, default=500, help='Stop index for the range of users')
args = parser.parse_args()

users_df = pd.read_csv('datasets/users.csv')
reviews_df = pd.read_csv('datasets/reviews.csv')
details_df = pd.read_csv('datasets/details.csv')

users_df['preferences'] = users_df['preferences'].astype(str)

output_file = 'datasets/users_with_preferences.csv'
if os.path.exists(output_file):
    saved_users_df = pd.read_csv(output_file)
    users_df = pd.merge(users_df, saved_users_df[['user_id', 'preferences']], on='user_id', how='left')
else:
    users_df['preferences'] = ''


###--- Local Approach ---###

zero_shot_model_name = "zero-shot classification model"
zero_shot_tokenizer = AutoTokenizer.from_pretrained(zero_shot_model_name)
zero_shot_model = AutoModelForSequenceClassification.from_pretrained(zero_shot_model_name)
zero_shot_classifier = pipeline("zero-shot-classification", model=zero_shot_model, tokenizer=zero_shot_tokenizer) # device=0 >> to use a hardware accelerator e.g. GPU

text_to_text_model_name = "text-to-text generation model"
text_to_text_tokenizer = AutoTokenizer.from_pretrained(text_to_text_model_name)
text_to_text_model = AutoModelForSeq2SeqLM.from_pretrained(text_to_text_model_name)


###--- Online Approach ---###

# zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# text_to_text_tokenizer = AutoTokenizer.from_pretrained("beogradjanka/bart_finetuned_keyphrase_extraction")
# text_to_text_model = AutoModelForSeq2SeqLM.from_pretrained("beogradjanka/bart_finetuned_keyphrase_extraction")

#############################

def split_text(text, chunk_size=512):
    sentences = sent_tokenize(text)
    chunks = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            break
        chunks.append(sentence)
        current_length += sentence_length
    return chunks

def extract_key_phrases(chunk):
    model_inputs = text_to_text_tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
    translation = text_to_text_model.generate(
        **model_inputs,
        max_new_tokens=100,
        num_beams=5,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )
    key_phrases = text_to_text_tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
    return key_phrases

def zero_shot_classify(text, candidate_labels):
    result = zero_shot_classifier(text, candidate_labels, multi_label=True)
    labels = result['labels']
    scores = result['scores']
    labeled_scores = [(label, score) for label, score in zip(labels, scores) if score >= 0.60]
    return labeled_scores

start_idx = args.start
stop_idx = args.stop if args.stop is not None else users_df.shape[0]

for index, user in tqdm(users_df.iloc[start_idx:stop_idx].iterrows(), total=stop_idx-start_idx, desc="Processing users"):
    user_id = user['user_id']
    user_reviews = reviews_df[reviews_df['user_id'] == user_id]
    user_preferences = set()
    
    for _, review in user_reviews.iterrows():
        location_id = review['location_id']
        location_info = details_df[details_df['location_id'] == location_id].iloc[0]
        amenities = location_info['amenities'] if pd.notnull(location_info['amenities']) else ''
        keywords = location_info['keywords'] if pd.notnull(location_info['keywords']) else ''
        
        candidate_labels = []
        if amenities:
            candidate_labels += amenities.split(', ')
        if keywords:
            candidate_labels += keywords.split(', ')
        
        if not candidate_labels:
            chunks = split_text(review['text'])
            for chunk in chunks:
                extracted_phrases = extract_key_phrases(chunk).split(', ')
                candidate_labels += extracted_phrases
        
        candidate_labels = list(set(candidate_labels)) 
        classification_results = zero_shot_classify(review['text'], candidate_labels)
        
        # user_preferences.update(classification_results)

        for label, _ in classification_results:
            user_preferences.add(label.lower())
    
    users_df.at[index, 'preferences'] = ', '.join(user_preferences)


if os.path.exists(output_file):
    saved_users_df.update(users_df.iloc[start_idx:stop_idx])
    saved_users_df.to_csv(output_file, index=False)
else:
    users_df.to_csv(output_file, index=False)

# users_df.to_csv('datasets/users_with_preferences.csv', index=False)