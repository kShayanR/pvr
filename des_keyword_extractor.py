import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
# import nltk
# nltk.download('punkt')

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

model_name = "text-to-text generation model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_key_phrases(chunk):
    model_inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
    translation = model.generate(
        **model_inputs,
        max_new_tokens=50,
        num_beams=5,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )
    key_phrases = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
    return key_phrases

def get_key_phrases_from_description(description):
    chunks = split_text(description)
    all_key_phrases = []
    unique_phrases = set()
    for chunk in chunks:
        key_phrases = extract_key_phrases(chunk)
        phrases = key_phrases.split(', ')
        for phrase in phrases:
            phrase = phrase.lower()
            if phrase not in unique_phrases:
                unique_phrases.add(phrase)
                all_key_phrases.append(str(phrase))
    return ', '.join(all_key_phrases)

df = pd.read_csv('datasets/details.csv')

tqdm.pandas()
df['keywords'] = df['description'].progress_apply(lambda x: get_key_phrases_from_description(x) if pd.notnull(x) else '')

df.to_csv('datasets/details_with_keywords.csv', index=False)

print(df.head())

