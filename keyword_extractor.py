import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def split_text(text, chunk_size=512):
    sentences = sent_tokenize(text)
    chunks = []
    # current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            break
            # chunks.append(' '.join(sentences))
            # current_chunk = []
            # current_length = 0
        chunks.append(sentence)
        current_length += sentence_length
    
    # if sentences:
    #     chunks.append(' '.join(sentences))
    
    return chunks


model_name = "text-to-text generation model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Conrad Dubai in the United Arab Emirates is an incredible destination of choice for todayâ€™s smart luxury traveller. Located conveniently located in the hub of Dubaiâ€™s commercial centre on Sheikh Zayed Road, the property is a stylish city haven within close proximity to Dubaiâ€™s international financial and convention centre, one of the worldâ€™s fastest growing airport and world-class shopping destinations. The 555-room property is the perfect celebration of one of the world's favourite luxury cities and boasts locally inspired international design and tailor-made services to help each guest feel connected to their surroundings."

# chunk_size = 20
chunks = split_text(text)

def extract_key_phrases(chunk):
    model_inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
    # translation = model.generate(**model_inputs, max_new_tokens=50)  
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


all_key_phrases = []
unique_phrases = set()
for chunk in chunks:
    key_phrases = extract_key_phrases(chunk)
    # all_key_phrases.extend(key_phrases)
    # combined_key_phrases = ' '.join(key_phrases)
    phrases = key_phrases.split(', ')
    for phrase in phrases:
        phrase = phrase.lower()
        if phrase not in unique_phrases:
            unique_phrases.add(phrase)
            all_key_phrases.append(str(phrase))

print(all_key_phrases)


