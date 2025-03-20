from transformers import pipeline
from datasets import load_dataset
import re
from transformers import BertTokenizer
import nltk
from nltk.corpus import wordnet
import random

# Initialize sentiment analysis pipeline for French
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Test sentences in French
sentences = [
    "J'adore ce magnifique restaurant !",
    "Ce film était vraiment terrible, je ne recommande pas.",
    "La journée est superbe aujourd'hui !"
]

# Get predictions
results = classifier(sentences)

# Extract only the labels
labels = [result['label'] for result in results]

print("\nSentiment Analysis Results:")
for sentence, label in zip(sentences, labels):
    print(f"Phrase: {sentence}")
    print(f"Sentiment: {label}\n")


# Load the Pokemon Cards dataset
pokemon_dataset = load_dataset("TheFusion21/PokemonCards")

# Display basic information about the dataset
print("\nPokemon Cards Dataset Information:")
print(pokemon_dataset)

# Display a few examples from the dataset
print("\nSample entries from the dataset:")
for i, example in enumerate(pokemon_dataset['train'].select(range(3))):
    print(f"\nCard {i+1}:")
    print(f"Name: {example['name']}")
    print(f"Type: {example['caption']}")
    print(f"HP: {example['hp']}")
    print("-" * 30)

# Get all HP values
all_hp = [card['hp'] for card in pokemon_dataset['train']]
print("\nAll HP values (first 5):")
print(all_hp[:5])

# Get all Fire type Pokemon names
fire_pokemon = [card['name'] for card in pokemon_dataset['train'] 
                if card['caption'] == 'Fire']
print("\nFire type Pokemon (first 5):")
print(fire_pokemon[:5])

# Get all Stage 1 Pokemon names
evolved_pokemon = [card['name'] for card in pokemon_dataset['train'] 
                  if card['caption'] == 'Stage 1']
print("\nStage 1 Pokemon (first 5):")
print(evolved_pokemon[:5])

# Some statistics
print("\nDataset Statistics:")
print(f"Total number of HP values: {len(all_hp)}")
print(f"Total number of Fire Pokemon: {len(fire_pokemon)}")
print(f"Total number of Stage 1 Pokemon: {len(evolved_pokemon)}")


def clean_text(text):
    if not isinstance(text, str):
        return text
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace special chars with space
    text = re.sub(r'\s+', ' ', text)      # Replace multiple spaces with single space
    text = text.strip()                    # Remove leading/trailing whitespace
    
    return text

# Test the function
test_texts = [
    "Hello <b>World</b>!!!",
    "Pokemon's HP: 100%",
    "  Multiple    Spaces   Here  ",
    "<p>Some HTML tags</p> & special chars $#@"
]

print("\nText Cleaning Examples:")
for text in test_texts:
    print(f"\nOriginal: {text}")
    print(f"Cleaned : {clean_text(text)}")

# Apply to Pokemon dataset example
cleaned_names = [clean_text(card['name']) for card in pokemon_dataset['train'].select(range(3))]
print("\nCleaned Pokemon Names:")
for name in cleaned_names:
    print(name)


def tokenize_example():
    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Get a sample text from the dataset
    sample_text = pokemon_dataset['train'][0]['caption']
    cleaned_text = clean_text(sample_text)
    
    # Tokenize the text
    tokens = tokenizer.tokenize(cleaned_text)
    token_ids = tokenizer.encode(cleaned_text, add_special_tokens=True)
    
    # Display results
    print("\nTokenization Example:")
    print(f"Original text: {sample_text}")
    print(f"Cleaned text: {cleaned_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded back: {tokenizer.decode(token_ids)}")

# Test the tokenization
tokenize_example()


# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

def get_synonym(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    
    # Get all lemmas from all synsets
    lemmas = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
    # Remove duplicates and the original word
    lemmas = list(set(lemma for lemma in lemmas if lemma != word))
    
    return random.choice(lemmas) if lemmas else word

def generate_synonym_sentence(text):
    # Clean the text first
    cleaned_text = clean_text(text)
    
    # Split into words
    words = cleaned_text.split()
    
    # Generate new sentence with synonyms
    new_words = []
    for word in words:
        try:
            synonym = get_synonym(word)
            new_words.append(synonym.replace('_', ' '))
        except:
            new_words.append(word)
    
    return ' '.join(new_words)

# Test the function
test_sentences = [
    "The fire pokemon is strong",
    "Pikachu uses thunder attack",
    "The dragon flies high in sky"
]

print("\nSynonym Generation Examples:")
for sentence in test_sentences:
    print(f"\nOriginal: {sentence}")
    print(f"With synonyms: {generate_synonym_sentence(sentence)}")


