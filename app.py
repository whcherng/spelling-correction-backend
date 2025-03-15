import os
import pickle
import re
import difflib
from concurrent.futures import ThreadPoolExecutor

import joblib
from nltk import edit_distance, WordNetLemmatizer, word_tokenize
from flask import Flask, render_template, request, send_from_directory, jsonify
from nltk.corpus import stopwords

app = Flask(__name__)

# Load pre-trained language model
with open("pretrained_model.pkl", "rb") as f:
    unigram_fd, bigram_cfd = pickle.load(f)

correct_word_set = set(unigram_fd.keys())  # Fast lookup
top_common_words = set(w for w, _ in unigram_fd.most_common(int(len(unigram_fd) * 0.3)))  # Top 30%

with open("final_linearsvm_model.pkl", "rb") as f:
    final_linearsvm_model = joblib.load(f)


def generate_suggestions(word):
    """Get up to 5 close matches for a given word."""
    return difflib.get_close_matches(word, correct_word_set, n=5, cutoff=0.7)


def calc_prob(word, prev_word, vocab_size):
    """Calculate probability using unigram and bigram data."""
    unigram_prob = unigram_fd[word] / unigram_fd.N()
    bigram_prob = unigram_prob  # Default to unigram

    if prev_word and prev_word in bigram_cfd:
        bigram_prob = (bigram_cfd[prev_word][word] + 1) / (unigram_fd[prev_word] + vocab_size)

    return 0.1 * unigram_prob + 0.9 * bigram_prob


def process_word(word, prev_word, vocab_size, start_index):
    """Process a single word to find correction suggestions."""
    if word in correct_word_set or word in top_common_words:
        return None  # No correction needed

    suggestions = generate_suggestions(word)
    if not suggestions:
        return None

    # Score each suggestion
    scored_suggestions = [
        {
            "replacement_substring": suggestion,
            "replacement_substring_char_start": start_index,
            "replacement_substring_char_end": start_index + len(suggestion) - 1,
            "probability": calc_prob(suggestion, prev_word, vocab_size),
            "edit_distance": edit_distance(word, suggestion, transpositions=True)
        }
        for suggestion in suggestions
    ]

    # Sort by probability (desc) and edit distance (asc)
    sorted_suggestions = sorted(scored_suggestions, key=lambda x: (-x["probability"], x["edit_distance"]))

    return {
        "original_substring": word,
        "original_substring_char_start": start_index,
        "original_substring_char_end": start_index + len(word) - 1,
        "suggestions": sorted_suggestions,
    }


def correct_spell(statement, vocab_size):
    """Find spelling mistakes and provide multiple correction suggestions."""
    words = re.findall(r"\b\w+\b", statement.lower())  # Tokenize words
    prev_word = None

    # Track character positions
    char_positions = []
    for match in re.finditer(r"\b\w+\b", statement):
        char_positions.append((match.group(), match.start()))

    # Process words in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda w: process_word(w[0], prev_word, vocab_size, w[1]), char_positions)

    # Construct corrected response
    content_to_replace = [r for r in results if r]

    corrected_statement = []
    words = statement.split()

    corrections_map = {correction["original_substring"]: correction["suggestions"][0]["replacement_substring"] for correction in content_to_replace}

    for word in words:
        if word.lower() in corrections_map:
            corrected_statement.append(corrections_map[word.lower()])  # To add corrected word
        else:
            corrected_statement.append(word)

    return {
        "fixed": " ".join(corrected_statement),  # Convert back to string
        "text": statement,
        "contentToReplace": content_to_replace,
    }

# Removing HTML tags, special characters, numbers, extra whitespace, and converting to lowercase
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

# Define negation terms to exclude from stopwords
negation_words = {'not', 'no', 'never', 'nor', 'neither', 'none', 'nobody', 'nowhere', 'cannot', 'n\'t', 'nt'}

# Update stopwords by removing negation terms
stop_words = set(stopwords.words('english')) - negation_words

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = clean_text(text)
    words = word_tokenize(text)

    # Lemmatize and remove non-negation stopwords
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Handle negation by tagging subsequent words
    processed_words = []
    negation = False
    window = 0  # Number of words to tag after negation

    for word in words:
        if word in negation_words:
            negation = True
            window = 3  # Set window size (adjust as needed)
            processed_words.append(word)
        elif negation:
            processed_words.append(f"{word}_NOT")
            window -= 1
            if window <= 0:
                negation = False
        else:
            processed_words.append(word)

    return ' '.join(processed_words)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/grammar-check', methods=['POST'])
def grammar_check():
    text = request.get_json().get('text')
    if not text:
        return jsonify({'message': 'please send some text'})

    vocab_size = len(unigram_fd)
    results = correct_spell(text, vocab_size)

    return jsonify(results)


@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    text = request.get_json().get('text')

    if not text:
        return jsonify({'message': 'please send some text'})

    processed_text = preprocess_text(text)

    return jsonify({'prediction': {
        "sentiment": final_linearsvm_model.predict([processed_text])[0].item(),
        "score": final_linearsvm_model.decision_function([processed_text])[0].item()
        }
    })


if __name__ == '__main__':
    app.run()
