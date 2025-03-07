import os
import pickle
import requests
import io
import re
from nltk import edit_distance

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)

app = Flask(__name__)

PICKLE_FILE_URL = "https://tp080497.blob.core.windows.net/spelling-correction-container/pretrained_model.pkl"

# Global variable to store the loaded pickle object
loaded_pickle_data = None

def load_pickle_at_startup():
    """Download and load the pickle file when the app starts."""
    global loaded_pickle_data

    try:
        response = requests.get(PICKLE_FILE_URL)
        response.raise_for_status()  # Raise an error for bad responses

        # Load the pickle file into a Python object
        loaded_pickle_data = pickle.load(io.BytesIO(response.content))
        print("Pickle file loaded successfully at startup.")
    except Exception as e:
        print(f"Error loading pickle file at startup: {e}")

# Call this function when the app starts
load_pickle_at_startup()

def generate_suggestions(word, correct_word, max_distance=2):
    suggestions = []
    for suggestion in correct_word:
        distance = edit_distance(word, suggestion, transpositions=True)
        if distance <= max_distance:
            suggestions.append((suggestion, distance))
    # Sort by edit distance first
    return [c[0] for c in sorted(suggestions, key=lambda x: x[1])][:100]


# Calculate the probability with corpus validation
def calc_prob(word, prev_word, unigram_fd, bigram_cfd, vocab_size):
    # Check if word exists in corpus file
    if word not in unigram_fd:
        return 0.0  # Treat OOV words as invalid

    # Unigram probability
    unigram_prob = unigram_fd[word] / unigram_fd.N()

    # Bigram probability if context exists
    if prev_word and prev_word in bigram_cfd:
        bigram_prob = (bigram_cfd[prev_word][word] + 1) / (unigram_fd[prev_word] + vocab_size)
    else:
        bigram_prob = unigram_prob

    return 0.1 * unigram_prob + 0.9 * bigram_prob


# Improve correction logic
def correct_spell(statement, unigram_fd, bigram_cfd, vocab_size, correct_word):
    words = statement.lower().split()
    results = []

    # Use top 10% most frequent words as threshold
    common_words = {word for word, _ in unigram_fd.most_common(int(len(unigram_fd) * 0.3))}  # 30% top freq word
    # common_words = {word for word, _ in unigram_fd.most_common(int(len(unigram_fd)*0.1))}

    for i, word in enumerate(words):
        # Skip punctuation-only tokens
        if re.fullmatch(r"[.,!?']+", word):
            continue

        # Get previous word context
        prev_word = words[i - 1] if i > 0 and words[i - 1] in unigram_fd else None

        # Generate suggestions if:
        # 1. Word is OOV (not in correct_word), or
        # 2. Word is rare (not in top 10% frequent words)
        if word not in unigram_fd or word not in common_words:
            suggestions = generate_suggestions(word, correct_word)
            if suggestions:
                scored = []
                for suggestion in suggestions:
                    # Skip suggestion if same as original
                    if suggestion == word:
                        continue

                    edit_dist = edit_distance(word, suggestion, transpositions=True)
                    suggestion_prob = calc_prob(suggestion, prev_word, unigram_fd, bigram_cfd, vocab_size)
                    error_prob = 0.7 ** edit_dist  # Less aggressive error model
                    score = suggestion_prob * error_prob
                    scored.append((suggestion, suggestion_prob, edit_dist, score))

                if scored:
                    # Sort by score and distance
                    top_suggestions = sorted(scored, key=lambda x: (-x[3], x[2]))[:5]
                    # top_suggestions = sorted(scored, key=lambda x: (-x[3], x[2]))[:5]
                    results.append((word, top_suggestions))

    return results


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/grammar-check', methods=['POST'])
def grammar_check():
   text = request.get_json()

   if text:
       unigram_fd, bigram_cfd = loaded_pickle_data
       vocab_size = len(unigram_fd)
       correct_word = list(unigram_fd.keys())

       results = correct_spell(text, unigram_fd, bigram_cfd, vocab_size, correct_word)

       return jsonify({'data': results})
   else:
       return jsonify({'message': 'please send some text'})

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
   text = request.get_json()

   if text:
       return jsonify({'data': text})
   else:
       return jsonify({'message': 'please send some text'})

if __name__ == '__main__':
   app.run()
