from flask import Flask, request, jsonify
import os

from utils import categorize_question, summarize_transcript


app = Flask(__name__)


@app.route('/categorize', methods=['POST'])
def categorize_question_route():
    question = request.json['question']
    response = categorize_question(question)
    return jsonify({'response': response})

@app.route('/summarize', methods=['POST'])
def summarize_text_route():
    text = request.json['text']
    response = summarize_transcript(text)
    print(f"Summarized text: {response}")
    return jsonify({'response': response})

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=5002, host='0.0.0.0')

def create_app():
    return app