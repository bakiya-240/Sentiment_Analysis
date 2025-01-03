import os
from flask import Flask, render_template, request, redirect, url_for
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import nltk
from werkzeug.utils import secure_filename

nltk.download('stopwords')

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure 'uploads' directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Check if a file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def my_form_post():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                data = pd.read_csv(filepath)
            except Exception as e:
                return f"Error reading the uploaded file: {str(e)}"
            
            if 'text' not in data.columns:
                return "Dataset must contain a 'text' column for analysis."
            
            analyzer = SentimentIntensityAnalyzer()

            sentiments = []
            positive_scores = []
            neutral_scores = []
            negative_scores = []

            for text in data['text']:
                sentiment_scores = analyzer.polarity_scores(str(text))  # Ensure text is a string
                sentiments.append(sentiment_scores['compound'])
                positive_scores.append(sentiment_scores['pos'])
                neutral_scores.append(sentiment_scores['neu'])
                negative_scores.append(sentiment_scores['neg'])
            
            data['sentiment'] = sentiments
            data['positive'] = positive_scores
            data['neutral'] = neutral_scores
            data['negative'] = negative_scores

            positive_pct = sum(positive_scores) / len(positive_scores)
            neutral_pct = sum(neutral_scores) / len(neutral_scores)
            negative_pct = sum(negative_scores) / len(negative_scores)
            
            labels = ['Positive', 'Neutral', 'Negative']
            scores = [positive_pct, neutral_pct, negative_pct]
            fig, ax = plt.subplots()
            ax.bar(labels, scores, color=['green', 'blue', 'red'])
            ax.set_title('Sentiment Distribution')

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close(fig)  # Close the figure to free up memory

            return render_template('form.html', data=data.to_html(classes='table table-striped'), 
                                   img_base64=img_base64, 
                                   positive_pct=positive_pct, neutral_pct=neutral_pct, negative_pct=negative_pct)
    
    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True)
