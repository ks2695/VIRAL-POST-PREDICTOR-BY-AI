from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import os
import webbrowser
import threading
import random
import matplotlib
matplotlib.use('Agg') # Used to safely generate plots in the background
import matplotlib.pyplot as plt
import sqlite3
import io
import base64
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Initialize SQLite Database for Analytics Tracking
def init_db():
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ig_url TEXT,
            caption TEXT,
            followers INTEGER,
            hashtags INTEGER,
            likes INTEGER,
            comments INTEGER,
            sentiment TEXT,
            prediction TEXT,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Auto-create model if not exists
if not os.path.exists("model.pkl"):
    import model

model_data = pickle.load(open("model.pkl", "rb"))
if isinstance(model_data, dict):
    ml_model = model_data.get('model')
    scaler = model_data.get('scaler')
else:
    ml_model = model_data
    scaler = None

# Hybrid Meta-Data Extractor
def extract_social_meta(url):
    meta = {'caption': '', 'username': '@anonymous'}
    if not url: return meta
    
    try:
        parsed = urlparse(url)
        # Extract basic username from URL structure (e.g., instagram.com/username)
        if 'instagram.com' in parsed.netloc or 'twitter.com' in parsed.netloc or 'x.com' in parsed.netloc:
            parts = [p for p in parsed.path.split('/') if p and p not in ['p', 'reel', 'status']]
            if parts:
                meta['username'] = f"@{parts[0]}"
    except:
        pass
        
    try:
        # Safe realistic headers to bypass basic blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            og_desc = soup.find('meta', property='og:description')
            if og_desc and og_desc.get('content'):
                meta['caption'] = og_desc.get('content')
            elif soup.title:
                meta['caption'] = soup.title.text
    except Exception:
        pass
    return meta

@app.route('/')
def home():
    if 'logged_in' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
        
    ig_url = request.form.get('ig_url', '').strip()
    caption = request.form.get('caption', '').strip()
    post_time = request.form.get('post_time', 'Morning')

    # Safely parse optional inputs
    def parse_int(field):
        val = request.form.get(field, '').replace(',', '')
        return int(val) if val.isdigit() else None
        
    # Attempt Real Extraction
    extraction_notice = ""
    username = "@anonymous"
    if ig_url:
        meta = extract_social_meta(ig_url)
        username = meta['username']
        if not caption and meta['caption']:
            caption = meta['caption']
        
        extraction_notice = "⚠️ Some values are AI-estimated due to API limitations"

    followers = parse_int('followers')
    hashtags = parse_int('hashtags')
    likes = parse_int('likes')
    comments = parse_int('comments')

    is_simulated = False
    if ig_url or followers is None:
        is_simulated = True
        if followers is None:
            followers = 5000 # Deterministic baseline instead of random
        if not caption:
            caption = "Sharing this amazing content! 🚀 #viral #trend"

    # Extract hashtags from caption if count is not provided
    extracted_hashtags = [w for w in caption.split() if w.startswith('#')]
    if hashtags is None:
        hashtags = len(extracted_hashtags) if extracted_hashtags else 5

    # 1. Mock Engagement Estimation (Feeds the ML Model)
    if likes is None or comments is None:
        base_engagement = followers * 0.05
        time_multiplier = 1.2 if post_time in ['Evening', 'Night'] else 1.0
        if likes is None:
            likes = int((base_engagement * time_multiplier) + (hashtags * 10))
        if comments is None:
            comments = int(likes * 0.1)

    # Input Validation: Prevent unrealistic values
    if followers <= 0: followers = 1
    if likes > followers: likes = followers
    if comments >= likes and likes > 0: comments = likes - 1
    elif comments >= likes: comments = 0

    # 2. Sentiment Analysis
    positive_words = ['great', 'awesome', 'good', 'amazing', 'love', 'best', 'viral', 'trend', 'happy', 'fire']
    negative_words = ['bad', 'hate', 'terrible', 'worst', 'sad', 'boring']
    caption_lower = caption.lower()
    pos_count = sum(1 for word in positive_words if word in caption_lower)
    neg_count = sum(1 for word in negative_words if word in caption_lower)
    sentiment = "Positive 🌟" if pos_count > neg_count else "Negative 📉" if neg_count > pos_count else "Neutral 😐"

    # 3. True Hybrid ML + Rule-Based Logic
    features = np.array([[followers, hashtags, likes, comments]])
    if scaler:
        model_features = scaler.transform(features)
    else:
        model_features = features

    ml_prediction = ml_model.predict(model_features)[0]
    
    try:
        ml_probability = ml_model.predict_proba(model_features)[0][1] * 100
    except:
        ml_probability = 85.0 if ml_prediction == 1 else 15.0

    # Calculate concrete Engagement Rate
    engagement_rate = (likes + comments) / followers
    engagement_pct = engagement_rate * 100

    # Hybrid Scoring Engine
    # Engagement (70% weight, maxed at 10% engagement)
    score_eng = min(70.0, (engagement_rate / 0.10) * 70.0)
    
    # Sentiment (15% weight)
    if "Positive" in sentiment:
        score_sent = 15.0
    elif "Neutral" in sentiment:
        score_sent = 7.0
    else:
        score_sent = 0.0
        
    # Hashtags (15% weight)
    if 5 <= hashtags <= 15:
        score_hash = 15.0
    else:
        score_hash = 5.0

    # Final Probability Score
    probability = round(score_eng + score_sent + score_hash, 1)

    # Result Categorization based on Engagement Rate
    if engagement_rate > 0.06:
        result = "🔥 Viral Post"
        status_class = "viral"
    elif engagement_rate >= 0.03:
        result = "📈 Medium Reach"
        status_class = "medium"
    else:
        result = "❌ Not Viral"
        status_class = "not-viral"
        
    # AI Reasoning Explanation
    sent_text = "positive" if score_sent == 15.0 else "neutral" if score_sent == 7.0 else "negative"
    hash_text = "optimal" if score_hash == 15.0 else "suboptimal"
    eng_text = "High" if engagement_rate > 0.06 else "Moderate" if engagement_rate >= 0.03 else "Low"
    reasoning = f"{eng_text} engagement ({engagement_pct:.1f}%) + {sent_text} sentiment + {hash_text} hashtag count → {result}"
    
    # 4. Hashtag Intelligence Engine
    words = [w for w in caption.split() if len(w) > 3 and not w.startswith('#')]
    base_tag = words[0].capitalize() if words else "Viral"
    suggested_hashtags = {
        'trending': ["#viral", "#trending", "#explorepage"],
        'medium': [f"#{base_tag}Vibes", f"#{base_tag}Life", "#Explore"],
        'low': [f"#{base_tag}Daily", f"#{base_tag}Post", "#NewPost"]
    }

    # Trend Analysis & AI Score Boost
    trending_topics = ['viral', 'trend', 'challenge', 'ai', 'dance', 'fashion']
    is_trending = any(topic in caption_lower for topic in trending_topics)
    
    if is_trending:
        trend_message = "🔥 Matches Current Trends! (+Boost applied)"
        # Apply a boost to probability if it matches trends
        probability = min(99.9, probability + 5.5)
    else:
        trend_message = "📅 Standard Content"

    engagement_score = round(engagement_pct, 2)

    # Debugging / Logging Engine
    print("\n" + "="*40)
    print("📊 AI PREDICTION DEBUG LOG")
    print("="*40)
    print(f"Inputs       : Followers={followers} | Likes={likes} | Comments={comments} | Hashtags={hashtags}")
    print(f"Engagement   : {engagement_score}%")
    print(f"Sentiment    : {sentiment}")
    print(f"ML Features  : {features}")
    print(f"ML Output    : Pred={ml_prediction} | Prob={ml_probability}%")
    print(f"Hybrid Final : {result} | {probability}%")
    print("="*40 + "\n")

    # 5. Save Request to Analytics Database
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (ig_url, caption, followers, hashtags, likes, comments, sentiment, prediction, probability)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ig_url, caption, followers, hashtags, likes, comments, sentiment, result, probability))
    conn.commit()
    conn.close()

    return render_template('index.html', 
                           prediction_text=result, 
                           status_class=status_class, 
                           probability=probability, 
                           likes=likes, 
                           comments=comments, 
                           sentiment=sentiment, 
                           hashtags=suggested_hashtags,
                           extracted_hashtags=extracted_hashtags,
                           caption=caption,
                           trend_message=trend_message,
                           is_simulated=is_simulated,
                           extraction_notice=extraction_notice,
                           username=username,
                           followers=followers,
                           engagement_score=engagement_score,
                           ig_url=ig_url,
                           post_time=post_time,
                           reasoning=reasoning,
                           confidence_score=probability)

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
        
    # Fetch analytics from database
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions")
    rows = c.fetchall()
    conn.close()

    total_predictions = len(rows)
    viral_count = sum(1 for r in rows if "Viral" in r[8] and "Not Viral" not in r[8])
    not_viral_count = total_predictions - viral_count

    viral_pct = round((viral_count / total_predictions * 100), 1) if total_predictions > 0 else 0
    not_viral_pct = round((not_viral_count / total_predictions * 100), 1) if total_predictions > 0 else 0

    pie_url = None
    bar_url = None
    
    if total_predictions > 0:
        # Generate Viral vs Not Viral Pie Chart
        plt.figure(figsize=(5, 4))
        plt.pie([viral_count, not_viral_count], labels=['Viral', 'Not Viral'], autopct='%1.1f%%', colors=['#28a745', '#dc3545'])
        plt.title('Prediction Distribution')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
        img.seek(0)
        pie_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        # Generate Engagement Bar Chart
        plt.figure(figsize=(5, 4))
        likes_data = [r[5] for r in rows]
        comments_data = [r[6] for r in rows]
        plt.bar(['Avg Likes', 'Avg Comments'], [sum(likes_data)/len(likes_data), sum(comments_data)/len(comments_data)], color=['#007bff', '#ffc107'])
        plt.title('Average Engagement Estimations')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
        img.seek(0)
        bar_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

    return render_template('dashboard.html', 
                           total=total_predictions, 
                           viral_pct=viral_pct, 
                           not_viral_pct=not_viral_pct,
                           pie_chart=pie_url,
                           bar_chart=bar_url,
                           history=reversed(rows[-10:])) # Send last 10 requests to show in a table

# AUTO OPEN CHROME
def open_browser():
    try:
        chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"
        webbrowser.get(chrome_path).open("http://127.0.0.1:5000/")
    except:
        webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(2, open_browser).start()
    app.run(debug=True, use_reloader=False)