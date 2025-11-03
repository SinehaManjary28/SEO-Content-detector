import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re
from nltk.tokenize import sent_tokenize

# -------------------- Helper Functions --------------------

def extract_title_and_body_from_html(html):
    try:
        soup = BeautifulSoup(html, 'lxml')
        for script in soup(["script", "style"]):
            script.extract()
        title = soup.title.get_text(strip=True) if soup.title else ''
        main = soup.find('article') or soup.find('main')
        if main:
            parts = [p.get_text(separator=' ', strip=True) for p in main.find_all('p')]
            body = ' '.join(parts)
        else:
            parts = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
            body = ' '.join(parts)
        if not body:
            body = soup.get_text(separator=' ', strip=True)
        body = re.sub(r'\s+', ' ', body).strip()
        return title, body
    except Exception:
        return '', ''


def estimate_syllables(word):
    vowels = 'aeiouy'
    word = word.lower()
    count = 0
    if word and word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count = max(1, count - 1)
    return max(1, count)


def compute_readability(text):
    from nltk.tokenize import sent_tokenize
    sents = max(1, len(sent_tokenize(text)))
    words = text.split()
    words_count = max(1, len(words))
    sylls = sum(estimate_syllables(w) for w in words)
    asl = words_count / sents
    asw = sylls / words_count
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return score


def analyze_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; SEO-Content-Detector/1.0)'}
        headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    }
        r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)

        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"

        title, body = extract_title_and_body_from_html(r.text)
        if not body.strip():
            return None, "Empty content extracted."

        wc = len(body.split())
        sc = len([s for s in body.split('.') if s.strip()])
        fr = compute_readability(body)

        model_path = Path('models/quality_model.pkl')
        encoder_path = Path('models/label_encoder.pkl')

        if model_path.exists() and encoder_path.exists():
            clf = joblib.load(model_path)
            le = joblib.load(encoder_path)
            features = pd.DataFrame([[wc, sc, fr]], columns=['word_count', 'sentence_count', 'flesch_reading_ease'])
            pred = clf.predict(features)
            label = le.inverse_transform(pred)[0]
        else:
            label = "Unknown"

        result = {
            "title": title,
            "word_count": wc,
            "sentence_count": sc,
            "flesch_reading_ease": round(fr, 2),
            "quality_label": label
        }
        return result, None

    except Exception as e:
        return None, str(e)

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="SEO Content Quality Detector", page_icon="🔍", layout="centered")

st.title(" SEO Content Quality & Duplicate Detector")
st.write("Analyze any webpage to assess its SEO content quality using NLP and machine learning.")

url = st.text_input("Enter webpage URL:", placeholder="https://example.com")

if st.button("Analyze"):
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Fetching and analyzing content..."):
            result, error = analyze_url(url)

        if error:
            st.error(f" Error: {error}")
        else:
            st.success(" Analysis Complete!")

            st.subheader(result['title'])
            st.write(f"**Word Count:** {result['word_count']}")
            st.write(f"**Sentence Count:** {result['sentence_count']}")
            st.write(f"**Flesch Reading Ease:** {result['flesch_reading_ease']}")
            st.write(f"**Predicted Quality:** 🏷️ {result['quality_label']}")

            # Color-coded indicator
            if result['quality_label'] == "High":
                st.markdown(" **High-quality SEO content!** Well-written and detailed.")
            elif result['quality_label'] == "Medium":
                st.markdown(" **Moderate quality.** Could be improved with more details or clarity.")
            else:
                st.markdown(" **Low-quality content.** Consider expanding or improving readability.")

st.markdown("---")
st.caption("Built with  using Streamlit, BeautifulSoup, and Scikit-learn.")
