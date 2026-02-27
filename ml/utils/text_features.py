import re
import math

URGENCY_WORDS = [
    'urgent', 'immediately', 'account suspended', 'verify now',
    'click here', 'act now', 'limited time', 'expires', 'warning',
    'congratulations', 'winner', 'prize', 'free', 'risk', 'threat',
    'unusual activity', 'unauthorized', 'confirm your', 'update your'
]

PHISHING_PATTERNS = [
    r'dear\s+(customer|user|member|account holder)',
    r'click\s+(here|below|this link)',
    r'verify\s+your\s+(account|identity|email|information)',
    r'your\s+account\s+(has been|will be|is)\s+(suspended|blocked|locked)',
    r'(win|won|winner).{0,20}(prize|reward|gift|cash)',
    r'(update|confirm|validate).{0,20}(password|credential|information)',
]

def calculate_text_entropy(text):
    if not text or len(text) == 0:
        return 0
    freq = {}
    for c in str(text):
        freq[c] = freq.get(c, 0) + 1
    entropy = 0
    for count in freq.values():
        p = count / len(text)
        entropy -= p * math.log2(p)
    return round(entropy, 4)

def count_urgency_words(text):
    text = str(text).lower()
    return sum(1 for w in URGENCY_WORDS if w in text)

def count_phishing_patterns(text):
    text = str(text).lower()
    return sum(1 for p in PHISHING_PATTERNS if re.search(p, text))

def count_urls_in_text(text):
    return len(re.findall(r'http[s]?://\S+', str(text)))

def count_html_tags(text):
    return len(re.findall(r'<[^>]+>', str(text)))

def extract_text_features(text):
    text = str(text) if text else ''
    words = text.split()
    sentences = re.split(r'[.!?]', text)
    features = {
        'text_length': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': round(sum(len(w) for w in words) / max(len(words), 1), 4),
        'urgency_word_count': count_urgency_words(text),
        'phishing_pattern_count': count_phishing_patterns(text),
        'url_count_in_text': count_urls_in_text(text),
        'html_tag_count': count_html_tags(text),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'capital_ratio': round(sum(1 for c in text if c.isupper()) / max(len(text), 1), 4),
        'digit_ratio': round(sum(1 for c in text if c.isdigit()) / max(len(text), 1), 4),
        'text_entropy': calculate_text_entropy(text),
        'has_html': 1 if count_html_tags(text) > 0 else 0,
        'has_url_in_text': 1 if count_urls_in_text(text) > 0 else 0,
    }
    return features
