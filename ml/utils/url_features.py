import re
import math
from urllib.parse import urlparse
from ml.utils.homoglyph_detector import get_homoglyph_features

SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'secure', 'update', 'account', 'banking',
    'confirm', 'password', 'credential', 'signin', 'wallet',
    'alert', 'suspend', 'unlock', 'validate', 'authorization'
]

SHORTENERS = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
    'shorturl.at', 'is.gd', 'buff.ly', 'adf.ly', 'tiny.cc'
]

def calculate_entropy(text):
    if not text:
        return 0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    entropy = 0
    for count in freq.values():
        p = count / len(text)
        entropy -= p * math.log2(p)
    return round(entropy, 4)

def extract_url_features(url):
    url = str(url)
    try:
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
    except:
        domain, path, query = '', '', ''

    homoglyph = get_homoglyph_features(url)

    features = {
        'url_length': len(url),
        'domain_length': len(domain),
        'path_length': len(path),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_underscores': url.count('_'),
        'num_slashes': url.count('/'),
        'num_at': url.count('@'),
        'num_question': url.count('?'),
        'num_ampersand': url.count('&'),
        'num_equal': url.count('='),
        'num_percent': url.count('%'),
        'num_digits': sum(c.isdigit() for c in url),
        'digit_ratio': round(sum(c.isdigit() for c in url) / max(len(url), 1), 4),
        'letter_ratio': round(sum(c.isalpha() for c in url) / max(len(url), 1), 4),
        'url_entropy': calculate_entropy(url),
        'domain_entropy': calculate_entropy(domain),
        'has_ip': 1 if re.match(r'http[s]?://\d+\.\d+\.\d+\.\d+', url) else 0,
        'is_https': 1 if url.startswith('https') else 0,
        'has_port': 1 if re.search(r':\d{2,5}', domain) else 0,
        'num_subdomains': len(domain.split('.')) - 2 if domain else 0,
        'is_shortener': 1 if any(s in url for s in SHORTENERS) else 0,
        'has_suspicious_keyword': 1 if any(k in url.lower() for k in SUSPICIOUS_KEYWORDS) else 0,
        'suspicious_keyword_count': sum(1 for k in SUSPICIOUS_KEYWORDS if k in url.lower()),
        'has_double_slash': 1 if '//' in path else 0,
        'has_prefix_suffix': 1 if '-' in domain else 0,
        'query_length': len(query),
        'num_query_params': len(query.split('&')) if query else 0,
        'has_homoglyph': homoglyph['has_homoglyph'],
        'brand_impersonation_count': homoglyph['brand_impersonation_count'],
    }
    return features
