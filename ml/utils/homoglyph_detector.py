HOMOGLYPH_MAP = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's',
    '6': 'g', '7': 't', '8': 'b', '@': 'a', '$': 's',
    'vv': 'w', 'rn': 'm', 'cl': 'd', 'cj': 'g'
}

TRUSTED_BRANDS = [
    'google', 'facebook', 'apple', 'microsoft', 'amazon',
    'paypal', 'netflix', 'instagram', 'twitter', 'linkedin',
    'yahoo', 'gmail', 'outlook', 'bank', 'secure', 'update'
]

def normalize_url(url):
    url = str(url).lower()
    for fake, real in HOMOGLYPH_MAP.items():
        url = url.replace(fake, real)
    return url

def detect_homoglyph(url):
    original = str(url).lower()
    normalized = normalize_url(original)
    for brand in TRUSTED_BRANDS:
        if brand in normalized and brand not in original:
            return 1
    return 0

def detect_brand_impersonation(url):
    original = str(url).lower()
    hits = []
    for brand in TRUSTED_BRANDS:
        if brand in original:
            hits.append(brand)
    return len(hits)

def get_homoglyph_features(url):
    return {
        'has_homoglyph': detect_homoglyph(url),
        'brand_impersonation_count': detect_brand_impersonation(url),
        'normalized_url': normalize_url(url)
    }
