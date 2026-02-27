import requests

def follow_redirects(url, max_hops=5, timeout=5):
    try:
        hops = []
        current_url = url
        for _ in range(max_hops):
            response = requests.head(current_url, allow_redirects=False, timeout=timeout)
            hops.append(current_url)
            if response.status_code in (301, 302, 303, 307, 308):
                current_url = response.headers.get('Location', current_url)
            else:
                break
        return {
            'redirect_count': len(hops) - 1,
            'final_url': current_url,
            'redirect_chain': hops
        }
    except:
        return {
            'redirect_count': 0,
            'final_url': url,
            'redirect_chain': [url]
        }

def get_redirect_features(url):
    result = follow_redirects(url)
    return {
        'redirect_count': result['redirect_count'],
        'has_redirect': 1 if result['redirect_count'] > 0 else 0,
        'multi_hop_redirect': 1 if result['redirect_count'] > 2 else 0,
        'final_url': result['final_url']
    }
