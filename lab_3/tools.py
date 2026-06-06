import requests

_HEADERS = {"User-Agent": "ITMO-Lab3-Agent/1.0 (education; contact@example.com)"}


def search_openalex(query: str, per_page: int = 5) -> list:
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": per_page,
        "select": "id,display_name,publication_year,abstract_inverted_index,authorships",
    }
    response = requests.get(url, params=params, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()["results"]


def invert_abstract(inv_idx: dict) -> str:
    if not inv_idx:
        return ""
    words = []
    for token, positions in inv_idx.items():
        for pos in positions:
            words.append((pos, token))
    words.sort(key=lambda x: x[0])
    return " ".join(token for _, token in words)


def search_wikipedia(query: str) -> str:
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }
    search_resp = requests.get(search_url, params=search_params, headers=_HEADERS, timeout=30)
    if search_resp.status_code != 200:
        return ""
    hits = search_resp.json().get("query", {}).get("search", [])
    if not hits:
        title = query.replace(" ", "_")
    else:
        title = hits[0]["title"].replace(" ", "_")

    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title
    response = requests.get(url, timeout=30, headers=_HEADERS)
    if response.status_code != 200:
        return ""
    data = response.json()
    return data.get("extract", "")
