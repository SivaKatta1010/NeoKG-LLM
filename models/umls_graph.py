import requests


class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"

    def search_cui(self, query):
        try:
            response = requests.get(self.search_url, params={"string": query, "apiKey": self.apikey, "pageNumber": 1, "pageSize": 1})
            response.raise_for_status()
            results = response.json().get("result", {}).get("results", [])
            return [(res["ui"], res["name"]) for res in results]
        except Exception as e:
            print(e)
            return []

    def get_definitions(self, cui):
        try:
            response = requests.get(self.content_url + self.content_suffix.format(cui, "definitions", self.apikey))
            response.raise_for_status()
            return response.json().get("result", [])
        except Exception as e:
            print(e)
            return []

    def get_relations(self, cui, pages=20):
        relations = []
        try:
            for page in range(1, pages + 1):
                response = requests.get(self.content_url + self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}")
                response.raise_for_status()
                relations.extend(response.json().get("result", []))
        except Exception as e:
            print(e)
        return relations
