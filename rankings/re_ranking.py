from sentence_transformers.cross_encoder import CrossEncoder
from models.constants import UMLS_API_KEY
from models.umls_graph import UMLS_API
from models.model import UMLSBERT
from utils.ranking_utils import get_similarity
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_sci_lg")

class UMLS_CrossEncoder:
    def __init__(self):
        self.model = CrossEncoder("ncbi/MedCPT-Cross-Encoder")

    def score(self, query, rels):
        return self.model.predict([[query, rel] for rel in rels]) if rels else []


umls_api = UMLS_API(UMLS_API_KEY)
umlsbert = UMLSBERT()
cross_encoder = UMLS_CrossEncoder()

def generate_context(query):
    umls_res = {}
    entities = [ent.text.lower() for ent in nlp(query).ents]

    for key in entities[:20]:
        cuis = umls_api.search_cui(key)
        if not cuis:
            continue
        cui, name = cuis[0]
        definitions = umls_api.get_definitions(cui)
        definition = next((d["value"] for d in definitions if d["rootSource"] == "MSH"), 
                          next((d["value"] for d in definitions), ""))

        relations = umls_api.get_relations(cui)
        if relations:
            rel_texts = [query] + [f"{rel.get('relatedFromIdName', '')} {rel.get('additionalRelationLabel', '').replace('_', ' ')} {rel.get('relatedIdName', '')}" for rel in relations]
            embeddings = umlsbert.batch_encode(rel_texts)
            query_embedding, relation_embeddings = embeddings[0], embeddings[1:]
            rel_scores = sorted(zip(cross_encoder.score(query, [f"{rel.get('relatedFromIdName', '')} {rel.get('additionalRelationLabel', '').replace('_', ' ')} {rel.get('relatedIdName', '')}" for rel in relations]), 
                                    [rel for _, rel in sorted(zip(get_similarity([query_embedding], relation_embeddings), relations), reverse=True)[:200]]), reverse=True)
            rels = [(rel.get("relatedFromIdName", ""), rel.get("additionalRelationLabel", "").replace("_", " "), rel.get("relatedIdName", "")) for _, rel in rel_scores[:20]]
        else:
            rels = []

        umls_res[cui] = {"name": name, "definition": definition, "rels": rels}

    context = []
    for k, v in umls_res.items():
        rels_text = "\n".join([f"({rel[0]},{rel[1]},{rel[2]})" for rel in v["rels"]])
        context.append(f"Name: {v['name']}\nDefinition: {v['definition']}\nRelations: \n{rels_text}" if rels_text else f"Name: {v['name']}\nDefinition: {v['definition']}")
    return "\n\n".join(context)
