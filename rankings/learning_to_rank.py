import numpy as np
import spacy
from sklearn.ensemble import RandomForestRegressor
from models.constants import *
from models.graph import Neo4jKnowledgeGraph
from models.model import UMLSBERT
from utils.ranking_utils import get_similarity

# Load the spaCy model
nlp = spacy.load("en_core_sci_lg")

kg = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
umlsbert = UMLSBERT()

def learning_to_rank(query_embedding, relation_embeddings, relations):
    model = RandomForestRegressor()
    features = np.array([get_similarity([query_embedding], [emb])[0] for emb in relation_embeddings])
    model.fit(features, range(len(features)))
    scores = model.predict(features)

    # Sort relations by the predicted scores in descending order
    ranked_relations = sorted(zip(relations, scores), key=lambda x: x[1], reverse=True)
    top_20_relations = ranked_relations[:20]

    rels = []
    for _, rel in top_20_relations:
        related_from_id_name = _.get("relatedFromIdName")
        additional_relation_label = (_.get('additionalRelationLabel', '') or '').replace('_', ' ')
        related_id_name = _.get("relatedIdName")

        rels.append((related_from_id_name, additional_relation_label, related_id_name))

    return rels

def generate_context(query):
    umls_res = {}
    results = nlp(query)
    entities = [ent.text.lower() for ent in results.ents]

    for key in entities[:20]:
        cuis = kg.search_cui(key)

        if not cuis:
            continue
        cui = cuis[0]['cui']
        name = cuis[0]['name']
        definition = kg.get_definitions(cui)

        relations = kg.get_relations(cui)
        if relations:
            relation_texts = [query] + [f"{rel['relatedFromIdName']} {(rel.get('additionalRelationLabel', '') or '').replace('_', ' ')} {rel['relatedIdName']}" for rel in relations]
            embeddings = umlsbert.batch_encode(relation_texts)
            query_embedding = embeddings[0]
            relation_embeddings = embeddings[1:]
            rels = learning_to_rank(query_embedding, relation_embeddings, relations)
        else:
            rels = []

        umls_res[cui] = {"name": name, "definition": definition, "rels": rels}

    context_parts = []
    for k, v in umls_res.items():
        rels_text = "\n".join([f"({rel[0]},{rel[1]},{rel[2]})" for rel in v["rels"]])
        context_parts.append(f"Name: {v['name']}\nDefinition: {v['definition']}\nRelations: \n{rels_text}")
    return "\n\n".join(context_parts)