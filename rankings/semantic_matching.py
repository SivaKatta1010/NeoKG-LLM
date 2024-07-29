import numpy as np
import spacy
from models.constants import *
from models.graph import Neo4jKnowledgeGraph
from models.model import UMLSBERT
from utils.ranking_utils import get_similarity

# Load the spaCy model
nlp = spacy.load("en_core_sci_lg")
kg = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
bert = UMLSBERT()

def semantic_matching(query_embedding, relation_embeddings, relations):
    similarity_scores = get_similarity([query_embedding], relation_embeddings)[0]
    scored_relations = sorted(zip(relations, similarity_scores), key=lambda x: x[1], reverse=True)
    top_20_relations = scored_relations[:20]
    return [(rel["relatedFromIdName"], (rel.get('additionalRelationLabel', '') or '').replace('_', ' '), rel["relatedIdName"]) for rel, _ in top_20_relations]

def get_umls_keys(query):
    umls_res = {}
    entities = [ent.text.lower() for ent in nlp(query).ents]

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
            embeddings = bert.batch_encode(relation_texts)
            query_embedding = embeddings[0]
            relation_embeddings = embeddings[1:]
            rels = semantic_matching(query_embedding, relation_embeddings, relations)
        else:
            rels = []

        umls_res[cui] = {"name": name, "definition": definition, "rels": rels}

    context_parts = []
    for k, v in umls_res.items():
        rels_text = "\n".join([f"({rel[0]},{rel[1]},{rel[2]})" for rel in v["rels"]])
        context_parts.append(f"Name: {v['name']}\nDefinition: {v['definition']}\nRelations: \n{rels_text}")
    return "\n\n".join(context_parts)
