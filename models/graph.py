from neo4j import GraphDatabase

class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _execute_query(self, query, parameters):
        result, _, _ = self.driver.execute_query(query, **parameters, database_="neo4j")
        return [record.data() for record in result]

    def search_cui(self, query_text):
        queries = [
            ("MATCH (c:Concept) WHERE c.STR = $query RETURN c.CUI AS cui, c.STR AS name", {"query": query_text}),
            ("MATCH (c:Concept) WHERE c.STR CONTAINS $query RETURN c.CUI AS cui, c.STR AS name", {"query": query_text}),
            ("MATCH (c:Concept) WHERE c.STR_extension CONTAINS $query RETURN c.CUI AS cui, c.STR AS name", {"query": query_text})
        ]
        for query, params in queries:
            results = self._execute_query(query, params)
            if results:
                return results
        return []

    def get_definitions(self, cui):
        result = self._execute_query("MATCH (c:Concept) WHERE c.CUI = $cui RETURN coalesce(c.DEF, 'No definition found.') AS definition", {"cui": cui})
        return result[0]['definition'] if result else "No definition found."

    def get_relations(self, cui):
        query = """
            MATCH (c1:Concept)-[r:RELATED_TO]->(c2:Concept)
            WHERE c1.CUI = $cui
            RETURN c1.STR AS relatedFromIdName, r.RELA AS additionalRelationLabel, c2.STR AS relatedIdName LIMIT 100
        """
        return self._execute_query(query, {"cui": cui})
    
    
    