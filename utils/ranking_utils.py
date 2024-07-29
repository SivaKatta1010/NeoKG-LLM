from sklearn.metrics.pairwise import cosine_similarity

def get_similarity(query_vec, rel_vec):
    return cosine_similarity(query_vec, rel_vec)