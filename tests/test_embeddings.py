from agent_memory.embeddings import HashingEmbedder


def test_unit_norm():
    emb = HashingEmbedder()
    vecs = emb.embed(["hello world", "a completely different sentence"])
    norms = (vecs**2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 1e-5 for n in norms)


def test_deterministic_across_instances():
    a = HashingEmbedder().embed(["bookings are stored in UTC"])[0]
    b = HashingEmbedder().embed(["bookings are stored in UTC"])[0]
    assert (a == b).all()


def test_lexical_similarity_orders_correctly():
    emb = HashingEmbedder()
    query = emb.embed(["how is the database connection configured"])[0]
    related = emb.embed(["the app connects to a PostgreSQL database"])[0]
    unrelated = emb.embed(["the hero image uses a pink gradient"])[0]
    assert float(query @ related) > float(query @ unrelated)
