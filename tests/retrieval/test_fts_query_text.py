from memory_base.retrieval import search


def test_drops_unquoted_stopwords():
    assert search.fts_query_text("What awards did ZX Bank win in 2023?") == (
        "awards ZX Bank win 2023?"
    )


def test_preserves_quoted_phrases_verbatim():
    assert search.fts_query_text('What "the rise of AI" did change?') == '"the rise of AI" change?'


def test_preserves_punctuation_on_surviving_tokens():
    assert search.fts_query_text("In? PostgreSQL, the 2023?") == "PostgreSQL, 2023?"


def test_returns_original_query_when_every_token_is_a_stopword():
    query = "What did we do in the?"

    assert search.fts_query_text(query) == query


def test_leaves_query_without_stopwords_unchanged():
    query = "ZX Bank awards 2023?"

    assert search.fts_query_text(query) == query
