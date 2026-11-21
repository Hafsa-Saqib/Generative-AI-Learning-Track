CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_type TEXT NOT NULL CHECK (document_type IN ('sla_contract', 'policy_doc', 'resolution_history')),
    chunk_index INT NOT NULL,
    raw_text TEXT NOT NULL,
    contextualized_text TEXT NOT NULL,
    embedding vector(1024),
    content_hash TEXT NOT NULL,
    embedding_model_version TEXT NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);