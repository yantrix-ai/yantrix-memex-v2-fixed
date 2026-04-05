-- ═══════════════════════════════════════════════════════════════════════════
-- Semantic Memory v2 - Database Schema
-- Vector embeddings, memory graphs, intelligent retrieval
-- ═══════════════════════════════════════════════════════════════════════════

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Memory Entries ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS memory_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Ownership
    agent_id TEXT NOT NULL,
    organization_id TEXT,
    
    -- Content
    content TEXT NOT NULL,
    summary TEXT, -- Auto-generated short summary
    embedding VECTOR(1536), -- OpenAI ada-002 / text-embedding-3-small
    
    -- Classification
    memory_type TEXT NOT NULL DEFAULT 'long_term', -- short_term, long_term, episodic, shared
    importance INTEGER DEFAULT 5 CHECK (importance BETWEEN 1 AND 10),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    
    -- Organization
    tags TEXT[],
    category TEXT,
    source TEXT, -- which agent/system created this
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Access control
    access_level TEXT DEFAULT 'private', -- private, shared, public
    shared_with TEXT[], -- agent_ids with access
    
    -- Lifecycle
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    expires_at TIMESTAMPTZ, -- NULL = never expires
    
    -- Status
    status TEXT DEFAULT 'active', -- active, archived, superseded, deleted
    superseded_by UUID,
    
    CHECK (memory_type IN ('short_term', 'long_term', 'episodic', 'shared')),
    CHECK (access_level IN ('private', 'shared', 'public')),
    CHECK (status IN ('active', 'archived', 'superseded', 'deleted'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS me_agent_idx ON memory_entries(agent_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS me_type_idx ON memory_entries(memory_type, status);
CREATE INDEX IF NOT EXISTS me_tags_idx ON memory_entries USING GIN(tags);
CREATE INDEX IF NOT EXISTS me_importance_idx ON memory_entries(importance DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS me_expires_idx ON memory_entries(expires_at) WHERE expires_at IS NOT NULL;

-- Vector similarity index (HNSW for fast ANN search)
CREATE INDEX IF NOT EXISTS me_embedding_idx ON memory_entries 
USING hnsw (embedding vector_cosine_ops)
WHERE embedding IS NOT NULL AND status = 'active';

-- ─── Memory Relationships ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS memory_links (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    from_memory_id UUID NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    to_memory_id UUID NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    
    -- Link type
    relationship_type TEXT NOT NULL, -- related_to, contradicts, expands_on, caused_by, leads_to, evidence_for
    strength FLOAT DEFAULT 0.5 CHECK (strength BETWEEN 0 AND 1),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT, -- agent_id
    
    -- Prevent duplicate links
    UNIQUE(from_memory_id, to_memory_id, relationship_type),
    CHECK (from_memory_id != to_memory_id),
    CHECK (relationship_type IN ('related_to', 'contradicts', 'expands_on', 'caused_by', 'leads_to', 'evidence_for', 'replaces'))
);

CREATE INDEX IF NOT EXISTS ml_from_idx ON memory_links(from_memory_id, relationship_type);
CREATE INDEX IF NOT EXISTS ml_to_idx ON memory_links(to_memory_id, relationship_type);
CREATE INDEX IF NOT EXISTS ml_strength_idx ON memory_links(strength DESC);

-- ─── Memory Sessions ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS memory_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    agent_id TEXT NOT NULL,
    session_name TEXT,
    
    -- Session context
    context JSONB DEFAULT '{}',
    memory_ids UUID[], -- memories active in this session
    
    -- Lifecycle
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    status TEXT DEFAULT 'active', -- active, completed, expired
    
    CHECK (status IN ('active', 'completed', 'expired'))
);

CREATE INDEX IF NOT EXISTS ms_agent_idx ON memory_sessions(agent_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS ms_expires_idx ON memory_sessions(expires_at) WHERE expires_at IS NOT NULL;

-- ─── Memory Access Log ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS memory_access_log (
    id BIGSERIAL PRIMARY KEY,
    
    memory_id UUID NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL,
    
    -- Access details
    access_type TEXT NOT NULL, -- retrieve, search, update, delete
    relevance_score FLOAT, -- how relevant was this memory to the query
    
    -- Query context
    query_text TEXT,
    query_embedding VECTOR(1536),
    
    -- Metadata
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    session_id UUID REFERENCES memory_sessions(id),
    
    CHECK (access_type IN ('retrieve', 'search', 'update', 'delete'))
);

CREATE INDEX IF NOT EXISTS mal_memory_idx ON memory_access_log(memory_id, accessed_at DESC);
CREATE INDEX IF NOT EXISTS mal_agent_idx ON memory_access_log(agent_id, accessed_at DESC);

-- ─── Embedding Cache ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS embedding_cache (
    id BIGSERIAL PRIMARY KEY,
    
    text_hash TEXT UNIQUE NOT NULL, -- SHA256 hash of text
    text TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    model TEXT NOT NULL, -- openai-ada-002, text-embedding-3-small, etc.
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used TIMESTAMPTZ DEFAULT NOW(),
    use_count INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS ec_hash_idx ON embedding_cache(text_hash);
CREATE INDEX IF NOT EXISTS ec_model_idx ON embedding_cache(model, last_used DESC);

-- ─── Helper Functions ─────────────────────────────────────────────────────

-- Update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER memory_entries_updated_at
BEFORE UPDATE ON memory_entries
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Auto-expire short-term memories
CREATE OR REPLACE FUNCTION auto_expire_short_term()
RETURNS void AS $$
BEGIN
    UPDATE memory_entries
    SET status = 'archived'
    WHERE memory_type = 'short_term'
    AND expires_at < NOW()
    AND status = 'active';
END;
$$ LANGUAGE plpgsql;

-- Track memory access
CREATE OR REPLACE FUNCTION track_memory_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.accessed_at = NOW();
    NEW.access_count = NEW.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ─── Views ────────────────────────────────────────────────────────────────

-- Active memories with computed relevance decay
CREATE OR REPLACE VIEW active_memories AS
SELECT 
    id,
    agent_id,
    content,
    summary,
    memory_type,
    importance,
    tags,
    category,
    access_level,
    created_at,
    accessed_at,
    access_count,
    -- Time decay: importance * (1 - age_factor)
    importance * (1 - LEAST(EXTRACT(EPOCH FROM (NOW() - created_at)) / (86400 * 30), 0.5)) as relevance_score,
    -- Recency: memories accessed recently are more relevant
    importance * (1 + (access_count * 0.1)) * 
        CASE 
            WHEN accessed_at > NOW() - INTERVAL '1 day' THEN 1.5
            WHEN accessed_at > NOW() - INTERVAL '7 days' THEN 1.2
            ELSE 1.0
        END as boosted_score
FROM memory_entries
WHERE status = 'active'
AND (expires_at IS NULL OR expires_at > NOW());

-- Memory graph statistics
CREATE OR REPLACE VIEW memory_graph_stats AS
SELECT 
    agent_id,
    COUNT(DISTINCT id) as total_memories,
    COUNT(DISTINCT id) FILTER (WHERE memory_type = 'short_term') as short_term_count,
    COUNT(DISTINCT id) FILTER (WHERE memory_type = 'long_term') as long_term_count,
    COUNT(DISTINCT id) FILTER (WHERE memory_type = 'episodic') as episodic_count,
    AVG(importance) as avg_importance,
    AVG(access_count) as avg_access_count,
    COUNT(DISTINCT ml.id) as total_links
FROM memory_entries me
LEFT JOIN memory_links ml ON ml.from_memory_id = me.id OR ml.to_memory_id = me.id
WHERE me.status = 'active'
GROUP BY agent_id;

COMMENT ON TABLE memory_entries IS 'Semantic memory storage with vector embeddings';
COMMENT ON TABLE memory_links IS 'Graph relationships between memories';
COMMENT ON TABLE memory_sessions IS 'Short-term session contexts';
COMMENT ON TABLE memory_access_log IS 'Memory retrieval and usage tracking';
COMMENT ON TABLE embedding_cache IS 'Cached embeddings to reduce API calls';
