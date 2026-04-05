"""
Yantrix Semantic Memory v2
Vector-based semantic search, memory graphs, intelligent retrieval
"""

import os, uuid, hashlib, logging
from typing import Optional, List
from datetime import datetime, timezone, timedelta

import uvicorn
import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
from pgvector.asyncpg import register_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Yantrix Semantic Memory v2", version="2.0.0")

# ─── Configuration ────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

pool = None

@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    # Register pgvector types
    async with pool.acquire() as conn:
        await register_vector(conn)
    logger.info("[Memory v2] ✓ Database connected, pgvector registered")

@app.on_event("shutdown")
async def shutdown():
    await pool.close()

# ─── Models ───────────────────────────────────────────────────────────────

class MemoryStore(BaseModel):
    agent_id: str
    content: str
    memory_type: str = "long_term"  # short_term, long_term, episodic, shared
    importance: int = 5
    tags: List[str] = []
    category: Optional[str] = None
    access_level: str = "private"
    expires_in_hours: Optional[int] = None  # Auto-set expires_at
    metadata: dict = {}

class MemorySearch(BaseModel):
    agent_id: str
    query: str
    memory_types: List[str] = ["long_term", "episodic", "shared"]
    limit: int = 10
    min_relevance: float = 0.7
    include_related: bool = True

class MemoryLink(BaseModel):
    from_memory_id: str
    to_memory_id: str
    relationship_type: str  # related_to, contradicts, expands_on, etc.
    strength: float = 0.5

class SessionContext(BaseModel):
    agent_id: str
    session_name: Optional[str] = None
    context: dict = {}
    expires_in_hours: int = 24

# ─── Embedding Functions ──────────────────────────────────────────────────

async def get_embedding(text: str, use_cache: bool = True) -> List[float]:
    """Get embedding with caching"""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    
    # Check cache
    if use_cache:
        async with pool.acquire() as conn:
            cached = await conn.fetchrow(
                "SELECT embedding FROM embedding_cache WHERE text_hash = $1 AND model = $2",
                text_hash, EMBEDDING_MODEL
            )
            if cached:
                await conn.execute(
                    "UPDATE embedding_cache SET last_used = NOW(), use_count = use_count + 1 WHERE text_hash = $1",
                    text_hash
                )
                return cached["embedding"]
    
    # Call OpenAI API
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"input": text, "model": EMBEDDING_MODEL},
                timeout=10.0
            )
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            
            # Cache it
            if use_cache:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO embedding_cache (text_hash, text, embedding, model)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (text_hash) DO UPDATE
                        SET last_used = NOW(), use_count = embedding_cache.use_count + 1
                    """, text_hash, text[:1000], embedding, EMBEDDING_MODEL)
            
            return embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(500, detail="Embedding generation failed")

async def generate_summary(text: str) -> str:
    """Generate short summary using OpenAI"""
    if len(text) < 100:
        return text
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Summarize in 1-2 sentences."},
                        {"role": "user", "content": text}
                    ],
                    "max_tokens": 100
                },
                timeout=10.0
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return text[:200]  # Fallback: first 200 chars

# ─── API Endpoints ────────────────────────────────────────────────────────

@app.post("/v2/memory/store")
async def store_memory(memory: MemoryStore, background_tasks: BackgroundTasks):
    """Store memory with auto-embedding"""
    
    # Generate embedding
    embedding = await get_embedding(memory.content)
    
    # Generate summary for long content
    summary = None
    if len(memory.content) > 200:
        background_tasks.add_task(generate_summary, memory.content)
        summary = memory.content[:200] + "..."
    
    # Calculate expiry
    expires_at = None
    if memory.expires_in_hours:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=memory.expires_in_hours)
    elif memory.memory_type == "short_term" and not memory.expires_in_hours:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    
    # Insert memory
    async with pool.acquire() as conn:
        memory_id = await conn.fetchval("""
            INSERT INTO memory_entries 
            (agent_id, content, summary, embedding, memory_type, importance, tags, category, 
             access_level, metadata, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
        """, 
            memory.agent_id, memory.content, summary, embedding, memory.memory_type,
            memory.importance, memory.tags, memory.category, memory.access_level,
            memory.metadata, expires_at
        )
    
    return {
        "success": True,
        "memory_id": str(memory_id),
        "embedding_dim": len(embedding)
    }

@app.post("/v2/memory/search")
async def search_memories(search: MemorySearch):
    """Semantic + keyword hybrid search"""
    
    # Get query embedding
    query_embedding = await get_embedding(search.query)
    
    # Hybrid search: vector similarity + keyword matching + time decay
    async with pool.acquire() as conn:
        memories = await conn.fetch("""
            SELECT 
                id,
                agent_id,
                content,
                summary,
                memory_type,
                importance,
                tags,
                category,
                created_at,
                accessed_at,
                access_count,
                -- Vector similarity (cosine distance)
                1 - (embedding <=> $1) as similarity,
                -- Keyword boost
                CASE 
                    WHEN content ILIKE '%' || $2 || '%' THEN 0.2
                    ELSE 0.0
                END as keyword_boost,
                -- Time decay
                importance * (1 - LEAST(EXTRACT(EPOCH FROM (NOW() - created_at)) / (86400 * 30), 0.5)) as time_score
            FROM memory_entries
            WHERE agent_id = $3
            AND memory_type = ANY($4)
            AND status = 'active'
            AND (expires_at IS NULL OR expires_at > NOW())
            AND embedding IS NOT NULL
            ORDER BY (similarity + keyword_boost + (time_score / 10)) DESC
            LIMIT $5
        """, 
            query_embedding, search.query, search.agent_id, 
            search.memory_types, search.limit
        )
        
        # Filter by minimum relevance
        results = [
            dict(m) for m in memories 
            if m["similarity"] >= search.min_relevance
        ]
        
        # Optionally include related memories
        if search.include_related and results:
            memory_ids = [r["id"] for r in results]
            related = await conn.fetch("""
                SELECT DISTINCT me.*
                FROM memory_links ml
                JOIN memory_entries me ON me.id = ml.to_memory_id
                WHERE ml.from_memory_id = ANY($1)
                AND me.status = 'active'
                AND ml.relationship_type IN ('related_to', 'expands_on')
                LIMIT 5
            """, memory_ids)
            
            for r in related:
                results.append(dict(r))
    
    return {
        "query": search.query,
        "results": results,
        "count": len(results)
    }

@app.get("/v2/memory/{memory_id}")
async def get_memory(memory_id: str):
    """Retrieve specific memory"""
    async with pool.acquire() as conn:
        memory = await conn.fetchrow("""
            SELECT * FROM memory_entries
            WHERE id = $1 AND status = 'active'
        """, uuid.UUID(memory_id))
        
        if not memory:
            raise HTTPException(404, detail="Memory not found")
        
        # Track access
        await conn.execute("""
            UPDATE memory_entries
            SET accessed_at = NOW(), access_count = access_count + 1
            WHERE id = $1
        """, uuid.UUID(memory_id))
        
        return dict(memory)

@app.post("/v2/memory/link")
async def create_link(link: MemoryLink):
    """Create relationship between memories"""
    async with pool.acquire() as conn:
        link_id = await conn.fetchval("""
            INSERT INTO memory_links (from_memory_id, to_memory_id, relationship_type, strength)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (from_memory_id, to_memory_id, relationship_type) 
            DO UPDATE SET strength = $4
            RETURNING id
        """, 
            uuid.UUID(link.from_memory_id), 
            uuid.UUID(link.to_memory_id),
            link.relationship_type, 
            link.strength
        )
    
    return {"success": True, "link_id": str(link_id)}

@app.get("/v2/memory/graph/{agent_id}")
async def get_memory_graph(agent_id: str, limit: int = 50):
    """Get memory graph for agent"""
    async with pool.acquire() as conn:
        # Get memories
        memories = await conn.fetch("""
            SELECT id, content, summary, memory_type, importance, tags, created_at
            FROM memory_entries
            WHERE agent_id = $1 AND status = 'active'
            ORDER BY importance DESC, created_at DESC
            LIMIT $2
        """, agent_id, limit)
        
        memory_ids = [m["id"] for m in memories]
        
        # Get links between them
        links = await conn.fetch("""
            SELECT from_memory_id, to_memory_id, relationship_type, strength
            FROM memory_links
            WHERE from_memory_id = ANY($1) AND to_memory_id = ANY($1)
        """, memory_ids)
        
        return {
            "agent_id": agent_id,
            "memories": [dict(m) for m in memories],
            "links": [dict(l) for l in links]
        }

@app.post("/v2/memory/consolidate")
async def consolidate_memories(agent_id: str, similarity_threshold: float = 0.9):
    """Find and merge similar memories"""
    async with pool.acquire() as conn:
        # Find similar memory pairs
        similar = await conn.fetch("""
            SELECT 
                m1.id as id1, 
                m2.id as id2,
                m1.content as content1,
                m2.content as content2,
                1 - (m1.embedding <=> m2.embedding) as similarity
            FROM memory_entries m1
            JOIN memory_entries m2 ON m1.agent_id = m2.agent_id AND m1.id < m2.id
            WHERE m1.agent_id = $1
            AND m1.status = 'active' AND m2.status = 'active'
            AND 1 - (m1.embedding <=> m2.embedding) > $2
            ORDER BY similarity DESC
            LIMIT 10
        """, agent_id, similarity_threshold)
        
        consolidated = []
        for pair in similar:
            # Mark older as superseded
            await conn.execute("""
                UPDATE memory_entries
                SET status = 'superseded', superseded_by = $2
                WHERE id = $1
            """, pair["id2"], pair["id1"])
            
            consolidated.append({
                "kept": str(pair["id1"]),
                "superseded": str(pair["id2"]),
                "similarity": float(pair["similarity"])
            })
    
    return {
        "agent_id": agent_id,
        "consolidated": consolidated,
        "count": len(consolidated)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "semantic-memory-v2"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
