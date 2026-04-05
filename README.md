# Yantrix Semantic Memory v2

Vector-based semantic search, memory graphs, and intelligent retrieval for AI agents.

## Features

✅ **Semantic Search** - Vector embeddings with pgvector  
✅ **Memory Types** - Short-term, long-term, episodic, shared  
✅ **Memory Graphs** - Relationship links between memories  
✅ **Hybrid Search** - Vector similarity + keyword + time decay  
✅ **Auto-Consolidation** - Merge similar memories  
✅ **Access Control** - Private, shared, public memories  
✅ **Embedding Cache** - Reduce API costs  
✅ **Importance Scoring** - 1-10 relevance weights  

---

## Architecture

**Database:** PostgreSQL with pgvector extension  
**Embeddings:** OpenAI text-embedding-3-small (1536 dimensions)  
**Search:** HNSW index for fast ANN (approximate nearest neighbor)  
**API:** FastAPI with async/await  

---

## Deployment

### 1. Create GitHub Repo

```bash
cd C:\Users\Praveen
mkdir semantic-memory-v2
cd semantic-memory-v2

# Copy these files:
# - semantic_memory_v2_api.py (rename to main.py)
# - semantic_memory_v2_requirements.txt (rename to requirements.txt)
# - semantic_memory_v2_schema.sql
# - semantic_memory_v2_railway.json (rename to railway.json)

git init
git add .
git commit -m "feat: Semantic Memory v2 initial implementation"

# Create repo and push
gh repo create yantrix-ai/semantic-memory-v2 --public --source=. --push
```

### 2. Deploy to Railway

1. **Railway** → New Project → Deploy from GitHub
2. Select `yantrix-ai/semantic-memory-v2`
3. **Add Postgres** database
4. **Enable pgvector:**
   - Click Postgres → Settings → Extensions
   - Enable `vector` extension
5. **Set Variables:**
   - `OPENAI_API_KEY` = `<your_openai_key>`
   - `EMBEDDING_MODEL` = `text-embedding-3-small`
6. **Generate Domain:** `memory-v2.yantrix.ai`

### 3. Run Database Migration

**Railway → Postgres → Data → Query:**

Paste and run `semantic_memory_v2_schema.sql`

### 4. Test

```powershell
# Health check
Invoke-WebRequest https://memory-v2.yantrix.ai/health

# Store memory
Invoke-WebRequest https://memory-v2.yantrix.ai/v2/memory/store `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"agent_id":"test-agent","content":"Paris is the capital of France","tags":["geography","facts"]}'

# Search memories
Invoke-WebRequest https://memory-v2.yantrix.ai/v2/memory/search `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"agent_id":"test-agent","query":"What is the capital of France?"}'
```

---

## API Reference

### **POST /v2/memory/store**
Store memory with auto-embedding

**Request:**
```json
{
  "agent_id": "agent-123",
  "content": "The meeting was productive. We decided to launch Q3.",
  "memory_type": "episodic",
  "importance": 8,
  "tags": ["meeting", "planning"],
  "category": "work",
  "access_level": "private",
  "expires_in_hours": 720,
  "metadata": {"project": "alpha"}
}
```

**Response:**
```json
{
  "success": true,
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "embedding_dim": 1536
}
```

---

### **POST /v2/memory/search**
Semantic + keyword hybrid search

**Request:**
```json
{
  "agent_id": "agent-123",
  "query": "Tell me about our Q3 plans",
  "memory_types": ["long_term", "episodic"],
  "limit": 10,
  "min_relevance": 0.7,
  "include_related": true
}
```

**Response:**
```json
{
  "query": "Tell me about our Q3 plans",
  "results": [
    {
      "id": "550e8400-...",
      "content": "The meeting was productive...",
      "summary": "Q3 launch decision",
      "memory_type": "episodic",
      "importance": 8,
      "tags": ["meeting", "planning"],
      "similarity": 0.89,
      "created_at": "2026-04-05T10:30:00Z"
    }
  ],
  "count": 1
}
```

---

### **GET /v2/memory/{memory_id}**
Retrieve specific memory

**Response:**
```json
{
  "id": "550e8400-...",
  "agent_id": "agent-123",
  "content": "...",
  "memory_type": "long_term",
  "importance": 8,
  "tags": [...],
  "created_at": "...",
  "access_count": 5
}
```

---

### **POST /v2/memory/link**
Create relationship between memories

**Request:**
```json
{
  "from_memory_id": "550e8400-...",
  "to_memory_id": "660e9500-...",
  "relationship_type": "expands_on",
  "strength": 0.8
}
```

---

### **GET /v2/memory/graph/{agent_id}**
Get memory graph

**Response:**
```json
{
  "agent_id": "agent-123",
  "memories": [{...}, {...}],
  "links": [
    {
      "from_memory_id": "...",
      "to_memory_id": "...",
      "relationship_type": "related_to",
      "strength": 0.7
    }
  ]
}
```

---

### **POST /v2/memory/consolidate**
Find and merge similar memories

**Request:**
```json
{
  "agent_id": "agent-123",
  "similarity_threshold": 0.9
}
```

**Response:**
```json
{
  "agent_id": "agent-123",
  "consolidated": [
    {
      "kept": "550e8400-...",
      "superseded": "660e9500-...",
      "similarity": 0.95
    }
  ],
  "count": 1
}
```

---

## Memory Types

- **short_term** - Auto-expires in 24h (session context)
- **long_term** - Persistent knowledge
- **episodic** - Event sequences with timestamps
- **shared** - Agent-to-agent memory pools

---

## Relationship Types

- `related_to` - General association
- `contradicts` - Conflicting information
- `expands_on` - Additional details
- `caused_by` - Causal link
- `leads_to` - Consequence
- `evidence_for` - Supporting evidence
- `replaces` - Supersedes older memory

---

## Access Levels

- **private** - Only the creating agent
- **shared** - Specific agents (via `shared_with` array)
- **public** - All agents

---

## Cost Optimization

**Embedding Cache:**
- SHA256 hash-based deduplication
- Reduces OpenAI API costs by ~80%
- Automatic cache hits tracking

**Efficient Search:**
- HNSW index for sub-millisecond vector search
- Hybrid scoring reduces over-retrieval
- Time decay prevents stale results

---

## Integration Example

```python
import httpx

async def store_and_retrieve():
    # Store
    await httpx.post("https://memory-v2.yantrix.ai/v2/memory/store", json={
        "agent_id": "my-agent",
        "content": "User prefers morning meetings",
        "memory_type": "long_term",
        "importance": 7,
        "tags": ["preferences", "scheduling"]
    })
    
    # Search
    results = await httpx.post("https://memory-v2.yantrix.ai/v2/memory/search", json={
        "agent_id": "my-agent",
        "query": "When should I schedule meetings?",
        "limit": 5
    })
    
    print(results.json())
```

---

## Files

- `semantic_memory_v2_schema.sql` - Database schema
- `semantic_memory_v2_api.py` - FastAPI service (rename to main.py)
- `semantic_memory_v2_requirements.txt` - Dependencies (rename to requirements.txt)
- `semantic_memory_v2_railway.json` - Railway config (rename to railway.json)
