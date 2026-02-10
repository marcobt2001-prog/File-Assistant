# FileAssistant — Updated Phase Plan
## Phase 2: Search & Intelligence

> **Key Change**: Search is now a first-class feature alongside organization. The system doesn't just file things away — it makes your entire file collection instantly searchable by description, meaning, and context.

---

## What's Changed From the Original Plan

The original Phase 2 ("Learning & Autonomy") focused on embeddings + vector store primarily as a *classification booster*. This updated plan elevates search to a core user-facing feature and restructures the phase into three sub-phases for manageable implementation.

**Original Phase 2 scope** (now redistributed):
- ~~Embedding generation + vector store~~ → Phase 2A (search foundation)
- ~~Similarity-based classification boost~~ → Phase 2B (integration)
- ~~Confidence scoring~~ → Phase 2C
- ~~Automatic action for high-confidence decisions~~ → Phase 2C
- ~~Learning from user corrections~~ → Phase 2C
- ~~Multiple inbox folders~~ → Phase 2B
- ~~Undo capability~~ → Phase 2B
- ~~Basic system tray UI~~ → Deferred to Phase 4

**New additions**:
- Bulk file indexer (scan existing files)
- Search CLI command with natural language queries
- LLM-powered query understanding and re-ranking
- Search result snippets and relevance scoring

---

## Phase 2A: Search Foundation (Week 6-7)

**Goal**: Every processed file gets an embedding; users can search by description.

### 2A.1 — Embedding Generator Component

**New file**: `src/fileassistant/embeddings/generator.py`

Responsibilities:
- Take extracted text (from the existing analyzer) and produce a vector embedding
- Use `sentence-transformers` with `all-MiniLM-L6-v2` (runs locally, fast, ~80MB model)
- Handle text chunking for long documents (chunk at ~512 tokens with overlap)
- Generate a single "summary embedding" per file (average of chunk embeddings)
- Cache embeddings to avoid recomputation

```
Analyzer (existing) → extracted text → Embedding Generator → vector
```

**Key design decisions**:
- One embedding per file for search (not per-chunk initially — keeps it simple)
- If a file's text exceeds the model's token limit, chunk and average
- Store the embedding ID in the existing `files` table (`embedding_id` column already exists)

**Dependencies to install**:
```bash
pip install sentence-transformers
```

**Implementation tasks**:
- [ ] Create `EmbeddingGenerator` class with `generate(text: str) -> list[float]` method
- [ ] Add text chunking utility (split on paragraphs/sentences, ~512 token chunks)
- [ ] Add embedding model download/initialization (first-run downloads the model)
- [ ] Integrate into analyzer pipeline: after text extraction, generate embedding
- [ ] Add config options: `embedding_model`, `chunk_size`, `chunk_overlap`

---

### 2A.2 — Index Manager Component

**New file**: `src/fileassistant/search/index_manager.py`

Responsibilities:
- Store embeddings in ChromaDB with rich metadata
- Handle CRUD operations (add, update, delete embeddings)
- Store metadata alongside vectors for filtering and display

**ChromaDB document structure**:
```python
collection.add(
    ids=["file_123"],
    embeddings=[[0.1, 0.2, ...]],
    documents=["First 1000 chars of extracted text..."],  # for snippet display
    metadatas=[{
        "file_path": "/Users/marco/Documents/receipts/homedepot-2025.pdf",
        "filename": "homedepot-2025.pdf",
        "extension": ".pdf",
        "file_type": "document",
        "tags": "receipt,home-depot,renovation",  # comma-separated for filtering
        "content_summary": "Home Depot receipt for lumber and hardware...",
        "created_at": "2025-01-15T10:30:00",
        "modified_at": "2025-01-15T10:30:00",
        "indexed_at": "2025-02-09T14:00:00",
        "size_bytes": 45000,
        "source_folder": "Downloads"
    }]
)
```

**Implementation tasks**:
- [ ] Create `IndexManager` class wrapping ChromaDB operations
- [ ] Method: `index_file(file_record, text, embedding)` — add/update a file in the index
- [ ] Method: `remove_file(file_id)` — remove from index (for deleted/moved files)
- [ ] Method: `get_indexed_count()` — stats for CLI display
- [ ] Method: `is_indexed(file_id, file_hash)` — skip re-indexing unchanged files
- [ ] Configure ChromaDB persistent storage path (default: `~/.fileassistant/chromadb/`)
- [ ] Handle collection creation on first run

---

### 2A.3 — Bulk Indexer Command

**New CLI command**: `fileassistant index <path> [--recursive] [--force]`

This is critical for search to be useful on day one. Users need to index their existing files, not just new ones flowing through the watcher.

**Behavior**:
```
$ fileassistant index ~/Documents

Scanning ~/Documents...
Found 1,247 files (892 supported types)
Already indexed: 0
To process: 892

Processing... [████████████████░░░░░░░░░░░░] 56% (499/892)
  ✓ 499 indexed
  ⚠ 12 skipped (unsupported type)
  ✗ 3 errors (see log)

Indexing complete. 499 files indexed in 4m 32s.
Search is ready! Try: fileassistant search "tax documents from 2024"
```

**Implementation tasks**:
- [ ] Create `fileassistant index` CLI command
- [ ] Recursive directory walker with file type filtering
- [ ] Progress bar with Rich library (already in the project)
- [ ] Skip already-indexed files (by hash comparison)
- [ ] `--force` flag to re-index everything
- [ ] `--dry-run` flag to show what would be indexed without doing it
- [ ] Batch processing with error handling (don't stop on one bad file)
- [ ] Summary statistics on completion
- [ ] Register indexed files in the SQLite `files` table if not already tracked

---

### 2A.4 — Search Engine Component

**New file**: `src/fileassistant/search/engine.py`

The core search experience. Takes a natural language query and returns ranked results.

**Search pipeline**:
```
User query
    │
    ▼
┌─────────────────────┐
│ 1. Embed the query  │  ← Same embedding model as indexing
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 2. ChromaDB vector  │  ← Returns top-K candidates (default: 20)
│    similarity search│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 3. Metadata filter  │  ← Optional: filter by type, date, tags
│    (post-retrieval)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 4. LLM re-ranking   │  ← Optional: use Ollama to re-rank by relevance
│    (if enabled)      │     to the original query
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 5. Format & return  │  ← Top N results with snippets
└─────────────────────┘
```

**Search result model**:
```python
@dataclass
class SearchResult:
    file_path: str
    filename: str
    relevance_score: float      # 0.0 to 1.0
    content_snippet: str        # First ~200 chars of relevant content
    tags: list[str]
    file_type: str
    modified_at: datetime
    match_reason: str           # Brief explanation of why this matched
```

**Implementation tasks**:
- [ ] Create `SearchEngine` class
- [ ] Method: `search(query: str, filters: dict = None, limit: int = 10) -> list[SearchResult]`
- [ ] Query embedding generation
- [ ] ChromaDB similarity search with configurable `n_results`
- [ ] Metadata-based post-filtering (by extension, date range, tags)
- [ ] Result formatting with content snippets
- [ ] Relevance score normalization (ChromaDB distance → 0-1 score)

---

### 2A.5 — Search CLI Command

**New CLI command**: `fileassistant search <query> [options]`

**Basic usage**:
```
$ fileassistant search "that receipt from Home Depot"

Found 3 results:

1. [0.94] homedepot-receipt-2025-01-15.pdf
   📁 ~/Documents/Finances/Receipts/
   🏷️  receipt, home-depot, renovation
   📝 "Home Depot receipt - lumber, screws, wood stain - $247.83"

2. [0.81] home-renovation-expenses.xlsx
   📁 ~/Documents/Projects/Home Renovation/
   🏷️  spreadsheet, renovation, budget
   📝 "Expense tracking spreadsheet with Home Depot purchases..."

3. [0.72] contractor-invoice-jan.pdf
   📁 ~/Documents/Finances/Invoices/
   🏷️  invoice, renovation, contractor
   📝 "Invoice from Bob's Carpentry referencing HD materials..."
```

**With filters**:
```
$ fileassistant search "meeting notes" --type pdf --after 2025-01-01
$ fileassistant search "code snippets" --type py,js,ts
$ fileassistant search "photos from vacation" --tag travel
```

**Implementation tasks**:
- [ ] Create `fileassistant search` CLI command with Click
- [ ] Required argument: query string
- [ ] Optional flags: `--type` (extension filter), `--after`/`--before` (date filter), `--tag` (tag filter), `--limit` (number of results, default 10)
- [ ] Rich-formatted output with color-coded relevance scores
- [ ] `--open` flag to open the top result directly
- [ ] `--json` flag for programmatic output
- [ ] Handle case where index is empty (prompt user to run `fileassistant index` first)

---

### Phase 2A Success Criteria
- [ ] Can index 500+ files in under 10 minutes
- [ ] Search returns relevant results for descriptive queries (e.g., "tax documents" finds tax-related PDFs)
- [ ] Search works on files that were indexed via bulk indexer AND files processed through the watcher
- [ ] Relevance scores are meaningful (top result is usually correct)
- [ ] Graceful handling of unsupported file types, empty files, and encoding issues

---

## Phase 2B: Integration & Improved Organization (Week 8-9)

**Goal**: Search intelligence feeds back into classification; organization pipeline matures.

### 2B.1 — Similarity-Based Classification Boost

Use the vector store to improve classification. When a new file arrives, find the most similar already-organized files and use their destinations/tags as strong hints.

**Enhanced classifier flow**:
```
New file arrives
    │
    ▼
Extract text + generate embedding (existing)
    │
    ▼
Query ChromaDB: "What existing files are most similar?"
    │
    ├── Top 3 similar files all live in Documents/Finances/Receipts/
    │   → Strong signal: this file probably goes there too
    │
    ▼
Build classifier prompt with similarity context:
    "Similar files: receipt-walmart.pdf (in Finances/Receipts/),
     receipt-amazon.pdf (in Finances/Receipts/), ..."
    │
    ▼
LLM makes decision with much better context
```

**Implementation tasks**:
- [ ] Add similarity lookup to classifier: before LLM call, find top-5 similar files
- [ ] Include similar file paths and tags in the LLM classification prompt
- [ ] Weight classification confidence based on agreement between LLM and similar files
- [ ] If similar files strongly agree on a destination, boost confidence score

---

### 2B.2 — Multiple Inbox Folders

Extend the watcher to monitor multiple configurable folders.

**Implementation tasks**:
- [ ] Update watcher to accept list of inbox folders from config
- [ ] Per-folder configuration (different processing rules per inbox)
- [ ] Default inboxes: Downloads, Desktop (configurable)
- [ ] CLI command: `fileassistant watch --add ~/Screenshots`

---

### 2B.3 — Undo Capability

Allow users to reverse any file operation.

**Implementation tasks**:
- [ ] Create `fileassistant undo` CLI command (undoes last action)
- [ ] `fileassistant undo --list` shows recent actions
- [ ] `fileassistant undo <action-id>` undoes a specific action
- [ ] Undo moves the file back and updates the index
- [ ] Actions table already exists in the schema — implement the undo logic

---

### 2B.4 — Auto-Index on Organization

When the mover places a file, automatically index it for search.

**Implementation tasks**:
- [ ] After successful file move, trigger embedding generation + ChromaDB insert
- [ ] Update existing index entry if file was already indexed (path changed)
- [ ] Ensure index stays in sync with actual file locations

---

### Phase 2B Success Criteria
- [ ] Classification accuracy improves measurably when similar files exist in the index
- [ ] Multiple inbox folders work simultaneously without conflict
- [ ] Undo works reliably for file moves
- [ ] Newly organized files are immediately searchable

---

## Phase 2C: Learning & Autonomy (Week 10-11)

**Goal**: The system gets smarter over time and can act more autonomously.

### 2C.1 — Confidence Scoring Improvements

Refine confidence scoring using multiple signals.

**Confidence formula**:
```
final_confidence = weighted_average(
    llm_confidence,           # What the LLM reports (weight: 0.4)
    similarity_confidence,     # How similar to known files (weight: 0.4)
    rule_match_confidence      # Whether explicit rules match (weight: 0.2)
)
```

**Implementation tasks**:
- [ ] Implement multi-signal confidence scoring
- [ ] Calibrate weights through testing
- [ ] Add confidence thresholds to config (already defined in schema)
- [ ] Different behaviors per confidence tier (auto-act, show, queue)

---

### 2C.2 — Automatic Actions for High-Confidence Decisions

When the system is very confident, just do it.

**Implementation tasks**:
- [ ] Implement confidence-threshold-based auto-action
- [ ] High confidence (>0.9): act automatically, log silently
- [ ] Medium confidence (0.6-0.9): act automatically, flag for review
- [ ] Low confidence (<0.6): queue for user decision
- [ ] All automatic actions are undoable
- [ ] CLI command: `fileassistant review` — show items needing attention

---

### 2C.3 — Learning From Corrections

When the user rejects a classification or corrects it, feed that back into the system.

**Implementation tasks**:
- [ ] Record corrections in the `corrections` table (schema exists)
- [ ] When user rejects and provides correct destination, store the correction
- [ ] Use corrections as negative/positive examples in future classifier prompts
- [ ] Update the file's index entry with corrected metadata
- [ ] CLI: `fileassistant correct <file> --destination <path>` for manual corrections

---

### Phase 2C Success Criteria
- [ ] High-confidence automatic actions are correct 95%+ of the time
- [ ] Classification accuracy improves measurably after user corrections
- [ ] User can review and manage the queue of uncertain classifications
- [ ] System becomes noticeably "smarter" after 50+ corrections

---

## Updated Architecture Diagram

```
                          ┌──────────────────────────────┐
                          │        USER INTERFACE         │
                          │                               │
                          │  CLI Commands:                │
                          │  • fileassistant watch        │
                          │  • fileassistant process      │
                          │  • fileassistant search  ◄── NEW
                          │  • fileassistant index   ◄── NEW
                          │  • fileassistant undo    ◄── NEW
                          │  • fileassistant review  ◄── NEW
                          │  • fileassistant correct ◄── NEW
                          │  • fileassistant status       │
                          └──────────────┬───────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
           ┌──────────────┐   ┌──────────────┐    ┌──────────────┐
           │   WATCHER    │   │  PROCESSOR   │    │   SEARCH     │
           │              │   │  (existing)  │    │   ENGINE     │
           │ • Monitor    │   │              │    │   (NEW)      │
           │   inboxes    │──▶│ Analyze ──▶  │    │              │
           │ • Debounce   │   │ Classify ──▶ │    │ • Query      │
           │ • Queue      │   │ Move ──▶     │    │   embedding  │
           └──────────────┘   │ Index ◄── NEW│    │ • Similarity │
                              └──────┬───────┘    │   search     │
                                     │            │ • Re-rank    │
                                     │            │ • Filter     │
                                     │            └──────┬───────┘
                                     │                   │
                    ┌────────────────┼───────────────────┘
                    │                │
                    ▼                ▼
           ┌─────────────────────────────────┐
           │           DATA LAYER            │
           │                                 │
           │  ┌────────────┐ ┌────────────┐  │
           │  │   SQLite   │ │  ChromaDB  │  │
           │  │            │ │            │  │
           │  │ • Files    │ │ • Vectors  │  │
           │  │ • Tags     │ │ • Metadata │  │
           │  │ • Actions  │ │ • Snippets │  │
           │  │ • Rules    │ │            │  │
           │  │ • Correct. │ │            │  │
           │  └────────────┘ └────────────┘  │
           └─────────────────────────────────┘
```

---

## New File Structure

```
src/fileassistant/
├── cli/
│   ├── __init__.py
│   ├── main.py              # Existing CLI entry point
│   ├── watch.py             # Existing
│   ├── process.py           # Existing
│   ├── search.py            # NEW — search command
│   ├── index.py             # NEW — bulk index command
│   ├── undo.py              # NEW — undo command
│   ├── review.py            # NEW — review queue command
│   └── correct.py           # NEW — correction command
├── embeddings/
│   ├── __init__.py
│   └── generator.py         # NEW — embedding generation
├── search/
│   ├── __init__.py
│   ├── engine.py            # NEW — search logic
│   └── index_manager.py     # NEW — ChromaDB wrapper
├── pipeline/
│   ├── watcher.py           # Existing
│   ├── analyzer.py          # Existing (add embedding step)
│   ├── classifier.py        # Existing (add similarity context)
│   └── mover.py             # Existing (add index update)
├── config/                  # Existing
├── database/                # Existing
└── utils/                   # Existing
```

---

## New Configuration Options

```yaml
# Add to existing config
search:
  enabled: true
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 512                    # tokens per chunk
  chunk_overlap: 50                  # overlap between chunks
  chromadb_path: "~/.fileassistant/chromadb"
  default_results: 10
  llm_reranking: false               # Enable LLM re-ranking (slower, more accurate)
  auto_index_on_move: true           # Index files when organized

indexing:
  supported_extensions:
    - .pdf
    - .docx
    - .txt
    - .md
    - .html
    - .py
    - .js
    - .ts
    - .json
    - .yaml
    - .csv
  max_file_size_mb: 50               # Skip files larger than this
  batch_size: 20                     # Files to process in parallel
```

---

## Implementation Order (Recommended for Claude Code)

This is the suggested order for implementing with Claude Code, designed so each step is testable before moving on.

### Sprint 1: Embedding + Index (2A.1 + 2A.2)
1. Install `sentence-transformers` dependency
2. Create `EmbeddingGenerator` class
3. Create `IndexManager` class wrapping ChromaDB
4. Write unit tests for both
5. Integration test: extract text from a PDF → generate embedding → store in ChromaDB → retrieve by ID

### Sprint 2: Bulk Indexer (2A.3)
1. Create `fileassistant index` CLI command
2. Directory walker with filtering
3. Progress bar and error handling
4. Test: index ~/Documents, verify count matches expectations

### Sprint 3: Search (2A.4 + 2A.5)
1. Create `SearchEngine` class
2. Create `fileassistant search` CLI command
3. Test: index some files → search by description → verify relevance
4. Add filters (type, date, tags)
5. Polish output formatting

### Sprint 4: Pipeline Integration (2B.1 + 2B.4)
1. Wire embedding generation into the existing analyzer
2. Wire ChromaDB insertion into the mover (post-move indexing)
3. Add similarity lookup to classifier
4. Test end-to-end: new file arrives → gets processed → gets indexed → is searchable

### Sprint 5: Multi-inbox + Undo (2B.2 + 2B.3)
1. Extend watcher for multiple folders
2. Implement undo logic
3. Test: move a file → undo → verify file returns and index updates

### Sprint 6: Autonomy + Learning (2C)
1. Multi-signal confidence scoring
2. Auto-action based on confidence
3. Correction recording and feedback loop
4. Review queue CLI

---

## Updated Phase Timeline (Full Project)

| Phase | Focus | Weeks | Status |
|-------|-------|-------|--------|
| **Phase 0** | Foundation | 1-2 | ✅ Complete |
| **Phase 1** | MVP Pipeline | 3-5 | ✅ Complete |
| **Phase 2A** | Search Foundation | 6-7 | 🔜 Next |
| **Phase 2B** | Integration & Multi-inbox | 8-9 | Planned |
| **Phase 2C** | Learning & Autonomy | 10-11 | Planned |
| **Phase 3** | Image & Vision | 12-14 | Planned |
| **Phase 4** | Full UI (Tauri) | 15-18 | Planned |
| **Phase 5** | Advanced Intelligence | 19-22 | Planned |
| **Phase 6** | Polish & Distribution | 23-26 | Planned |

---

## What Stays the Same

Everything from Phase 0 and Phase 1 remains as-is. The existing watcher, analyzer, classifier, and mover components continue working. Phase 2 *enhances* them rather than replacing them. The search system is additive — it sits alongside the organization pipeline and shares the same data layer.

Phases 3-6 from the original plan are unchanged in scope, just shifted slightly in timeline to accommodate the expanded Phase 2.
