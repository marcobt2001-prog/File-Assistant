# Intelligent File Organizer — Project Plan

> **Codename**: *FileAssistant* (working title)
> **Vision**: A local, privacy-first AI assistant that learns how you organize files and handles the tedium for you—like a thoughtful personal assistant who keeps your digital life in order.

---

## Table of Contents

1. [Project Vision & Principles](#1-project-vision--principles)
2. [Core Concepts](#2-core-concepts)
3. [System Architecture](#3-system-architecture)
4. [Component Breakdown](#4-component-breakdown)
5. [Development Phases](#5-development-phases)
6. [Technical Decisions](#6-technical-decisions)
7. [Open Questions](#7-open-questions)
8. [Success Metrics](#8-success-metrics)
9. [Risk & Mitigation](#9-risk--mitigation)

---

## 1. Project Vision & Principles

### The Problem

People accumulate files constantly—downloads, screenshots, documents, images—but organizing them is tedious. Most people either:
- Let chaos reign (thousands of files in Downloads)
- Spend significant time manually organizing
- Use rigid systems that don't adapt to their actual needs

### The Solution

An intelligent local assistant that:
- **Watches** designated inbox folders for new files
- **Understands** what each file is (content, context, purpose)
- **Learns** how the user prefers to organize things
- **Acts** autonomously for routine decisions, asks for guidance on uncertain ones
- **Evolves** its understanding as the user's needs change

### Core Principles

| Principle | What It Means |
|-----------|---------------|
| **Privacy First** | All processing happens locally. No data leaves the machine. User owns everything. |
| **Assistant, Not Autocrat** | The system serves the user's preferences, not its own ideas of "correct" organization. |
| **Confidence-Based Autonomy** | Act automatically when confident, ask when uncertain, always allow override. |
| **Transparent & Reversible** | Every action is logged and undoable. User can always see *why* a decision was made. |
| **Gentle on Resources** | Never degrade system performance. Process during idle time. Be invisible when user is active. |
| **Progressive Enhancement** | Start simple, add intelligence over time. Never require a powerful machine to be useful. |

### The "Good Assistant" Mental Model

Think of a skilled human assistant managing files:
- They'd observe how the boss currently organizes things before changing anything
- They'd ask clarifying questions early on, then act more autonomously once they understand
- They'd never make major structural changes without approval
- They'd surface anomalies ("I noticed 5 copies of this file—want me to clean that up?")
- They'd get better at anticipating needs over time
- They'd adapt when the boss's needs change

This is the behavior we're building.

---

## 2. Core Concepts

### 2.1 Confidence Levels

Every classification decision has a confidence score that determines behavior:

| Level | Score | Behavior |
|-------|-------|----------|
| **High** | >90% | Act automatically, log action silently |
| **Medium** | 60-90% | Act automatically, show in activity feed for review |
| **Low** | <60% | Queue for user decision with AI's recommendation |
| **Structural** | Any | Always require user approval (new folders, reorganization) |

Thresholds are user-configurable. Some users want more autonomy, some want more control.

### 2.2 File Understanding Layers

Each file is understood at multiple levels:

```
┌─────────────────────────────────────────────┐
│ Layer 4: CONTEXTUAL                         │
│ "Part of the home renovation project"       │
│ "Related to files from last week"           │
├─────────────────────────────────────────────┤
│ Layer 3: SEMANTIC                           │
│ "An invoice from Home Depot for lumber"     │
│ "A screenshot of an error message"          │
├─────────────────────────────────────────────┤
│ Layer 2: CONTENT                            │
│ Text extracted, entities identified         │
│ Image described, objects detected           │
├─────────────────────────────────────────────┤
│ Layer 1: METADATA                           │
│ File type, size, dates, source folder       │
│ EXIF data, file hash                        │
└─────────────────────────────────────────────┘
```

MVP needs Layer 1-2. Full vision includes Layer 3-4.

### 2.3 Classification Outputs

For each file, the system determines:

- **Tags**: Descriptive labels (e.g., "invoice", "2024", "home-depot", "renovation")
- **Destination**: Where the file should live (e.g., `Projects/Home Renovation/Receipts/`)
- **Confidence**: How sure the system is about this classification
- **Reasoning**: Human-readable explanation of why (for transparency)

### 2.4 Learning Sources

The system learns from multiple signals:

| Source | What It Teaches |
|--------|-----------------|
| **Existing file structure** | User's organizational preferences and patterns |
| **User corrections** | When AI is wrong, what's actually right |
| **Explicit rules** | Hard overrides that always apply |
| **User feedback** | Thumbs up/down on suggestions |
| **Temporal patterns** | "Files like this usually come on Mondays" |

### 2.5 Structural vs. Incremental Decisions

**Incremental decisions** (where to put a single file):
- Can be automated based on learned patterns
- Low risk—easy to undo

**Structural decisions** (changing folder organization):
- Always require user approval
- Examples: creating new folders, suggesting reorganization, detecting that a category is obsolete
- Presented as proposals with clear reasoning

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐ │
│  │ System Tray │ │  Activity   │ │  Decision   │ │   Settings    │ │
│  │   + Popup   │ │    Feed     │ │    Queue    │ │  & Preferences│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR                               │
│                                                                     │
│  • Processing queue management                                      │
│  • Confidence threshold enforcement                                 │
│  • User approval workflow                                           │
│  • Pattern detection over time                                      │
│  • Resource/idle monitoring                                         │
└─────────────────────────────────────────────────────────────────────┘
                                 │
         ┌───────────────┬───────┴───────┬───────────────┐
         ▼               ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   WATCHER   │  │  ANALYZER   │  │ CLASSIFIER  │  │   MOVER     │
│             │  │             │  │             │  │             │
│ • Monitor   │  │ • Extract   │  │ • Determine │  │ • Execute   │
│   folders   │  │   metadata  │  │   tags      │  │   file ops  │
│ • Detect    │  │ • Extract   │  │ • Determine │  │ • Create    │
│   changes   │  │   content   │  │   dest      │  │   folders   │
│ • Queue     │  │ • Generate  │  │ • Calculate │  │ • Log       │
│   files     │  │   embedding │  │   confidence│  │   actions   │
│ • Debounce  │  │ • Run OCR   │  │ • Provide   │  │ • Handle    │
│             │  │             │  │   reasoning │  │   undo      │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                │
│                                                                     │
│  ┌────────────────────────┐       ┌────────────────────────────┐   │
│  │    SQLite Database     │       │   Vector Store (ChromaDB)  │   │
│  │                        │       │                            │   │
│  │ • File records         │       │ • Content embeddings       │   │
│  │ • Tags & taxonomy      │       │ • Semantic similarity      │   │
│  │ • Folder structure     │       │                            │   │
│  │ • User preferences     │       │                            │   │
│  │ • Rules                │       │                            │   │
│  │ • Action history       │       │                            │   │
│  │ • Learning data        │       │                            │   │
│  └────────────────────────┘       └────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LOCAL AI ENGINE                              │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Ollama/llama   │  │ Embedding Model │  │  Vision Model       │ │
│  │  .cpp Backend   │  │ (MiniLM, etc)   │  │  (LLaVA, etc)       │ │
│  │                 │  │                 │  │                     │ │
│  │ • Chat/reason   │  │ • Fast, always  │  │ • Optional          │ │
│  │ • Classification│  │   available     │  │ • For images        │ │
│  │ • Decisions     │  │ • Similarity    │  │ • Screenshots       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│                                                                     │
│  • Model management (download, update, switch)                      │
│  • Inference queue with priority levels                             │
│  • Resource monitoring & throttling                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

**New file arrives:**
```
1. Watcher detects file in inbox folder
2. Watcher waits for file to be stable (not still downloading)
3. Watcher adds to processing queue
4. Orchestrator checks if system is idle (or user allows immediate processing)
5. Analyzer extracts metadata + content
6. Analyzer generates embedding, stores in vector DB
7. Classifier queries similar files, gets classification context
8. Classifier runs LLM to determine tags + destination + confidence
9. Orchestrator checks confidence level:
   - High: Mover executes, logs silently
   - Medium: Mover executes, adds to activity feed
   - Low: Adds to decision queue for user
10. Action recorded in database
```

**User corrects a decision:**
```
1. User moves file to different location (or uses UI to correct)
2. System detects the correction
3. Records: "For file with these characteristics, user preferred X over Y"
4. Updates classifier training data
5. Optionally: "I noticed you moved this. Should I handle similar files this way?"
```

---

## 4. Component Breakdown

### 4.1 Watcher Component

**Purpose**: Detect new/changed files in designated folders

**Responsibilities**:
- Monitor configured inbox folders (Downloads, Desktop, Screenshots, custom)
- Detect file creation, modification, rename, delete events
- Debounce events (wait for file to be fully written)
- Add stable files to processing queue
- Handle edge cases (partial downloads, temp files, locked files)

**Technical approach**:
- Python: `watchdog` library
- Filter out temporary files (`.tmp`, `.part`, `.crdownload`, etc.)
- Configurable debounce delay (default: 2 seconds of no changes)

**Complexity**: Low

### 4.2 Analyzer Component

**Purpose**: Extract all useful information from a file

**Responsibilities**:
- Extract metadata (size, dates, type, hash)
- Extract content based on file type:
  - Documents: full text via `pymupdf`, `python-docx`, `textract`
  - Images: EXIF data, OCR via `pytesseract`, optional vision model description
  - Code: language detection, basic parsing
  - Archives: list contents
- Generate text summary for complex files
- Create embedding vector for semantic search
- Cache results (don't re-analyze unchanged files)

**Technical approach**:
- Pluggable extractors per file type
- Embedding via `sentence-transformers` (MiniLM or similar)
- Vision understanding via `llama.cpp` with LLaVA (optional, Phase 3)

**Complexity**: Medium (many file types to handle)

### 4.3 Classifier Component

**Purpose**: Decide tags and destination for a file

**Responsibilities**:
- Query vector DB for similar files
- Query rules engine for any matching explicit rules
- Build context: similar files, existing folder structure, user preferences
- Prompt local LLM with context, get classification decision
- Parse LLM response into structured output (tags, destination, confidence, reasoning)
- Return classification result

**Technical approach**:
- Hybrid: check rules first (fast), then embedding similarity, then LLM for uncertain cases
- Prompt engineering for consistent, parseable output
- Confidence calibration based on LLM's stated confidence + similarity scores

**Complexity**: High (core intelligence lives here)

### 4.4 Mover Component

**Purpose**: Execute file operations safely and reversibly

**Responsibilities**:
- Move files to destination
- Create folders as needed (with approval for new ones)
- Handle naming conflicts (same name exists at destination)
- Record all actions for undo capability
- Handle failures gracefully (permissions, disk full, etc.)

**Technical approach**:
- Atomic operations where possible
- Comprehensive action log with before/after state
- Soft delete capability (move to trash rather than delete)

**Complexity**: Low-Medium

### 4.5 Orchestrator Component

**Purpose**: Coordinate everything, enforce policies

**Responsibilities**:
- Manage processing queue (priority, ordering)
- Enforce confidence thresholds
- Route low-confidence decisions to user queue
- Monitor system resources, pause processing when busy
- Detect patterns over time (new categories emerging, etc.)
- Handle user approvals and corrections
- Maintain overall system state

**Technical approach**:
- Central event bus / message queue
- State machine for file lifecycle
- Background thread for resource monitoring

**Complexity**: Medium-High (lots of coordination)

### 4.6 Data Layer

**Purpose**: Persist all data reliably

**Components**:

**SQLite Database**:
- `files`: all known files, their metadata, current location, tags
- `tags`: tag taxonomy and hierarchy
- `folders`: known folder structure and purposes
- `rules`: user-defined explicit rules
- `actions`: complete action history for undo
- `preferences`: user settings
- `learning`: classification corrections, feedback

**Vector Store (ChromaDB)**:
- Content embeddings for all processed files
- Enables semantic similarity search
- Lightweight, embeds in application

**Complexity**: Medium

### 4.7 Local AI Engine

**Purpose**: Run ML models locally

**Responsibilities**:
- Manage model downloads and updates
- Provide inference API for classifier
- Queue and prioritize inference requests
- Monitor resource usage, throttle as needed
- Support multiple model types (chat, embedding, vision)

**Technical approach**:
- Ollama as backend (simplifies model management)
- Fallback to `llama.cpp` direct for more control
- Sentence-transformers for embeddings (separate from Ollama)

**Complexity**: Medium

### 4.8 User Interface

**Purpose**: User interaction and control

**Components**:

**System Tray**:
- Status indicator (idle, processing, needs attention)
- Quick stats (files processed today)
- Click to open main UI

**Activity Feed**:
- Recent actions taken
- "Medium confidence" actions for review
- One-click undo

**Decision Queue**:
- Files awaiting user decision
- AI's recommendation with reasoning
- Accept / Modify / Skip options

**Settings**:
- Inbox folders configuration
- Confidence thresholds
- Performance settings
- Rules editor
- Folder structure viewer

**Onboarding**:
- Initial scan results
- Structure confirmation
- Preference gathering

**Technical approach**:
- TBD based on stack decision (Electron, Tauri, native, web-based)

**Complexity**: Medium-High (UX is critical)

---

## 5. Development Phases

### Phase 0: Foundation (Week 1-2)

**Goal**: Project scaffolding, core infrastructure

**Deliverables**:
- [ ] Project repository setup
- [ ] Basic project structure
- [ ] Configuration management
- [ ] Logging infrastructure
- [ ] SQLite database schema + migrations
- [ ] Basic CLI for testing

**No user-facing functionality yet.**

---

### Phase 1: MVP — Basic Pipe (Week 3-5)

**Goal**: Prove the core concept works end-to-end

**Scope**:
- Watch ONE folder (Downloads)
- Extract text from documents (PDF, TXT, MD, DOCX)
- Classify using local LLM
- Move files to suggested destination
- Simple CLI interface
- Action logging (no undo yet)

**Capabilities**:
- Watches Downloads folder
- Processes text-based documents
- Uses local LLM (Ollama + Llama 3.2 8B recommended)
- Proposes destination, waits for user confirmation via CLI
- Moves file on approval

**Not included**:
- Image understanding
- Embedding/similarity search
- Automatic actions (all require confirmation)
- GUI
- Learning from corrections
- Multiple inbox folders

**Success criteria**:
- Can correctly classify 70%+ of test documents
- End-to-end flow works reliably
- Processing doesn't crash or hang

---

### Phase 2: Learning & Autonomy (Week 6-8)

**Goal**: System gets smarter and can act autonomously

**Scope**:
- Embedding generation + vector store
- Similarity-based classification boost
- Confidence scoring
- Automatic action for high-confidence decisions
- Learning from user corrections
- Multiple inbox folders
- Undo capability
- Basic system tray UI (status only)

**Capabilities**:
- Learns from existing file structure
- Gets better as user corrects mistakes
- Acts automatically when confident
- Shows activity feed of recent actions
- Supports undo for recent actions

**Success criteria**:
- Classification accuracy improves over time
- High-confidence automatic actions are correct 95%+
- User can easily review and undo actions

---

### Phase 3: Image & Vision (Week 9-11)

**Goal**: Understand images and screenshots

**Scope**:
- EXIF extraction for images
- OCR for screenshots and document photos
- Vision model integration (LLaVA)
- Screenshot-specific handling (detect app, extract text)

**Capabilities**:
- Organizes photos by date/location
- Reads text in screenshots
- Describes and categorizes images
- Handles document scans

**Success criteria**:
- Screenshots with text are correctly categorized
- Photos are organized sensibly

---

### Phase 4: Full UI (Week 12-15)

**Goal**: Polished user experience

**Scope**:
- Full desktop application
- Onboarding flow
- Settings and preferences UI
- Rules editor
- Decision queue interface
- Activity feed with filters
- Manual tagging interface
- Folder structure visualization

**Capabilities**:
- Complete graphical interface
- New users can set up easily
- Power users can configure deeply

**Success criteria**:
- Non-technical user can install and use successfully
- All features accessible via UI

---

### Phase 5: Advanced Intelligence (Week 16-20)

**Goal**: Proactive, pattern-aware assistant

**Scope**:
- Pattern detection over time
- Proactive suggestions ("You've been getting a lot of X files...")
- Structural change proposals
- Duplicate detection
- "Related files" grouping
- Natural language commands ("Find that receipt from Home Depot")

**Capabilities**:
- Notices emerging categories
- Suggests organizational improvements
- Finds duplicates and near-duplicates
- Semantic search across files

**Success criteria**:
- System proactively identifies useful improvements
- Users find files faster than manual search

---

### Phase 6: Polish & Distribution (Week 21-24)

**Goal**: Ready for public release

**Scope**:
- Installer/packaging for Windows, macOS, Linux
- Auto-update mechanism
- Performance optimization
- Documentation
- User profiles (personal, creative, developer, etc.)
- Onboarding templates

**Capabilities**:
- Easy installation
- Works across platforms
- Handles edge cases gracefully

**Success criteria**:
- Someone can download, install, and get value within 10 minutes
- No data loss bugs
- Reasonable performance on modest hardware

---

## 6. Technical Decisions

### 6.1 Decided

| Decision | Choice | Rationale |
|----------|--------|-----------|
| AI Processing | Local only | Privacy is core principle |
| Primary LLM | Ollama + Llama 3.x | Good balance of capability and resource usage |
| Embedding Model | all-MiniLM-L6-v2 or similar | Fast, small, good quality |
| Database | SQLite | Simple, no server, portable |
| Vector Store | ChromaDB | Embeddable, Python-native |

### 6.2 To Decide

| Decision | Options | Considerations |
|----------|---------|----------------|
| **Programming Language** | Python / Rust / Go | Python: fastest dev, best ML ecosystem. Rust: performance, single binary. Go: good middle ground. |
| **UI Framework** | Electron / Tauri / Native / Web | Electron: easy, heavy. Tauri: lighter, Rust-based. Native: best UX, most work. Web: accessible, limited system integration. |
| **Distribution** | PyInstaller / Nuitka / Native build | Single executable preferred for ease of use |
| **Minimum model size** | 1B / 3B / 7B | Smaller = faster, less accurate. Need to test. |
| **Config format** | YAML / TOML / JSON | Preference/familiarity |

### 6.3 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Local LLM quality insufficient | Core feature broken | Test extensively in Phase 1; have tiered model options |
| Performance on low-end machines | Limits audience | Tiered processing; aggressive idle-only scheduling |
| File permission issues | Can't move files | Clear error handling; guide user to fix |
| Model download size | Barrier to entry | Offer "lite" mode with smaller model |

---

## 7. Open Questions

### Product Questions

- [ ] What's the product name?
- [ ] Free / paid / freemium model?
- [ ] What platforms to support initially? (Recommend: start with one you use)
- [ ] How to handle very large files? (GB+ videos)
- [ ] Should it organize existing files, or only new ones?
- [ ] How to handle cloud-synced folders (Dropbox, OneDrive, iCloud)?

### Technical Questions

- [ ] Best way to detect system idle cross-platform?
- [ ] How to handle files that are opened/locked by other applications?
- [ ] Optimal embedding model for file content?
- [ ] How to persist vector store efficiently?
- [ ] How to handle file renames/moves by other applications?

### UX Questions

- [ ] How much information to show in the activity feed?
- [ ] How to make the decision queue not feel like a chore?
- [ ] How to visualize the folder structure and AI's understanding?
- [ ] How to explain AI confidence in a user-friendly way?

---

## 8. Success Metrics

### MVP Success

- [ ] End-to-end flow works without crashes
- [ ] Processes a file in <30 seconds on target hardware
- [ ] Correct classification 70%+ of the time

### v1.0 Success

- [ ] User can set up and get value in <10 minutes
- [ ] High-confidence automatic actions correct 95%+
- [ ] System measurably improves over 1 week of use
- [ ] No data loss incidents
- [ ] Runs without noticeable performance impact

### Long-term Success

- [ ] Users report feeling "more organized"
- [ ] Users spend less time managing files
- [ ] System handles 90%+ of files without user intervention
- [ ] Active user retention after 30 days

---

## 9. Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Accidental data loss** | Medium | Critical | Extensive testing; undo capability; optional "dry run" mode |
| **Poor classification quality** | Medium | High | Hybrid approach (rules + similarity + LLM); easy correction; continuous learning |
| **Performance issues** | Medium | Medium | Aggressive idle scheduling; tiered processing; user controls |
| **Scope creep** | High | Medium | Strict phase boundaries; MVP-first mentality |
| **Complex installation** | Medium | Medium | Single-file distribution; clear docs; first-run wizard |
| **Model compatibility** | Low | Medium | Abstract model layer; support multiple backends |

---

## Appendix A: File Type Support Matrix

| File Type | Phase | Metadata | Content | Embedding | Vision |
|-----------|-------|----------|---------|-----------|--------|
| .txt, .md | 1 | ✓ | ✓ | ✓ | — |
| .pdf | 1 | ✓ | ✓ | ✓ | — |
| .docx | 1 | ✓ | ✓ | ✓ | — |
| .xlsx | 2 | ✓ | ✓ | ✓ | — |
| .jpg, .png | 3 | ✓ | OCR | ✓ | ✓ |
| .html | 2 | ✓ | ✓ | ✓ | — |
| Code files | 2 | ✓ | ✓ | ✓ | — |
| .mp3, .wav | 4+ | ✓ | Transcript? | ✓ | — |
| .mp4, .mov | 4+ | ✓ | Limited | Limited | — |
| .zip, .rar | 2 | ✓ | List contents | — | — |

---

## Appendix B: Example User Rules

```yaml
rules:
  # Explicit file type routing
  - name: "Photoshop files to Creative"
    condition:
      extension: [.psd, .psb]
    action:
      destination: "Creative/Photoshop/"
      tags: ["creative", "photoshop"]

  # Content-based routing
  - name: "Invoices to Finances"
    condition:
      content_contains: ["invoice", "receipt", "payment"]
    action:
      destination: "Documents/Finances/{year}/"
      tags: ["financial", "invoice"]

  # Source-based routing
  - name: "Screenshots to Screenshots"
    condition:
      source_folder: "Screenshots"
    action:
      destination: "Screenshots/{year}/{month}/"

  # Exclusions
  - name: "Never touch Archives"
    condition:
      destination_starts_with: "Archives/"
    action:
      skip: true

  # Complex condition
  - name: "Work receipts"
    condition:
      all:
        - content_contains: ["invoice", "receipt"]
        - content_contains: ["Acme Corp", "work expense"]
    action:
      destination: "Work/Expenses/{year}/"
      tags: ["work", "expense", "tax-deductible"]
```

---

## Appendix C: Database Schema (Draft)

```sql
-- Core file tracking
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT,
    size_bytes INTEGER,
    hash_md5 TEXT,
    created_at TIMESTAMP,
    modified_at TIMESTAMP,
    processed_at TIMESTAMP,
    status TEXT DEFAULT 'pending', -- pending, processed, error, skipped
    content_summary TEXT,
    embedding_id TEXT, -- reference to vector store
    UNIQUE(path)
);

-- Tags
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    color TEXT,
    parent_tag_id INTEGER REFERENCES tags(id),
    auto_generated BOOLEAN DEFAULT FALSE
);

CREATE TABLE file_tags (
    file_id INTEGER REFERENCES files(id),
    tag_id INTEGER REFERENCES tags(id),
    confidence REAL,
    source TEXT, -- 'ai', 'user', 'rule'
    PRIMARY KEY (file_id, tag_id)
);

-- Classification history
CREATE TABLE classifications (
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    suggested_destination TEXT,
    suggested_tags TEXT, -- JSON array
    confidence REAL,
    reasoning TEXT,
    status TEXT, -- 'pending', 'accepted', 'rejected', 'modified'
    final_destination TEXT,
    final_tags TEXT
);

-- Action log for undo
CREATE TABLE actions (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action_type TEXT, -- 'move', 'tag', 'create_folder', 'delete'
    file_id INTEGER REFERENCES files(id),
    before_state TEXT, -- JSON
    after_state TEXT, -- JSON
    undone BOOLEAN DEFAULT FALSE,
    undone_at TIMESTAMP
);

-- User-defined rules
CREATE TABLE rules (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    condition_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User preferences
CREATE TABLE preferences (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learning data
CREATE TABLE corrections (
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_classification TEXT, -- JSON
    corrected_classification TEXT, -- JSON
    learned BOOLEAN DEFAULT FALSE
);
```

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| [Today] | 0.1 | Initial project plan created |

