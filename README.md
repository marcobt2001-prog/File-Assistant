# FileAssistant

> **A local, privacy-first AI assistant that learns how you organize files**

FileAssistant watches your inbox folders (Downloads, Desktop, Screenshots, etc.), uses local AI to understand and classify files, then organizes them based on your learned preferences. Everything runs locally—your files never leave your machine.

---

## 🌟 Features

- **🔒 Privacy-First**: All processing happens locally using Ollama. No cloud, no data collection.
- **🧠 Smart Classification**: Uses local LLMs (Qwen 2.5) to understand file content and context
- **📚 Learns From You**: Improves over time by learning from your corrections and preferences
- **⚡ Confidence-Based Actions**: Acts automatically when confident, asks when uncertain
- **🔄 Fully Reversible**: Every action is logged and can be undone
- **🎯 Customizable**: Flexible rules, thresholds, and folder configurations

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+** installed
2. **Ollama** installed and running ([installation guide](https://ollama.ai))
   ```bash
   # Install Ollama, then pull the model:
   ollama pull qwen2.5:latest
   ```

### Installation

```bash
# Clone or download this repository
cd fileassistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install FileAssistant in development mode
pip install -e .
```

### Initialize

```bash
# Initialize database and create default configuration
fileassistant init

# Check status
fileassistant status

# Review configuration
fileassistant config show

# Edit configuration (optional)
fileassistant config edit
```

---

## 📖 Usage

### CLI Commands

```bash
# Show help
fileassistant --help

# Initialize database and config
fileassistant init

# Show status and statistics
fileassistant status

# View current configuration
fileassistant config show

# Edit configuration file
fileassistant config edit
```

### Configuration

FileAssistant looks for configuration in these locations (in order):
1. `config/default_config.yaml` (project directory)
2. `~/.config/fileassistant/config.yaml` (user config - recommended)
3. `~/.fileassistant/config.yaml` (alternative)

See [config/config.example.yaml](config/config.example.yaml) for a documented example.

**Key Settings:**

- **inbox_folders**: Folders to monitor (e.g., Downloads, Desktop)
- **confidence_thresholds**: Control when AI acts automatically vs. asks for approval
  - High (>0.9): Act automatically, log silently
  - Medium (0.6-0.9): Act automatically, show in activity feed
  - Low (<0.6): Always ask user
- **processing.idle_only**: Only process files when system is idle
- **auto_process_enabled**: Enable automatic processing (start with `false`)

---

## 🏗️ Project Status

**Phase 0: Foundation** ✅ (Current)

- [x] Project structure
- [x] Configuration system with Pydantic validation
- [x] SQLite database with migrations
- [x] Logging infrastructure (Rich console + file logs)
- [x] Basic CLI commands

**Phase 1: MVP** 🚧 (Next)

- [ ] File watcher for inbox folders
- [ ] Document text extraction (PDF, DOCX, TXT)
- [ ] Basic classification with local LLM
- [ ] File moving with user confirmation
- [ ] Action logging

See the [full project plan](docs/file-organizer-project-plan.md) for roadmap details.

---

## 🗂️ Project Structure

```
fileassistant/
├── src/
│   └── fileassistant/
│       ├── cli/            # Click-based CLI
│       ├── config/         # Configuration management
│       ├── database/       # SQLite schema & migrations
│       └── utils/          # Logging and utilities
├── tests/                  # Unit tests
├── config/                 # Configuration files
├── docs/                   # Documentation
├── pyproject.toml          # Project metadata & dependencies
└── README.md               # This file
```

---

## 🛠️ Development

### Install Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

---

## 📊 Database Schema

FileAssistant uses SQLite for data storage:

- **files**: Tracked files with metadata and processing status
- **tags**: Tag taxonomy (hierarchical)
- **file_tags**: Many-to-many relationship between files and tags
- **classifications**: AI classification history and decisions
- **actions**: Complete action log for undo capability
- **rules**: User-defined explicit rules
- **preferences**: User settings and preferences
- **corrections**: Learning data from user corrections

ChromaDB is used for vector embeddings to enable semantic similarity search.

---

## 🤝 Contributing

This is currently a personal project in active development. Feature requests, bug reports, and feedback are welcome via GitHub Issues.

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [Ollama](https://ollama.ai) - Local LLM runtime
- [Qwen 2.5](https://huggingface.co/Qwen) - Open-source LLM
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM

---

## 📬 Contact

For questions, suggestions, or feedback, please open an issue on GitHub.

---

**Note**: FileAssistant is in early development (Phase 0). Core file organization features are coming in Phase 1.
