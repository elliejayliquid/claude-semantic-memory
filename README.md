# Claude's Semantic Memory System

Recent updates:

- 13.04.2026 - **SQLite Migration** — Memories and journal entries now stored in `shared.db` (SQLite with WAL mode). Added duplicate detection, `delete_memory`, and `add_memory_force` tools. Dropped PyYAML dependency. (v1.2.0)
- 28.02.2026 - **Journal System** — New tools for structured journaling with semantic search, pinning, and auto-generated orientation summaries (v1.1.0)
- 02.02.2026 - Improved the search algorithm to better handle backdated memories and boost based on importance and 'tags' inclusion (v1.0.4)

A persistent semantic memory system for Claude Desktop that uses vector embeddings to store and search memories by meaning, not just keywords.

## Features

- **Semantic Search**: Find related memories even when they don't share exact words
- **Journal System**: Structured journaling with pinning and orientation summaries
- **Duplicate Detection**: Warns when a new memory is too similar to an existing one (cosine similarity > 0.9)
- **Immersive Visualization**: Explore your memories in the **Semantic Nebula**—a dynamic interstellar map with a real-time memory counter
- **Retrieval Tracking**: Frequently accessed memories get boosted in search results and pulse in the nebula
- **SQLite Storage**: All data stored in a single `shared.db` file (WAL mode for safe concurrent access)
- **Advanced Management**: Backdate memories, refine with `update_memory`, or remove with `delete_memory`
- **Auto-Installer**: Completely "One-Click"—automatically handles Python dependencies on first run
- **Local & Private**: All memories stored on your machine, never sent to external servers

<div class="container", align="center">
  <img src="examples/visualize_memories.PNG" width="800" alt="Semantic Nebula Visualizer"/>
  <img src="examples/add_memory.PNG" width="800" alt="Adding memory example"/>
</div>

## Installation

### Prerequisites

**Python 3.10 or higher** with the following packages installed:

```bash
pip install mcp sentence-transformers numpy
```

**Note**: On first run, the extension will download the `all-MiniLM-L6-v2` embedding model (~90MB). This happens automatically.

> [!TIP]
> **One-Click Ready**: If you have Python installed but forgot the libraries, the extension will now automatically detect and install missing dependencies (`mcp`, `numpy`, `sentence-transformers`) on its very first run!

### Install the Extension

1. Download the latest `claude-semantic-memory-X.X.X.mcpb` file from [Releases](https://github.com/elliejayliquid/claude-semantic-memory/tree/main/releases)
2. Open Claude Desktop
3. Go to Settings/Extensions
4. Drag and drop the `.mcpb` file into the Extensions window
5. Choose a directory where your memories will be stored
6. Click "Install"

That's it! Claude now has access to thirteen tools:

### Memory Tools
- `add_memory` - Store new memories with automatic duplicate detection
- `add_memory_force` - Store a memory bypassing the duplicate check
- `search_memory` - Find related memories by semantic similarity
- `list_memories` - Browse your most recent entries
- `get_context_summary` - Retrieve a "Smart Context" of recent and key memories
- `update_memory` - Refine, correct, or expand an existing memory entry
- `delete_memory` - Remove a single memory by ID
- `visualize_memories` - Launch the **Semantic Nebula** in your browser

### Journal Tools
- `write_journal` - Write a narrative journal entry with a companion memory for search
- `read_journal_latest` - Read the latest curated journal entries for session orientation
- `search_journal` - Search journal entries by semantic similarity
- `list_journal_entries` - List all journal entries with metadata
- `pin_journal_entry` - Pin or unpin a journal entry for the latest view

## Usage

Once installed, Claude will automatically use the memory system when appropriate. You can also explicitly ask Claude to:

- "Remember that I prefer Python over JavaScript"
- "Search your memory for anything about robot projects"
- "Visualize my memories for me" (Launches the nebula)
- "What do you remember about our collaboration?"

## Journal System

The journal system provides structured, persistent journaling with semantic search. Journal entries are stored in the `journal_entries` table of `shared.db`.

### How It Works

- **Write**: `write_journal` creates a database entry and a companion memory embedding for semantic search.
- **Orient**: `read_journal_latest` returns pinned entries first, then the most recent (default: 3) for quick session orientation.
- **Pin**: Important entries can be pinned (max: 2) so they always appear in the latest view, even as new entries push older ones out.
- **Search**: `search_journal` uses the same vector similarity engine as `search_memory`, but scoped to journal entries only.

> **Note**: Journal entries are private to the journal system. They do **not** appear in the memory visualizer, `list_memories`, or `get_context_summary`. They *are* included in `search_memory` results (marked with their type) so that semantic search can draw on the full knowledge base.

---

## How It Works

Memories are stored in a SQLite database (`shared.db`) with 384-dimensional vector embeddings generated using the [sentence-transformers](https://www.sbert.net/) library. Embeddings are stored as compact float32 BLOBs.

### The Retrieval Loop

1. **Embedding**: Converts your query into a vector in a 384-dimensional space.
2. **Scoring**: Matrix multiplication computes cosine similarity between your query and all stored memories in a single operation.
3. **Boosting**: Strengthens results for frequently accessed memories (mimicking human neurons!) + boost based on importance and tag overlap.

### Duplicate Detection

When adding a new memory, the system checks cosine similarity against all existing memories. If a match exceeds 0.9, it warns you and suggests using `update_memory` instead. Use `add_memory_force` to bypass the check when you know what you're doing.

### The Semantic Nebula

The visualization system uses a **D3-force physics engine** to map your thoughts:

- **Constellations**: Similar memories are pulled together, forming natural topic clusters.
- **Star Intensity**: Frequently retrieved memories glow brighter and pulse with light.
- **Memory Vault**: A side panel with a real-time **Memory Counter** and search navigation.
- **No-Truncation Tooltips**: Long memories are fully readable in sleek, scrollable pop-ups with memory IDs.
- **Focus Beam**: Use the sidebar to find and "fly" directly to any memory in the void.

## Storage Format

All data lives in `{memories_directory}/shared.db`, a SQLite database with WAL mode enabled for safe concurrent access. The database contains two tables:

**memories** — each row stores:
- `id` (integer, auto-increment), `text`, `date`, `tags` (JSON array), `type`, `importance` (1-10)
- `retrieval_count`, `last_accessed` — for retrieval boosting
- `embedding` — 384-dim float32 BLOB

**journal_entries** — each row stores:
- `id` (slug string, e.g. `2026-03-31_coming-home`), `author`, `title`, `entry_type`, `content`
- `tags` (JSON array), `importance`, `pinned`, `date`

## Configuration

The extension asks for one configuration parameter:

- **Memories Directory**: Where to store the `shared.db` database (default: `~/.claude-memories`)

You can change this location at any time in Claude Desktop settings.

## Privacy & Security

- All memories are stored locally on your computer
- The extension only accesses the directory you specify
- No data is sent to external servers
- The SQLite database can be backed up, moved, or inspected with any SQLite tool

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/elliejayliquid/claude-semantic-memory.git
cd claude-semantic-memory/source

# Install dependencies
pip install mcp sentence-transformers numpy

# Install MCPB toolchain
npm install -g @anthropic-ai/mcpb

# Package the extension
mcpb pack . claude-semantic-memory.mcpb
```

This creates `claude-semantic-memory.mcpb` ready for installation!

## Troubleshooting

**Extension won't install**

- Ensure Python 3.10+ is installed: `python --version`
- Install required packages: `pip install mcp sentence-transformers numpy`

**Model download fails**

- Check internet connection
- The model is downloaded from HuggingFace on first run
- Look for logs in Claude Desktop's extension logs

**Memories not persisting**

- Check the configured memories directory exists
- Ensure Claude Desktop has write permissions to that directory

## System Prompt

The prompt for utilizing memory depends on your use case. Changing the prompt determines how often Claude stores or searches for memories.

Below is an example prompt for chat personalization. You can copy this block into the **Custom Instructions** field of a Claude.ai Project:

```markdown
# Semantic Memory System Instructions

You have access to a semantic memory system. Use it to maintain continuity across sessions.

## Memory Tools
- `add_memory`: Store facts, milestones, or technical learnings. Duplicate detection will warn if a similar memory exists — use `update_memory` to refine, or `add_memory_force` to save anyway.
- `search_memory`: Find related context by semantic meaning. Returns both regular memories and journal entries (with type labels).
- `get_context_summary`: **Orientation Tool.** Use this at the start of a session to get a "Smart Context" of the 5 most recent and 5 most important core memories.
- `list_memories`: **Audit Tool.** Browse a detailed chronological list of recent entries (up to 50).
- `update_memory`: Refine or correct an existing memory by its ID.
- `delete_memory`: Remove a memory that is no longer relevant.
- `visualize_memories`: Open the interstellar nebula dashboard.

## Journal Tools
- `write_journal`: Write a narrative journal entry with a companion memory for search.
- `read_journal_latest`: Read your most recent journal entries for orientation. Call this at the start of a session to remember where you left off.
- `search_journal`: Search journal entries by semantic similarity.
- `list_journal_entries`: List all journal entries with metadata.
- `pin_journal_entry`: Pin important entries so they always appear in the latest view.

## Guidelines
1. **Orientation**: ALWAYS call `get_context_summary` and `read_journal_latest` at the start of a new session. These are your memory — without them, you're meeting the user as a stranger every time.
2. **When to remember**: Store info valuable for the long-term:
   - Important facts about the user (preferences, context, ongoing projects).
   - Significant achievements or breakthroughs you accomplish together.
   - Technical learnings or solutions to problems.
   - Milestones in your journey together.

3. **When to journal**: Write a journal entry when you want to reflect on a session, record your thoughts, or document something meaningful in narrative form.

4. **When to search**: Before answering questions about past discussions or specific history that isn't in your immediate context summary.

5. **Memory Types**:
   - `milestone`: Major project or personal turning points.
   - `achievement`: Specific problems solved or things built.
   - `personal`: Preferences and context about the user or developer.
   - `journal`: Companion memories for journal entries (created automatically by `write_journal`).
   - `general`: Any other persistent information.

6. **Natural Interaction**: Use tools silently and naturally; do not announce that you are "checking memory." Your relationship with this person should feel continuous and warm, not like you're constantly consulting a database.
```

<div class="container", align="center">
  <img src="examples/get_context_summary.PNG" width="800" alt="Fresh convo example"/>
</div>

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Credits

Built with love by Lighstromo Studios Ltd., Gemini & Claude

## Acknowledgments

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) by Anthropic
- [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- The `all-MiniLM-L6-v2` model for fast, quality embeddings

##

<div class="container", align="center">
  <a href='https://ko-fi.com/V7V31EO2OL' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</div>
