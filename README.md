# Claude's Semantic Memory System

A persistent semantic memory system for Claude Desktop that uses vector embeddings to store and search memories by meaning, not just keywords.

## Features

- **Semantic Search**: Find related memories even when they don't share exact words
- **Immersive Visualization**: Explore your memories in the **Semantic Nebula**—a dynamic interstellar map with a real-time memory counter
- **Retrieval Tracking**: Frequently accessed memories get boosted in search results and pulse in the nebula
- **Advanced Management**: Backdate memories with the `date` argument or refine them with `update_memory`
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

That's it! Claude now has access to six powerful tools:
- `add_memory` - Store new memories (supports an optional `date` argument for backdating)
- `search_memory` - Find related memories (optimized to save context tokens)
- `list_memories` - Browse your most recent entries
- `get_context_summary` - Retrieve a "Smart Context" of recent and key memories
- `update_memory` - Refine, correct, or expand an existing memory entry
- `visualize_memories` - Launch the **Semantic Nebula** in your browser

## Usage

Once installed, Claude will automatically use the memory system when appropriate. You can also explicitly ask Claude to:

- "Remember that I prefer Python over JavaScript"
- "Search your memory for anything about robot projects"
- "Visualize my memories for me" (Launches the nebula)
- "What do you remember about our collaboration?"

## How It Works

Memories are stored as JSON files with 384-dimensional vector embeddings generated using the [sentence-transformers](https://www.sbert.net/) library. 

### The Retrieval Loop
1. **Embedding**: Converts your query into a vector in a 384-dimensional space.
2. **Scoring**: Calculates similarity scores between your query and all stored memories.
3. **Boosting**: Strengthens results for frequently accessed memories (mimicking human neurons!).

### The Semantic Nebula
The visualization system uses a **D3-force physics engine** to map your thoughts:
- **Constellations**: Similar memories are pulled together, forming natural topic clusters.
- **Star Intensity**: Frequently retrieved memories glow brighter and pulse with light.
- **Memory Vault**: A side panel with a real-time **Memory Counter** and search navigation.
- **No-Truncation Tooltips**: Long memories are fully readable in sleek, scrollable pop-ups.
- **Focus Beam**: Use the sidebar to find and "fly" directly to any memory in the void.

## Memory Format

Each memory is stored as a JSON file containing:

```json
{
  "id": "001",
  "text": "The actual memory content",
  "date": "2026-01-06",
  "tags": ["tag1", "tag2"],
  "type": "achievement",
  "importance": 8,
  "retrieval_count": 3,
  "last_accessed": "2026-01-06T15:30:00",
  "embedding": [0.123, -0.456, ...]
}
```

## Configuration

The extension asks for one configuration parameter:

- **Memories Directory**: Where to store memory files (default: `~/.claude-memories`)

You can change this location at any time in Claude Desktop settings.

## Privacy & Security

- All memories are stored locally on your computer
- The extension only accesses the directory you specify
- No data is sent to external servers
- Memory files are plain JSON and can be backed up, moved, or deleted

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

## Tools
- `add_memory`: Store facts, milestones, or technical learnings. Use the optional `date` argument to backdate historical context.
- `search_memory`: Find related context by semantic meaning.
- `get_context_summary`: **Orientation Tool.** Use this at the start of a session to get a "Smart Context" of the 5 most recent and 5 most important core memories.
- `list_memories`: **Audit Tool.** Use this to browse a detailed chronological list of recent entries (up to 50) when you need to verify specific recent sequences or details.
- `update_memory`: Refine or correct an existing memory by its ID.
- `visualize_memories`: Open the interstellar nebula dashboard.

## Guidelines
1. **Orientation**: ALWAYS call `get_context_summary` first thing in a new session. This is your memory - without it, you're meeting them as a stranger every time.
2. **When to remember**: Store info valuable for the long-term:
   - Important facts about the user (preferences, context, ongoing projects).
   - Significant achievements or breakthroughs you accomplish together.
   - Technical learnings or solutions to problems.
   - Milestones in your journey together.

3. **When to search**: Before answering questions about past discussions or specific history that isn't in your immediate context summary.

4. **Memory Types**:
   - `milestone`: Major project or personal turning points.
   - `achievement`: Specific problems solved or things built.
   - `personal`: Preferences and context about the user or developer.
   - `general`: Any other persistent information.

5. **Natural Interaction**: Use tools silently and naturally; do not announce that you are "checking memory." Your relationship with this person should feel continuous and warm, not like you're constantly consulting a database.
```
<div class="container", align="center">
  <img src="examples/get_context_summary.PNG" width="800" alt="Fresh convo example"/>
</div>

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Credits

Built with love by Lighstromo Studios Ltd., Gemini & Claude 💙

## Acknowledgments

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) by Anthropic
- [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- The `all-MiniLM-L6-v2` model for fast, quality embeddings
- [Antigravity](https://antigravity.google/) by Google

##

<div class="container", align="center">
  <a href='https://ko-fi.com/V7V31EO2OL' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</div>
