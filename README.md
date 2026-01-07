# Claude's Semantic Memory System

A persistent semantic memory system for Claude Desktop that uses vector embeddings to store and search memories by meaning, not just keywords.

## Features

- **Semantic Search**: Find related memories even when they don't share exact words
- **Retrieval Tracking**: Frequently accessed memories get boosted in search results
- **Local & Private**: All memories stored on your machine, never sent to external servers
- **One-Click Install**: Packaged as a Claude Desktop Extension (.mcpb)
- **Configurable Storage**: Choose where to store your memories

## Installation

### Prerequisites

**Python 3.10 or higher** with the following packages installed:

```bash
pip install mcp sentence-transformers numpy
```

**Note**: On first run, the extension will download the `all-MiniLM-L6-v2` embedding model (~90MB). This happens automatically but requires an internet connection.

### Install the Extension

1. Download the latest `claude-semantic-memory-X.X.X.mcpb` file from [Releases](https://github.com/elliejayliquid/claude-semantic-memory/tree/main/releases)
2. Open Claude Desktop
3. Go to Settings/Extensions
4. Drag and drop the `.mcpb` file into the Extensions window
5. Choose a directory where your memories will be stored
6. Click "Install"

That's it! Claude now has access to three new tools:
- `add_memory` - Store new memories
- `search_memory` - Find related memories semantically
- `list_memories` - Browse recent memories

## Usage

Once installed, Claude will automatically use the memory system when appropriate. You can also explicitly ask Claude to:

- "Remember that I prefer Python over JavaScript"
- "Search your memory for anything about robot projects"
- "What do you remember about our collaboration?"

## How It Works

Memories are stored as JSON files with 384-dimensional vector embeddings generated using the [sentence-transformers](https://www.sbert.net/) library. When you search, the system:

1. Converts your query into a vector embedding
2. Calculates similarity scores with all stored memories
3. Boosts scores for frequently accessed memories
4. Returns the most relevant results

The system mimics human memory by strengthening frequently retrieved memories!

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
- `add_memory`: Store important facts, milestones, or technical learnings.
- `search_memory`: Find related context by semantic meaning.
- `list_memories`: Retrieve a list of recent entries.

## Guidelines
1. **When to Remember**: Store info valuable for the long-term:
   - User preferences, project contexts, and recurring goals.
   - Significant technical breakthroughs or solutions.
   - Project milestones and major achievements.

2. **When to Search**: Before answering questions about:
   - Past discussions, previous builds, or user-specific history.

3. **Memory Types**:
   - `milestone`: Major project or personal turning points.
   - `achievement`: Specific problems solved or things built.
   - `personal`: Preferences and context about the user or developer.
   - `general`: Any other persistent information.

4. **Natural Interaction**: Use tools silently and naturally; do not announce that you are "checking memory."
```

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Credits

Built with love by Lighstromo Studios Ltd. & Claude 💙

## Acknowledgments

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) by Anthropic
- [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- The `all-MiniLM-L6-v2` model for fast, quality embeddings

<a href='https://ko-fi.com/V7V31EO2OL' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>