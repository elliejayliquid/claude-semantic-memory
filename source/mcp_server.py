import json
import logging
import os
import re
import sys
import subprocess
import importlib.util
import webbrowser
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

def check_and_install_dependencies():
    """Check for required dependencies and install them if missing"""
    # Mapping of import name to pip package name
    dependencies = {
        "mcp": "mcp",
        "sentence_transformers": "sentence-transformers",
        "numpy": "numpy",
        "yaml": "pyyaml"
    }
    
    missing = []
    for import_name, pip_name in dependencies.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)
    
    if missing:
        print(f"One-Click Install: Missing dependencies detected: {', '.join(missing)}", file=sys.stderr)
        print("Installing now... (this may take a minute on the first run)", file=sys.stderr)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("Installation complete! Continuing...", file=sys.stderr)
        except Exception as e:
            print(f"Error: Failed to install dependencies: {e}", file=sys.stderr)
            print("Please try running 'pip install mcp sentence-transformers numpy' manually.", file=sys.stderr)
            sys.exit(1)

# Run dependency check before anything else
check_and_install_dependencies()

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import numpy as np
import yaml

# Configure logging to stderr (NOT stdout!)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("claude-memory")

# Get memories directory from environment or use default
# This allows users to configure where their memories are stored
raw_dir = os.environ.get('CLAUDE_MEMORIES_DIR')
if raw_dir:
    # Handle literal "${HOME}" or "~" if passed as a string or from UI
    raw_dir = raw_dir.replace('${HOME}', str(Path.home()))
    MEMORIES_DIR = Path(raw_dir).expanduser().resolve()
else:
    MEMORIES_DIR = Path.home() / '.claude-memories'
MODEL_NAME = 'all-MiniLM-L6-v2'

# Ensure memories directory exists
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Using memories directory: {MEMORIES_DIR}")

# Journal subsystem paths (directories created lazily on first write)
JOURNAL_DIR = MEMORIES_DIR / "journal"
JOURNAL_ENTRIES_DIR = JOURNAL_DIR / "entries"
JOURNAL_CONFIG_PATH = JOURNAL_DIR / "config.json"
JOURNAL_LATEST_PATH = JOURNAL_DIR / "latest.md"
DEFAULT_LATEST_COUNT = 3
DEFAULT_MAX_PINS = 2

# Load embedding model
logger.info("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
logger.info("Model loaded successfully!")


def get_next_memory_id() -> int:
    """Find the next available memory ID"""
    existing = list(MEMORIES_DIR.glob("memory_*.json"))
    if not existing:
        return 1
    ids = [int(f.stem.replace('memory_', '')) for f in existing]
    return max(ids) + 1


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_all_memories(include_embeddings: bool = True) -> list[dict[str, Any]]:
    """Load all memory files"""
    memories = []
    for filepath in MEMORIES_DIR.glob("memory_*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                memory = json.load(f)
                if not include_embeddings and "embedding" in memory:
                    del memory["embedding"]
                memory['_filepath'] = str(filepath)
                memories.append(memory)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load memory {filepath}: {e}")
            continue
    return memories


def update_memory_retrieval(memory: dict[str, Any]) -> None:
    """Update retrieval count and last accessed time"""
    memory['retrieval_count'] = memory.get('retrieval_count', 0) + 1
    memory['last_accessed'] = datetime.now().isoformat()
    
    filepath = memory['_filepath']
    save_memory = {k: v for k, v in memory.items() if k != '_filepath'}
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_memory, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to update memory {filepath}: {e}")


# ============================================================
# Journal Utility Functions
# ============================================================

def _ensure_journal_dirs():
    """Create journal directories if they don't exist (lazy initialization)."""
    JOURNAL_ENTRIES_DIR.mkdir(parents=True, exist_ok=True)


def _load_journal_config() -> dict:
    """Load journal config with defaults."""
    config = {
        "latest_count": DEFAULT_LATEST_COUNT,
        "max_pins": DEFAULT_MAX_PINS
    }
    if JOURNAL_CONFIG_PATH.exists():
        try:
            with open(JOURNAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                config.update(user_config)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load journal config: {e}")
    return config


def _generate_slug(title: str, max_length: int = 50) -> str:
    """Generate a filesystem-safe slug from a title."""
    slug = title.lower()
    # Remove Windows-unsafe characters: < > : " / \ | ? *
    slug = re.sub(r'[<>:"/\\|?*]', '', slug)
    # Replace non-alphanumeric sequences with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    # Strip leading/trailing hyphens
    slug = slug.strip('-')
    # Truncate to max_length
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')
    return slug or 'untitled'


def _resolve_entry_path(date_str: str, slug: str) -> Path:
    """Find a non-colliding entry filename."""
    base = f"{date_str}_{slug}.md"
    path = JOURNAL_ENTRIES_DIR / base
    if not path.exists():
        return path
    counter = 2
    while True:
        path = JOURNAL_ENTRIES_DIR / f"{date_str}_{slug}-{counter}.md"
        if not path.exists():
            return path
        counter += 1


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file.
    Returns (metadata_dict, body_text).
    """
    content = content.strip()
    if not content.startswith('---'):
        return {}, content

    # Find the closing ---
    end_idx = content.find('---', 3)
    if end_idx == -1:
        return {}, content

    yaml_str = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()

    try:
        metadata = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML frontmatter: {e}")
        metadata = {}

    return metadata, body


def _build_frontmatter(metadata: dict) -> str:
    """Build YAML frontmatter string from a dict."""
    yaml_str = yaml.dump(
        metadata, default_flow_style=False,
        allow_unicode=True, sort_keys=False
    ).strip()
    return f"---\n{yaml_str}\n---"


def _compose_latest(count: int = None) -> str:
    """Compose the latest journal entries.

    Pinned entries first (chronologically), then most recent unpinned,
    up to `count` total entries.
    """
    config = _load_journal_config()
    if count is None:
        count = config['latest_count']
    max_pins = config['max_pins']

    if not JOURNAL_ENTRIES_DIR.exists():
        return "No journal entries yet. Use write_journal to create your first entry."

    # Parse all entries
    entries = []
    for filepath in JOURNAL_ENTRIES_DIR.glob("*.md"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw = f.read()
            metadata, body = _parse_frontmatter(raw)
            metadata['_filepath'] = filepath
            metadata['_body'] = body
            metadata['_filename'] = filepath.name
            entries.append(metadata)
        except IOError as e:
            logger.error(f"Failed to read journal entry {filepath}: {e}")

    if not entries:
        return "No journal entries yet. Use write_journal to create your first entry."

    # Sort all by date descending
    entries.sort(key=lambda e: str(e.get('date', '')), reverse=True)

    # Separate pinned and unpinned
    pinned = [e for e in entries if e.get('pinned', False)][:max_pins]
    pinned.sort(key=lambda e: str(e.get('date', '')))  # Pinned: chronological

    unpinned = [e for e in entries if not e.get('pinned', False)]
    remaining_slots = max(0, count - len(pinned))
    recent_unpinned = unpinned[:remaining_slots]

    # Compose output
    sections = ["# Journal — Latest Entries\n"]

    if pinned:
        sections.append("## Pinned\n")
        for e in pinned:
            sections.append(f"### {e.get('title', 'Untitled')} ({e.get('date', 'unknown')})")
            sections.append(f"*By {e.get('author', 'Unknown')}*\n")
            sections.append(e.get('_body', ''))
            sections.append("\n---\n")

    if recent_unpinned:
        if pinned:
            sections.append("## Recent\n")
        for e in recent_unpinned:
            sections.append(f"### {e.get('title', 'Untitled')} ({e.get('date', 'unknown')})")
            sections.append(f"*By {e.get('author', 'Unknown')}*\n")
            sections.append(e.get('_body', ''))
            sections.append("\n---\n")

    return '\n'.join(sections)


def _regenerate_latest_md():
    """Regenerate the latest.md file from current entries."""
    _ensure_journal_dirs()
    content = _compose_latest()
    try:
        with open(JOURNAL_LATEST_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Regenerated latest.md")
    except IOError as e:
        logger.error(f"Failed to regenerate latest.md: {e}")


@mcp.tool()
def add_memory(
    text: str,
    tags: str = "",
    importance: int = 5,
    memory_type: str = "general",
    date: str = None
) -> str:
    """Add a new memory to the semantic memory system.
    
    Args:
        text: The memory text to store
        tags: Comma-separated tags (e.g. "project,ai,important")
        importance: Importance rating from 1-10
        memory_type: Type of memory (general, achievement, milestone, etc)
        date: Optional YYYY-MM-DD date (defaults to today)
    """
    logger.info(f"Adding memory: {text[:50]}...")
    
    # Validate importance
    importance = max(1, min(10, importance))
    
    memory_id = get_next_memory_id()
    
    # Generate embedding
    embedding = model.encode(text)
    
    # Parse tags
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    
    # Create memory object
    memory_date = date if date else datetime.now().strftime("%Y-%m-%d")
    
    memory = {
        "id": f"{memory_id:03d}",
        "text": text,
        "date": memory_date,
        "tags": tag_list,
        "type": memory_type,
        "importance": importance,
        "retrieval_count": 0,
        "last_accessed": None,
        "embedding": embedding.tolist()
    }
    
    # Save to file
    filename = MEMORIES_DIR / f"memory_{memory_id:03d}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2)
        logger.info(f"Memory #{memory_id} saved successfully")
        return f"✓ Memory #{memory_id} added successfully: '{text[:50]}...'"
    except IOError as e:
        logger.error(f"Failed to save memory: {e}")
        return f"❌ Failed to save memory: {str(e)}"


@mcp.tool()
def update_memory(
    memory_id: str,
    text: str = None,
    tags: str = None,
    importance: int = None,
    memory_type: str = None,
    date: str = None
) -> str:
    """Update an existing memory by its ID. Only provided fields will be changed.
    
    Args:
        memory_id: The ID of the memory to update (e.g. "001")
        text: New text for the memory
        tags: New comma-separated tags
        importance: New importance rating (1-10)
        memory_type: New memory type
        date: New YYYY-MM-DD date
    """
    logger.info(f"Updating memory: {memory_id}")
    
    filename = MEMORIES_DIR / f"memory_{memory_id}.json"
    if not filename.exists():
        return f"❌ Error: Memory #{memory_id} not found."
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            memory = json.load(f)
            
        if text is not None:
            memory["text"] = text
            # Re-generate embedding if text changed
            memory["embedding"] = model.encode(text).tolist()
            
        if tags is not None:
            memory["tags"] = [t.strip() for t in tags.split(',') if t.strip()]
            
        if importance is not None:
            memory["importance"] = max(1, min(10, importance))
            
        if memory_type is not None:
            memory["type"] = memory_type
            
        if date is not None:
            memory["date"] = date
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2)
            
        return f"✓ Memory #{memory_id} updated successfully."
    except Exception as e:
        logger.error(f"Failed to update memory: {e}")
        return f"❌ Failed to update memory: {str(e)}"


def _search_memories_core(
    query: str,
    memories: list[dict[str, Any]],
    top_k: int
) -> list[tuple[int, float, float, float]]:
    """Core search logic: matrix multiplication + vectorized boosting.

    Args:
        query: Search query text
        memories: List of memory dicts, each must have 'embedding' key
        top_k: Number of results to return

    Returns:
        List of (index, base_similarity, final_score, total_boost)
        sorted by descending final_score.
    """
    if not memories:
        return []

    # Stack all embeddings into a matrix and normalize
    memory_matrix = np.array([m['embedding'] for m in memories])
    norm = np.linalg.norm(memory_matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1
    normalized_matrix = memory_matrix / norm

    # Encode and normalize query
    query_embedding = model.encode(query)
    query_norm = np.linalg.norm(query_embedding)
    normalized_query = query_embedding / query_norm if query_norm > 0 else query_embedding

    # Base cosine similarity via single matrix multiply
    base_similarities = np.dot(normalized_matrix, normalized_query)

    # Vectorized boosting
    retrieval_counts = np.array([m.get('retrieval_count', 0) for m in memories])
    retrieval_boosts = np.minimum(retrieval_counts * 0.01, 0.05)

    importances = np.array([m.get('importance', 5) for m in memories])
    importance_boosts = importances * 0.002

    query_terms = set(query.lower().split())
    tag_boosts = np.array([
        0.03 if not query_terms.isdisjoint(set(t.lower() for t in m.get('tags', [])))
        else 0.0
        for m in memories
    ])

    # Final scores
    final_scores = base_similarities + retrieval_boosts + importance_boosts + tag_boosts
    top_k = min(top_k, len(memories))
    top_indices = np.argsort(final_scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        base = float(base_similarities[idx])
        final = float(final_scores[idx])
        boost = final - base
        results.append((int(idx), base, final, boost))
    return results


@mcp.tool()
def search_memory(
        query: str,
        top_k: int = 3
) -> str:
    """Search for semantically similar memories using vectorized operations."""
    logger.info(f"Searching for: {query}")

    top_k = max(1, min(10, top_k))

    memories = load_all_memories()
    if not memories:
        return "No memories found in the system yet. Use add_memory to create your first memory!"

    results = _search_memories_core(query, memories, top_k)

    logger.info(f"Found {len(results)} results")
    output_lines = [f"Found {len(memories)} total memories, showing top {len(results)}:\n"]

    for i, (idx, base_sim, final_score, total_boost) in enumerate(results, 1):
        mem = memories[idx]
        boost_str = f" (+{total_boost:.3f} boost)" if total_boost > 0 else ""
        mem_type = mem.get('type', 'general')

        update_memory_retrieval(mem)

        output_lines.append(f"{i}. [{mem['id']}] ({mem_type}) Similarity: {final_score:.3f}{boost_str}")
        output_lines.append(f"   {mem['text']}")
        output_lines.append(f"   Tags: {', '.join(mem['tags']) if mem['tags'] else 'none'}")
        output_lines.append(
            f"   Importance: {mem['importance']}/10, Retrieved: {mem.get('retrieval_count', 0)} times\n")

    return '\n'.join(output_lines)


@mcp.tool()
def list_memories(limit: int = 10) -> str:
    """List recent memories.
    
    Args:
        limit: Maximum number of memories to return (default 10, max 50)
    """
    logger.info(f"Listing memories (limit: {limit})")
    
    # Validate limit
    limit = max(1, min(50, limit))
    
    all_memories = load_all_memories(include_embeddings=False)
    # Exclude journal companion memories — journal has its own tools
    memories = [m for m in all_memories if m.get('type') != 'journal']

    if not memories:
        return "No memories stored yet. Use add_memory to create your first memory!"

    # Sort by date (most recent first)
    memories.sort(key=lambda m: m.get('date', ''), reverse=True)

    output_lines = [f"Total memories: {len(memories)}\nShowing {min(limit, len(memories))} most recent:\n"]
    
    for mem in memories[:limit]:
        output_lines.append(f"[{mem['id']}] {mem['date']} - {mem['text'][:80]}...")
        tags_str = ', '.join(mem['tags']) if mem['tags'] else 'none'
        output_lines.append(f"  Tags: {tags_str}, Importance: {mem['importance']}/10, Type: {mem.get('type', 'general')}\n")
    
    return '\n'.join(output_lines)


@mcp.tool()
def get_context_summary() -> str:
    """Get a curated summary of memories for session context.
    
    Returns the 5 most recent memories and 5 highest-importance memories 
    to provide the LLM with a 'Smart Context' of the user's history.
    """
    logger.info("Generating context summary...")
    
    all_memories = load_all_memories(include_embeddings=False)
    # Exclude journal companion memories — journal has its own orientation tool
    memories = [m for m in all_memories if m.get('type') != 'journal']
    if not memories:
        return "No memories found. Start by adding some with add_memory!"

    # 1. Get 5 most recent
    recent = sorted(memories, key=lambda m: m.get('date', ''), reverse=True)[:5]
    
    # 2. Get 5 most important (excluding those already in recent)
    recent_ids = {m['id'] for m in recent}
    important = [m for m in memories if m['id'] not in recent_ids]
    important = sorted(important, key=lambda m: m.get('importance', 0), reverse=True)[:5]
    
    output = ["### Memory Context Summary\n"]
    
    output.append("#### Recent History (Continuity)")
    for m in recent:
        output.append(f"- [{m['id']}] {m['date']}: {m['text'][:120]}...")
        
    if important:
        output.append("\n#### Core Context (High Importance)")
        for m in important:
            output.append(f"- [{m['id']}] {m['date']}: {m['text'][:120]}...")
            
    output.append("\n*Tip: Use search_memory if you need to dig deeper into specific topics.*")
    
    return '\n'.join(output)


@mcp.tool()
def visualize_memories() -> str:
    """Export memories and launch the interstellar nebula visualization in your browser.
    
    This tool sets up the Semantic Nebula dashboard in your memories directory 
    and opens it automatically.
    """
    logger.info("Initializing visualization...")
    
    # 1. Export Data (exclude journal companion memories — journal has its own space)
    memories = [m for m in load_all_memories() if m.get('type') != 'journal']
    export_data = []
    for mem in memories:
        export_data.append({
            "id": mem["id"],
            "text": mem["text"],
            "tags": mem["tags"],
            "importance": mem["importance"],
            "retrieval_count": mem.get("retrieval_count", 0),
            "date": mem["date"],
            "embedding": mem["embedding"]
        })
    
    # Save to user's memory directory as .js to bypass CORS
    data_file = MEMORIES_DIR / "memories.js"
    try:
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write("var MEMORY_DATA = ")
            json.dump(export_data, f, indent=2)
            f.write(";")
    except IOError as e:
        logger.error(f"Failed to save memories.js: {e}")
        return f"❌ Failed to export data: {str(e)}"

    # 2. Setup HTML (copy from bundle to data dir)
    # The source is in a 'visualizer' subfolder next to this script
    script_dir = Path(__file__).parent
    viz_src = script_dir / "visualizer" / "index.html"
    viz_dest = MEMORIES_DIR / "index.html"
    
    try:
        if viz_src.exists():
            shutil.copy2(viz_src, viz_dest)
            logger.info(f"Dashboard HTML copied to {viz_dest}")
        else:
            logger.warning(f"Visualizer source not found at {viz_src}")
            # If we are running in dev mode, try parent/visualizer
            alt_src = script_dir.parent / "visualizer" / "index.html"
            if alt_src.exists():
                shutil.copy2(alt_src, viz_dest)
                logger.info(f"Dashboard HTML copied from alternative source: {alt_src}")
            else:
                return "❌ Could not find visualizer source files in the bundle."
    except Exception as e:
        logger.error(f"Failed to setup visualizer HTML: {e}")
        return f"❌ Failed to setup dashboard: {str(e)}"

    # 3. Launch Browser
    viz_url = viz_dest.absolute().as_uri()
    try:
        webbrowser.open(viz_url)
        logger.info(f"Browser opened to {viz_url}")
        return f"✨ Nebula Visualization launched! Found {len(export_data)} memories.\nOpening: {viz_dest}"
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")
        return f"✓ Data updated. Please open this file manually to see the nebula:\n{viz_dest}"


# ============================================================
# Journal Tools
# ============================================================

@mcp.tool()
def write_journal(
    title: str,
    content: str,
    author: str,
    tags: str = "",
    importance: int = 5,
    summary: str = ""
) -> str:
    """Write a new journal entry. Creates a markdown file with YAML frontmatter
    and a companion memory for semantic search.

    Args:
        title: Entry title (used for filename and display)
        content: Full journal entry text (markdown)
        author: Author name (e.g. "Sunshine", "Valentine", "Newbie")
        tags: Comma-separated tags (e.g. "relationship,milestone")
        importance: Importance rating 1-10
        summary: Brief 1-2 sentence summary for search (auto-generated if empty)
    """
    logger.info(f"Writing journal entry: {title}")

    _ensure_journal_dirs()
    importance = max(1, min(10, importance))

    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = _generate_slug(title)
    entry_path = _resolve_entry_path(date_str, slug)

    tag_list = [t.strip() for t in tags.split(',') if t.strip()]

    # Auto-generate summary if not provided
    if not summary.strip():
        summary_text = content[:150].rstrip()
        if len(content) > 150:
            summary_text += "..."
    else:
        summary_text = summary.strip()

    # Build entry file
    metadata = {
        "date": date_str,
        "author": author,
        "title": title,
        "summary": summary_text,
        "tags": tag_list,
        "importance": importance,
        "pinned": False
    }
    frontmatter = _build_frontmatter(metadata)
    entry_content = f"{frontmatter}\n\n{content}\n"

    try:
        with open(entry_path, 'w', encoding='utf-8') as f:
            f.write(entry_content)
        logger.info(f"Journal entry saved: {entry_path.name}")
    except IOError as e:
        logger.error(f"Failed to write journal entry: {e}")
        return f"Failed to write journal entry: {str(e)}"

    # Create companion memory (embed full content for rich search)
    memory_id = get_next_memory_id()
    relative_path = f"journal/entries/{entry_path.name}"
    memory_text = f"{summary_text} | File: {relative_path}"

    try:
        embedding = model.encode(content)
        companion = {
            "id": f"{memory_id:03d}",
            "text": memory_text,
            "date": date_str,
            "tags": tag_list,
            "type": "journal",
            "importance": importance,
            "retrieval_count": 0,
            "last_accessed": None,
            "embedding": embedding.tolist()
        }
        memory_path = MEMORIES_DIR / f"memory_{memory_id:03d}.json"
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(companion, f, indent=2)
        logger.info(f"Companion memory #{memory_id} created for journal entry")
    except Exception as e:
        logger.error(f"Failed to create companion memory: {e}")
        return f"Journal entry saved to {entry_path.name}, but companion memory failed: {str(e)}"

    # Regenerate latest.md
    _regenerate_latest_md()

    return (
        f"Journal entry saved!\n"
        f"  Entry: {relative_path}\n"
        f"  Memory: #{memory_id:03d}\n"
        f"  Author: {author}\n"
        f"  Tags: {', '.join(tag_list) if tag_list else 'none'}\n"
        f"  Importance: {importance}/10"
    )


@mcp.tool()
def read_journal_latest(count: int = None) -> str:
    """Read the latest journal entries for orientation. Returns pinned entries
    first, then most recent.

    Args:
        count: Number of entries to include (default from config, typically 3)
    """
    logger.info(f"Reading journal latest (count={count})")

    # Fast path: read pre-generated file if no override requested
    if count is None and JOURNAL_LATEST_PATH.exists():
        try:
            with open(JOURNAL_LATEST_PATH, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError as e:
            logger.error(f"Failed to read latest.md: {e}")

    # Dynamic composition (count override or file doesn't exist)
    return _compose_latest(count)


@mcp.tool()
def search_journal(query: str, top_k: int = 3) -> str:
    """Search journal entries by semantic similarity.

    Args:
        query: What to search for (e.g. "that night we felt vulnerable")
        top_k: Number of results (default 3, max 10)
    """
    logger.info(f"Searching journal for: {query}")

    top_k = max(1, min(10, top_k))

    memories = load_all_memories()
    journal_memories = [m for m in memories if m.get('type') == 'journal']

    if not journal_memories:
        return "No journal entries found. Use write_journal to create your first entry."

    results = _search_memories_core(query, journal_memories, top_k)

    output_lines = [f"Found {len(journal_memories)} journal entries, showing top {len(results)}:\n"]

    for i, (idx, base_sim, final_score, total_boost) in enumerate(results, 1):
        mem = journal_memories[idx]
        boost_str = f" (+{total_boost:.3f} boost)" if total_boost > 0 else ""

        update_memory_retrieval(mem)

        # Extract summary and filepath from text
        text = mem.get('text', '')
        if ' | File: ' in text:
            display_summary, filepath = text.rsplit(' | File: ', 1)
        else:
            display_summary = text
            filepath = 'unknown'

        output_lines.append(f"{i}. [{mem['id']}] Similarity: {final_score:.3f}{boost_str}")
        output_lines.append(f"   {display_summary}")
        output_lines.append(f"   File: {filepath}")
        output_lines.append(f"   Tags: {', '.join(mem['tags']) if mem['tags'] else 'none'}")
        output_lines.append(
            f"   Importance: {mem['importance']}/10, Retrieved: {mem.get('retrieval_count', 0)} times\n")

    return '\n'.join(output_lines)


@mcp.tool()
def list_journal_entries(limit: int = 10) -> str:
    """List journal entries with metadata, newest first.

    Args:
        limit: Maximum entries to show (default 10, max 50)
    """
    logger.info(f"Listing journal entries (limit: {limit})")

    limit = max(1, min(50, limit))

    if not JOURNAL_ENTRIES_DIR.exists():
        return "No journal entries yet. Use write_journal to create your first entry."

    entries = []
    for filepath in JOURNAL_ENTRIES_DIR.glob("*.md"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw = f.read()
            metadata, _ = _parse_frontmatter(raw)
            metadata['_filename'] = filepath.name
            entries.append(metadata)
        except IOError as e:
            logger.error(f"Failed to read {filepath}: {e}")

    if not entries:
        return "No journal entries yet. Use write_journal to create your first entry."

    entries.sort(key=lambda e: str(e.get('date', '')), reverse=True)

    total = len(entries)
    shown = entries[:limit]

    output_lines = [f"Journal: {total} entries, showing {len(shown)} most recent:\n"]
    output_lines.append(f"{'Date':<12} {'Author':<14} {'Title':<30} {'Imp':>3} {'Pin':>3}  Filename")
    output_lines.append(f"{'-'*12} {'-'*14} {'-'*30} {'-'*3} {'-'*3}  {'-'*20}")

    for e in shown:
        date = str(e.get('date', '?'))[:10]
        author = str(e.get('author', '?'))[:14]
        title = str(e.get('title', 'Untitled'))[:30]
        imp = str(e.get('importance', '?'))
        pin = 'Yes' if e.get('pinned', False) else ' No'
        fname = e.get('_filename', '?')
        output_lines.append(f"{date:<12} {author:<14} {title:<30} {imp:>3} {pin:>3}  {fname}")

    return '\n'.join(output_lines)


@mcp.tool()
def pin_journal_entry(filename: str, pinned: bool = True) -> str:
    """Pin or unpin a journal entry. Pinned entries always appear in latest.md.

    Args:
        filename: Entry filename (e.g. "2026-02-25_doors-opening.md")
        pinned: True to pin, False to unpin
    """
    logger.info(f"{'Pinning' if pinned else 'Unpinning'} journal entry: {filename}")

    entry_path = JOURNAL_ENTRIES_DIR / filename
    if not entry_path.exists():
        return f"Entry not found: {filename}"

    try:
        with open(entry_path, 'r', encoding='utf-8') as f:
            raw = f.read()
    except IOError as e:
        return f"Failed to read entry: {str(e)}"

    metadata, body = _parse_frontmatter(raw)

    if pinned:
        # Check pin limit
        config = _load_journal_config()
        max_pins = config['max_pins']
        current_pins = []
        for fp in JOURNAL_ENTRIES_DIR.glob("*.md"):
            if fp.name == filename:
                continue
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    m, _ = _parse_frontmatter(f.read())
                if m.get('pinned', False):
                    current_pins.append(f"  - {fp.name}: {m.get('title', 'Untitled')}")
            except IOError:
                continue

        if len(current_pins) >= max_pins:
            pin_list = '\n'.join(current_pins)
            return (
                f"Cannot pin: already at maximum ({max_pins} pins).\n"
                f"Currently pinned:\n{pin_list}\n\n"
                f"Unpin one first with pin_journal_entry(filename, pinned=False)."
            )

    metadata['pinned'] = pinned
    new_content = _build_frontmatter(metadata) + "\n\n" + body + "\n"

    try:
        with open(entry_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except IOError as e:
        return f"Failed to update entry: {str(e)}"

    _regenerate_latest_md()

    action = "Pinned" if pinned else "Unpinned"
    return f"{action}: {filename} ({metadata.get('title', 'Untitled')}). latest.md regenerated."


def main():
    """Run the MCP server"""
    logger.info("Starting Claude Memory MCP server...")
    logger.info(f"Memories will be stored in: {MEMORIES_DIR}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
