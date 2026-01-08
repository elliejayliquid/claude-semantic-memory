import json
import logging
import os
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
        "numpy": "numpy"
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


def load_all_memories() -> list[dict[str, Any]]:
    """Load all memory files"""
    memories = []
    for filepath in MEMORIES_DIR.glob("memory_*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                memory = json.load(f)
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


@mcp.tool()
def search_memory(
    query: str,
    top_k: int = 3
) -> str:
    """Search for semantically similar memories.
    
    Args:
        query: What to search for
        top_k: Number of results to return (default 3, max 10)
    """
    logger.info(f"Searching for: {query}")
    
    # Validate top_k
    top_k = max(1, min(10, top_k))
    
    # Generate query embedding
    query_embedding = model.encode(query)
    
    # Load all memories
    memories = load_all_memories()
    
    if not memories:
        return "No memories found in the system yet. Use add_memory to create your first memory!"
    
    # Calculate similarities
    results = []
    for memory in memories:
        memory_embedding = np.array(memory['embedding'])
        base_similarity = cosine_similarity(query_embedding, memory_embedding)
        
        # Boost based on retrieval count
        retrieval_count = memory.get('retrieval_count', 0)
        boosted_similarity = base_similarity + min(retrieval_count * 0.01, 0.05)
        
        results.append({
            'memory': memory,
            'similarity': base_similarity,
            'boosted_similarity': boosted_similarity
        })
    
    # Sort by boosted similarity
    results.sort(key=lambda x: x['boosted_similarity'], reverse=True)
    top_results = results[:top_k]
    
    # Update retrieval tracking
    for result in top_results:
        update_memory_retrieval(result['memory'])
    
    logger.info(f"Found {len(top_results)} results")
    
    # Format output
    output_lines = [f"Found {len(memories)} total memories, showing top {len(top_results)}:\n"]
    
    for i, result in enumerate(top_results, 1):
        mem = result['memory']
        sim = result['boosted_similarity']
        boost = sim - result['similarity']
        
        boost_str = f" (+{boost:.3f} boost)" if boost > 0 else ""
        
        output_lines.append(f"{i}. [{mem['id']}] Similarity: {sim:.3f}{boost_str}")
        output_lines.append(f"   {mem['text']}")
        output_lines.append(f"   Tags: {', '.join(mem['tags']) if mem['tags'] else 'none'}")
        output_lines.append(f"   Importance: {mem['importance']}/10, Retrieved: {mem.get('retrieval_count', 0)} times\n")
    
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
    
    memories = load_all_memories()
    
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
def visualize_memories() -> str:
    """Export memories and launch the interstellar nebula visualization in your browser.
    
    This tool sets up the Semantic Nebula dashboard in your memories directory 
    and opens it automatically.
    """
    logger.info("Initializing visualization...")
    
    # 1. Export Data
    memories = load_all_memories()
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


def main():
    """Run the MCP server"""
    logger.info("Starting Claude Memory MCP server...")
    logger.info(f"Memories will be stored in: {MEMORIES_DIR}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
