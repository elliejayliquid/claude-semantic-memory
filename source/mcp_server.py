"""
MCP Server for Claude's Semantic Memory System
Exposes memory tools through the Model Context Protocol
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
MEMORIES_DIR = Path(os.environ.get('CLAUDE_MEMORIES_DIR', Path.home() / '.claude-memories'))
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
    memory_type: str = "general"
) -> str:
    """Add a new memory to the semantic memory system.
    
    Args:
        text: The memory text to store
        tags: Comma-separated tags (e.g. "project,ai,important")
        importance: Importance rating from 1-10
        memory_type: Type of memory (general, achievement, milestone, etc)
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
    memory = {
        "id": f"{memory_id:03d}",
        "text": text,
        "date": datetime.now().strftime("%Y-%m-%d"),
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


def main():
    """Run the MCP server"""
    logger.info("Starting Claude Memory MCP server...")
    logger.info(f"Memories will be stored in: {MEMORIES_DIR}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
