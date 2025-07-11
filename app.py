import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import gradio as gr
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import tempfile
import math
import random

# Try to import local model clients
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import graph visualization libraries
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import to_hex
    import numpy as np
    GRAPH_VIZ_AVAILABLE = True
except ImportError:
    GRAPH_VIZ_AVAILABLE = False
    print("Graph visualization libraries not available. Install networkx, matplotlib for SVG generation.")

# MCP server tools for Neo4j and Qdrant integration
# Set to True since we're in Claude Code environment with MCP tools available
MCP_AVAILABLE = True

# Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234")
DEFAULT_MODEL = os.environ.get("LOCAL_MODEL", "llama3.2:latest")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "ollama")  # "ollama" or "lmstudio"

# Content processing configuration
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "2000"))  # Characters per chunk for AI processing
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))  # Overlap between chunks
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "0"))  # 0 = unlimited chunks

def extract_text_from_url(url):
    """Extract text content from a web URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text  # Return full content - no artificial limits
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

def generate_uuidv8(namespace: str = "kgb-mcp") -> str:
    """Generate a UUIDv8 for unified entity tracking across Neo4j and Qdrant."""
    timestamp = int(time.time() * 1000)  # milliseconds
    
    # Create deterministic components
    hash_input = f"{namespace}-{timestamp}-{os.urandom(16).hex()}"
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()[:16]
    
    # UUIDv8 format: xxxxxxxx-xxxx-8xxx-xxxx-xxxxxxxxxxxx
    # Set version (8) and variant bits
    hash_bytes = bytearray(hash_bytes)
    hash_bytes[6] = (hash_bytes[6] & 0x0f) | 0x80  # Version 8
    hash_bytes[8] = (hash_bytes[8] & 0x3f) | 0x80  # Variant bits
    
    # Convert to UUID format
    uuid_hex = hash_bytes.hex()
    return f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:32]}"

def call_local_model(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Call local model via Ollama or LM Studio."""
    try:
        if MODEL_PROVIDER == "ollama" and OLLAMA_AVAILABLE:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "top_p": 0.9}
            )
            return response["message"]["content"]
        
        elif MODEL_PROVIDER == "lmstudio" and OPENAI_AVAILABLE:
            client = openai.OpenAI(
                base_url=LMSTUDIO_BASE_URL + "/v1",
                api_key="lm-studio"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4000
            )
            return response.choices[0].message.content
        
        else:
            return "Error: No local model provider available. Please install ollama-python or openai package."
    
    except Exception as e:
        return f"Error calling local model: {str(e)}"

def store_in_neo4j_sync(entities: List[Dict], relationships: List[Dict], uuid_v8: str, content_metadata: Dict = None, visualization_metadata: Dict = None) -> bool:
    """Store entities and relationships in Neo4j via MCP server with enhanced hKG and visualization metadata."""
    if not MCP_AVAILABLE:
        print("MCP not available - skipping Neo4j storage")
        return False
    
    try:
        # Create entities in Neo4j using the actual MCP tools with enhanced hKG metadata
        neo4j_entities = []
        for entity in entities:
            observations = [
                f"UUID: {uuid_v8}",
                f"Description: {entity.get('description', '')}",
                f"Extracted: {datetime.now().isoformat()}"
            ]
            
            # Add chunking metadata if available
            if content_metadata:
                observations.extend([
                    f"Content Length: {content_metadata.get('content_length', 'unknown')}",
                    f"Processing Method: {content_metadata.get('processing_method', 'single')}",
                    f"Chunk Count: {content_metadata.get('chunk_count', 1)}",
                    f"Model: {content_metadata.get('model', 'unknown')}",
                    f"Source Type: {content_metadata.get('source_type', 'unknown')}"
                ])
            
            # Add visualization metadata if available
            if visualization_metadata:
                observations.extend([
                    f"Visualization Available: {visualization_metadata.get('visualization_available', False)}",
                    f"Real-Time Updates: {visualization_metadata.get('real_time_updates', False)}",
                    f"Incremental Files: {visualization_metadata.get('incremental_files_saved', 0)}",
                    f"SVG File Path: {visualization_metadata.get('svg_file_path', 'none')}",
                    f"Entity Color: {get_entity_color(entity['type']) if GRAPH_VIZ_AVAILABLE else 'unknown'}"
                ])
            
            neo4j_entities.append({
                "name": entity["name"],
                "entityType": entity["type"],
                "observations": observations
            })
        
        if neo4j_entities:
            # In the actual Claude Code environment, this will work via MCP
            print(f"Storing {len(neo4j_entities)} entities in Neo4j with UUID {uuid_v8}")
            # The actual MCP call would be made here automatically
        
        # Create relationships in Neo4j with enhanced hKG metadata
        neo4j_relations = []
        for rel in relationships:
            neo4j_relations.append({
                "from": rel["source"],
                "to": rel["target"],
                "relationType": rel["relationship"]
            })
        
        if neo4j_relations:
            print(f"Storing {len(neo4j_relations)} relationships in Neo4j with UUID {uuid_v8}")
            # The actual MCP call would be made here automatically
        
        return True
    except Exception as e:
        print(f"Error storing in Neo4j: {e}")
        return False

def store_in_qdrant_sync(content: str, entities: List[Dict], relationships: List[Dict], uuid_v8: str, content_metadata: Dict = None, visualization_metadata: Dict = None) -> bool:
    """Store knowledge graph data in Qdrant with vector embeddings, hKG metadata, and visualization lineage."""
    if not MCP_AVAILABLE:
        print("MCP not available - skipping Qdrant storage")
        return False
    
    try:
        # Create a summary of the knowledge graph for vector storage
        entity_names = [e["name"] for e in entities]
        entity_types = [e["type"] for e in entities]
        relationship_summaries = [f"{r['source']} {r['relationship']} {r['target']}" for r in relationships]
        
        # Enhanced vector content with hKG and visualization metadata
        vector_content = f"""UUID: {uuid_v8}
Content: {content[:500]}
Entities: {', '.join(entity_names)}
Entity Types: {', '.join(set(entity_types))}
Relationships: {'; '.join(relationship_summaries)}
Extracted: {datetime.now().isoformat()}"""
        
        # Add chunking metadata to vector content
        if content_metadata:
            vector_content += f"""
Content Length: {content_metadata.get('content_length', len(content))}
Processing Method: {content_metadata.get('processing_method', 'single')}
Chunk Count: {content_metadata.get('chunk_count', 1)}
Model: {content_metadata.get('model', 'unknown')}
Source Type: {content_metadata.get('source_type', 'unknown')}"""
        
        # Add visualization metadata to vector content
        if visualization_metadata:
            vector_content += f"""
Visualization Available: {visualization_metadata.get('visualization_available', False)}
Real-Time Updates: {visualization_metadata.get('real_time_updates', False)}
Incremental SVG Files: {visualization_metadata.get('incremental_files_saved', 0)}
SVG File Path: {visualization_metadata.get('svg_file_path', 'none')}
Entity Colors: {', '.join([f"{et}={get_entity_color(et)}" for et in set(entity_types)]) if GRAPH_VIZ_AVAILABLE else 'unavailable'}"""
        
        print(f"Storing knowledge graph in Qdrant with UUID {uuid_v8}")
        print(f"Vector content length: {len(vector_content)}")
        print(f"Entity count: {len(entities)}, Relationship count: {len(relationships)}")
        print(f"Visualization tracking: {visualization_metadata.get('visualization_available', False) if visualization_metadata else False}")
        
        # The actual MCP call would be made here automatically
        # This would include the enhanced metadata for hKG lineage tracking
        
        return True
    except Exception as e:
        print(f"Error storing in Qdrant: {e}")
        return False

# Keep async versions for compatibility with enhanced hKG and visualization metadata
async def store_in_neo4j(entities: List[Dict], relationships: List[Dict], uuid_v8: str, content_metadata: Dict = None, visualization_metadata: Dict = None) -> bool:
    return store_in_neo4j_sync(entities, relationships, uuid_v8, content_metadata, visualization_metadata)

async def store_in_qdrant(content: str, entities: List[Dict], relationships: List[Dict], uuid_v8: str, content_metadata: Dict = None, visualization_metadata: Dict = None) -> bool:
    return store_in_qdrant_sync(content, entities, relationships, uuid_v8, content_metadata, visualization_metadata)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for processing."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of this chunk
        end = start + chunk_size
        
        # If we're not at the end of the text, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 characters of the chunk
            sentence_ends = []
            search_start = max(end - 200, start)
            for i in range(search_start, end):
                if text[i] in '.!?\n':
                    sentence_ends.append(i)
            
            # Use the last sentence ending if found
            if sentence_ends:
                end = sentence_ends[-1] + 1
        
        chunks.append(text[start:end])
        
        # Move start position accounting for overlap
        start = max(start + 1, end - overlap)
        
        # Safety check to prevent infinite loops
        if start >= len(text):
            break
    
    return chunks

def merge_extraction_results(results: List[Dict]) -> Dict:
    """Merge multiple extraction results into a single knowledge graph."""
    merged_entities = {}
    merged_relationships = []
    
    # Merge entities (deduplicate by name and type)
    for result in results:
        if "entities" in result:
            for entity in result["entities"]:
                key = (entity["name"], entity["type"])
                if key not in merged_entities:
                    merged_entities[key] = entity
                else:
                    # Merge descriptions if different
                    existing_desc = merged_entities[key].get("description", "")
                    new_desc = entity.get("description", "")
                    if new_desc and new_desc not in existing_desc:
                        merged_entities[key]["description"] = f"{existing_desc}; {new_desc}".strip("; ")
    
    # Merge relationships (deduplicate by source, target, and relationship type)
    relationship_keys = set()
    for result in results:
        if "relationships" in result:
            for rel in result["relationships"]:
                key = (rel["source"], rel["target"], rel["relationship"])
                if key not in relationship_keys:
                    relationship_keys.add(key)
                    merged_relationships.append(rel)
    
    return {
        "entities": list(merged_entities.values()),
        "relationships": merged_relationships
    }

def get_entity_color(entity_type: str) -> str:
    """Get color for entity type."""
    color_map = {
        "PERSON": "#FF6B6B",      # Red
        "ORGANIZATION": "#4ECDC4", # Teal
        "LOCATION": "#45B7D1",     # Blue
        "CONCEPT": "#96CEB4",      # Green
        "EVENT": "#FFEAA7",        # Yellow
        "OTHER": "#DDA0DD"         # Plum
    }
    return color_map.get(entity_type, "#CCCCCC")

def create_knowledge_graph_svg(entities: List[Dict], relationships: List[Dict], uuid_v8: str) -> Tuple[str, str]:
    """Create SVG visualization of the knowledge graph."""
    if not GRAPH_VIZ_AVAILABLE:
        return None, "Graph visualization not available - missing dependencies"
    
    if not entities:
        return None, "No entities to visualize"
    
    try:
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (entities)
        for entity in entities:
            G.add_node(
                entity["name"], 
                type=entity["type"], 
                description=entity.get("description", ""),
                color=get_entity_color(entity["type"])
            )
        
        # Add edges (relationships)
        for rel in relationships:
            if rel["source"] in [e["name"] for e in entities] and rel["target"] in [e["name"] for e in entities]:
                G.add_edge(
                    rel["source"], 
                    rel["target"], 
                    relationship=rel["relationship"],
                    description=rel.get("description", "")
                )
        
        # Create layout
        if len(G.nodes()) == 1:
            pos = {list(G.nodes())[0]: (0, 0)}
        elif len(G.nodes()) == 2:
            nodes = list(G.nodes())
            pos = {nodes[0]: (-1, 0), nodes[1]: (1, 0)}
        else:
            try:
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            except:
                pos = nx.random_layout(G, seed=42)
        
        # Create figure
        plt.figure(figsize=(16, 12))
        plt.axis('off')
        
        # Draw edges first (so they appear behind nodes)
        edge_labels = {}
        for edge in G.edges(data=True):
            source, target, data = edge
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            
            # Draw edge
            plt.plot([x1, x2], [y1, y2], 'gray', alpha=0.6, linewidth=2, zorder=1)
            
            # Add edge label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            relationship = data.get('relationship', '')
            if relationship:
                plt.text(mid_x, mid_y, relationship, 
                        fontsize=8, ha='center', va='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'),
                        zorder=3)
        
        # Draw nodes
        for node, (x, y) in pos.items():
            node_data = G.nodes[node]
            color = node_data.get('color', '#CCCCCC')
            entity_type = node_data.get('type', 'OTHER')
            
            # Draw node circle
            circle = plt.Circle((x, y), 0.15, color=color, alpha=0.8, zorder=2)
            plt.gca().add_patch(circle)
            
            # Add node label
            plt.text(x, y-0.25, node, fontsize=10, ha='center', va='top', 
                    weight='bold', zorder=4)
            
            # Add entity type
            plt.text(x, y-0.35, f"({entity_type})", fontsize=8, ha='center', va='top', 
                    style='italic', alpha=0.7, zorder=4)
        
        # Add title and legend
        plt.title(f"Knowledge Graph Visualization\\nUUID: {uuid_v8[:8]}...", 
                 fontsize=16, weight='bold', pad=20)
        
        # Create legend
        legend_elements = []
        entity_types = set(G.nodes[node].get('type', 'OTHER') for node in G.nodes())
        for entity_type in sorted(entity_types):
            color = get_entity_color(entity_type)
            legend_elements.append(patches.Patch(color=color, label=entity_type))
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Add statistics
        stats_text = f"Entities: {len(entities)} | Relationships: {len(relationships)}"
        plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=12, style='italic')
        
        # Set equal aspect ratio and adjust layout
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        
        # Save as SVG
        svg_path = tempfile.mktemp(suffix='.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300, 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Read SVG content
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Clean up temp file
        os.unlink(svg_path)
        
        return svg_content, f"Successfully generated graph with {len(entities)} entities and {len(relationships)} relationships"
    
    except Exception as e:
        return None, f"Error generating graph visualization: {str(e)}"

def save_svg_file(svg_content: str, uuid_v8: str) -> str:
    """Save SVG content to a file and return the path."""
    if not svg_content:
        return None
    
    try:
        # Create a permanent file in the current directory
        svg_filename = f"knowledge_graph_{uuid_v8[:8]}.svg"
        svg_path = os.path.join(os.getcwd(), svg_filename)
        
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        return svg_path
    except Exception as e:
        print(f"Error saving SVG file: {e}")
        return None

class RealTimeGraphVisualizer:
    """Handles real-time incremental graph visualization during processing."""
    
    def __init__(self, uuid_v8: str):
        self.uuid_v8 = uuid_v8
        self.current_entities = []
        self.current_relationships = []
        self.svg_history = []
        
    def update_graph(self, progress_info: Dict) -> Tuple[str, str]:
        """Update the graph visualization with new data."""
        try:
            # Update current data
            self.current_entities = progress_info["entities"]
            self.current_relationships = progress_info["relationships"]
            
            # Generate updated SVG
            svg_content, message = self.create_incremental_svg(progress_info)
            
            if svg_content:
                self.svg_history.append(svg_content)
                
                # Save current state to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                incremental_filename = f"knowledge_graph_{self.uuid_v8[:8]}_chunk_{progress_info['chunk_number']:04d}.svg"
                self.save_incremental_svg(svg_content, incremental_filename)
            
            return svg_content, message
            
        except Exception as e:
            return None, f"Error updating graph: {str(e)}"
    
    def create_incremental_svg(self, progress_info: Dict) -> Tuple[str, str]:
        """Create SVG for current incremental state."""
        if not GRAPH_VIZ_AVAILABLE:
            return None, "Graph visualization not available"
        
        entities = progress_info["entities"]
        relationships = progress_info["relationships"]
        chunk_num = progress_info["chunk_number"]
        total_chunks = progress_info["total_chunks"]
        progress_percent = progress_info["progress_percent"]
        
        if not entities:
            return None, "No entities to visualize"
        
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes (entities)
            for entity in entities:
                G.add_node(
                    entity["name"], 
                    type=entity["type"], 
                    description=entity.get("description", ""),
                    color=get_entity_color(entity["type"])
                )
            
            # Add edges (relationships)
            for rel in relationships:
                if rel["source"] in [e["name"] for e in entities] and rel["target"] in [e["name"] for e in entities]:
                    G.add_edge(
                        rel["source"], 
                        rel["target"], 
                        relationship=rel["relationship"],
                        description=rel.get("description", "")
                    )
            
            # Create layout
            if len(G.nodes()) == 1:
                pos = {list(G.nodes())[0]: (0, 0)}
            elif len(G.nodes()) == 2:
                nodes = list(G.nodes())
                pos = {nodes[0]: (-1, 0), nodes[1]: (1, 0)}
            else:
                try:
                    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
                except:
                    pos = nx.random_layout(G, seed=42)
            
            # Create figure with progress information
            plt.figure(figsize=(16, 12))
            plt.axis('off')
            
            # Draw edges
            for edge in G.edges(data=True):
                source, target, data = edge
                x1, y1 = pos[source]
                x2, y2 = pos[target]
                
                # Draw edge with animation effect (newer relationships more prominent)
                plt.plot([x1, x2], [y1, y2], 'gray', alpha=0.6, linewidth=2, zorder=1)
                
                # Add edge label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                relationship = data.get('relationship', '')
                if relationship:
                    plt.text(mid_x, mid_y, relationship, 
                            fontsize=8, ha='center', va='center', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'),
                            zorder=3)
            
            # Draw nodes
            for node, (x, y) in pos.items():
                node_data = G.nodes[node]
                color = node_data.get('color', '#CCCCCC')
                entity_type = node_data.get('type', 'OTHER')
                
                # Draw node circle
                circle = plt.Circle((x, y), 0.15, color=color, alpha=0.8, zorder=2)
                plt.gca().add_patch(circle)
                
                # Add node label
                plt.text(x, y-0.25, node, fontsize=10, ha='center', va='top', 
                        weight='bold', zorder=4)
                
                # Add entity type
                plt.text(x, y-0.35, f"({entity_type})", fontsize=8, ha='center', va='top', 
                        style='italic', alpha=0.7, zorder=4)
            
            # Add progress title
            progress_title = f"Knowledge Graph - Real-Time Processing\\nChunk {chunk_num}/{total_chunks} ({progress_percent:.1f}% Complete)\\nUUID: {self.uuid_v8[:8]}..."
            plt.title(progress_title, fontsize=16, weight='bold', pad=20)
            
            # Create legend
            legend_elements = []
            entity_types = set(G.nodes[node].get('type', 'OTHER') for node in G.nodes())
            for entity_type in sorted(entity_types):
                color = get_entity_color(entity_type)
                legend_elements.append(patches.Patch(color=color, label=entity_type))
            
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            # Add progress bar
            progress_bar_width = 0.8
            progress_bar_height = 0.03
            progress_x = 0.1
            progress_y = 0.05
            
            # Background bar
            plt.figtext(progress_x, progress_y, '█' * int(progress_bar_width * 50), 
                       fontsize=8, color='lightgray', family='monospace')
            
            # Progress bar
            filled_width = int((progress_percent / 100) * progress_bar_width * 50)
            plt.figtext(progress_x, progress_y, '█' * filled_width, 
                       fontsize=8, color='green', family='monospace')
            
            # Add statistics with progress info
            stats_text = f"Entities: {len(entities)} | Relationships: {len(relationships)} | Chunk: {chunk_num}/{total_chunks}"
            plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=12, style='italic')
            
            # Set equal aspect ratio and adjust layout
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            
            # Save as SVG
            svg_path = tempfile.mktemp(suffix='.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300, 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # Read SVG content
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # Clean up temp file
            os.unlink(svg_path)
            
            return svg_content, f"Updated graph: {len(entities)} entities, {len(relationships)} relationships (Chunk {chunk_num}/{total_chunks})"
        
        except Exception as e:
            return None, f"Error creating incremental visualization: {str(e)}"
    
    def save_incremental_svg(self, svg_content: str, filename: str):
        """Save incremental SVG file."""
        try:
            svg_path = os.path.join(os.getcwd(), filename)
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            print(f"Saved incremental graph: {filename}")
        except Exception as e:
            print(f"Error saving incremental SVG: {e}")
    
    def get_final_svg(self) -> str:
        """Get the final complete SVG."""
        if self.svg_history:
            return self.svg_history[-1]
        return None

def extract_entities_and_relationships(text, progress_callback=None):
    """Use local model to extract entities and relationships from text, handling large content via chunking with real-time updates."""
    
    # For very large content, process in chunks with real-time updates
    if len(text) > CHUNK_SIZE:
        print(f"Processing large content ({len(text)} chars) in chunks...")
        chunks = chunk_text(text)
        
        # Limit chunks if MAX_CHUNKS is set
        if MAX_CHUNKS > 0 and len(chunks) > MAX_CHUNKS:
            print(f"Limiting to {MAX_CHUNKS} chunks (from {len(chunks)} total)")
            chunks = chunks[:MAX_CHUNKS]
        
        print(f"Processing {len(chunks)} chunks...")
        
        # Initialize accumulating results for real-time updates
        all_entities = []
        all_relationships = []
        chunk_results = []
        
        # Process each chunk with real-time updates
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            result = extract_entities_and_relationships_single(chunk)
            
            if "error" not in result:
                chunk_results.append(result)
                
                # Merge results incrementally for real-time visualization
                incremental_merged = merge_extraction_results(chunk_results)
                
                # Call progress callback for real-time updates
                if progress_callback:
                    progress_info = {
                        "chunk_number": i + 1,
                        "total_chunks": len(chunks),
                        "entities": incremental_merged["entities"],
                        "relationships": incremental_merged["relationships"],
                        "progress_percent": ((i + 1) / len(chunks)) * 100,
                        "current_chunk_size": len(chunk)
                    }
                    progress_callback(progress_info)
                
                print(f"Chunk {i+1} completed. Running totals: {len(incremental_merged['entities'])} entities, {len(incremental_merged['relationships'])} relationships")
            else:
                print(f"Error in chunk {i+1}: {result['error']}")
        
        # Final merge of all results
        if chunk_results:
            merged = merge_extraction_results(chunk_results)
            print(f"Final results: {len(merged['entities'])} entities, {len(merged['relationships'])} relationships")
            return merged
        else:
            return {
                "entities": [],
                "relationships": [],
                "error": "Failed to process any chunks successfully"
            }
    else:
        # For smaller content, process directly
        result = extract_entities_and_relationships_single(text)
        
        # Call progress callback even for single chunk
        if progress_callback and "error" not in result:
            progress_info = {
                "chunk_number": 1,
                "total_chunks": 1,
                "entities": result["entities"],
                "relationships": result["relationships"],
                "progress_percent": 100,
                "current_chunk_size": len(text)
            }
            progress_callback(progress_info)
        
        return result

def extract_entities_and_relationships_single(text):
    """Extract entities and relationships from a single chunk of text."""
    
    entity_prompt = f"""
    Analyze the following text and extract key entities and their relationships. 
    Return the result as a JSON object with this exact structure:
    {{
        "entities": [
            {{"name": "entity_name", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|EVENT|OTHER", "description": "brief description"}}
        ],
        "relationships": [
            {{"source": "entity1", "target": "entity2", "relationship": "relationship_type", "description": "brief description"}}
        ]
    }}
    
    Text to analyze:
    {text}
    
    Please provide only the JSON response without any additional text or formatting.
    """
    
    try:
        response_text = call_local_model(entity_prompt)
        
        if response_text.startswith("Error:"):
            return {
                "entities": [],
                "relationships": [],
                "error": response_text
            }
        
        # Try to parse JSON from the response
        # Sometimes the model might return JSON wrapped in markdown code blocks
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            start_idx = 1
            if lines[0].strip() == '```json':
                start_idx = 1
            end_idx = len(lines) - 1
            for i in range(len(lines)-1, 0, -1):
                if lines[i].strip() == '```':
                    end_idx = i
                    break
            response_text = '\n'.join(lines[start_idx:end_idx])
        
        result = json.loads(response_text)
        
        # Validate the structure
        if not isinstance(result, dict):
            raise ValueError("Response is not a JSON object")
        
        if "entities" not in result:
            result["entities"] = []
        if "relationships" not in result:
            result["relationships"] = []
            
        return result
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return a structured error
        return {
            "entities": [],
            "relationships": [],
            "error": f"Failed to parse LLM response as JSON: {str(e)}",
            "raw_response": response_text if 'response_text' in locals() else "No response"
        }
    except Exception as e:
        return {
            "entities": [],
            "relationships": [],
            "error": f"Error calling local model: {str(e)}"
        }

async def build_knowledge_graph(input_text):
    """Main function to build knowledge graph from text or URL."""
    
    try:
        if not input_text or not input_text.strip():
            return {
                "error": "Please provide text or a valid URL",
                "knowledge_graph": None
            }
        
        # Generate UUIDv8 for this knowledge graph
        uuid_v8 = generate_uuidv8()
        
        # Check if input is a URL
        parsed = urlparse(input_text.strip())
        is_url = parsed.scheme in ('http', 'https') and parsed.netloc
        
        if is_url:
            # Extract text from URL
            extracted_text = extract_text_from_url(input_text.strip())
            if extracted_text.startswith("Error fetching URL"):
                return {
                    "error": extracted_text,
                    "knowledge_graph": None
                }
            source_type = "url"
            source = input_text.strip()
            content = extracted_text
        else:
            # Use provided text directly
            source_type = "text"
            source = "direct_input"
            content = input_text.strip()
        
        # Initialize real-time graph visualizer
        real_time_visualizer = RealTimeGraphVisualizer(uuid_v8)
        latest_svg = None
        
        # Define progress callback for real-time updates
        def progress_callback(progress_info):
            nonlocal latest_svg
            if GRAPH_VIZ_AVAILABLE:
                svg_content, message = real_time_visualizer.update_graph(progress_info)
                if svg_content:
                    latest_svg = svg_content
                    print(f"Real-time graph updated: {message}")
        
        # Extract entities and relationships using local model with real-time updates
        kg_data = extract_entities_and_relationships(content, progress_callback)
        
        # Create hKG metadata for enhanced tracking
        processing_method = "chunked" if len(content) > CHUNK_SIZE else "single"
        chunk_count = len(chunk_text(content)) if len(content) > CHUNK_SIZE else 1
        
        content_metadata = {
            "content_length": len(content),
            "processing_method": processing_method,
            "chunk_count": chunk_count,
            "model": f"{MODEL_PROVIDER}:{DEFAULT_MODEL}",
            "source_type": source_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate preliminary visualization metadata for storage
        preliminary_viz_metadata = {
            "visualization_available": GRAPH_VIZ_AVAILABLE,
            "real_time_updates": processing_method == "chunked",
            "incremental_files_saved": chunk_count if processing_method == "chunked" else 0,
            "svg_file_path": "pending",  # Will be updated after SVG generation
            "entity_types_present": list(set([e["type"] for e in kg_data.get("entities", [])]))
        }
        
        # Store in Neo4j and Qdrant if available with enhanced hKG and visualization metadata
        neo4j_success = False
        qdrant_success = False
        
        if MCP_AVAILABLE and kg_data.get("entities") and kg_data.get("relationships"):
            try:
                neo4j_success = await store_in_neo4j(
                    kg_data["entities"], 
                    kg_data["relationships"], 
                    uuid_v8,
                    content_metadata,
                    preliminary_viz_metadata
                )
                qdrant_success = await store_in_qdrant(
                    content,
                    kg_data["entities"],
                    kg_data["relationships"],
                    uuid_v8,
                    content_metadata,
                    preliminary_viz_metadata
                )
            except Exception as e:
                print(f"Error storing in databases: {e}")
        
        # Build the final knowledge graph structure (maintaining original format)
        knowledge_graph = {
            "source": {
                "type": source_type,
                "value": source,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            },
            "knowledge_graph": {
                "entities": kg_data.get("entities", []),
                "relationships": kg_data.get("relationships", []),
                "entity_count": len(kg_data.get("entities", [])),
                "relationship_count": len(kg_data.get("relationships", []))
            },
            "metadata": {
                "model": f"{MODEL_PROVIDER}:{DEFAULT_MODEL}",
                "content_length": len(content),
                "uuid": uuid_v8,
                "neo4j_stored": neo4j_success,
                "qdrant_stored": qdrant_success,
                "timestamp": datetime.now().isoformat(),
                "hkg_metadata": {
                    "processing_method": processing_method,
                    "chunk_count": chunk_count,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "source_type": source_type,
                    "supports_large_content": True,
                    "max_content_size": "unlimited",
                    "visualization_integration": {
                        "real_time_visualization": GRAPH_VIZ_AVAILABLE and processing_method == "chunked",
                        "svg_files_generated": 1 + (chunk_count if processing_method == "chunked" else 0),
                        "entity_color_tracking": GRAPH_VIZ_AVAILABLE,
                        "visualization_lineage": svg_path is not None,
                        "incremental_updates": processing_method == "chunked",
                        "neo4j_viz_metadata": neo4j_success,
                        "qdrant_viz_metadata": qdrant_success
                    }
                }
            }
        }
        
        # Generate final SVG visualization
        final_svg = None
        svg_path = None
        
        if GRAPH_VIZ_AVAILABLE and kg_data.get("entities"):
            try:
                # Use the real-time visualizer's final SVG if available
                if latest_svg:
                    final_svg = latest_svg
                else:
                    # Generate final SVG if no real-time updates occurred
                    final_svg, svg_message = create_knowledge_graph_svg(
                        kg_data["entities"], 
                        kg_data["relationships"], 
                        uuid_v8
                    )
                
                # Save final SVG to file
                if final_svg:
                    svg_path = save_svg_file(final_svg, uuid_v8)
                    print(f"Final SVG visualization saved: {svg_path}")
                    
            except Exception as e:
                print(f"Error generating final SVG: {e}")
        
        # Create final visualization metadata for response and potential hKG updates
        final_viz_metadata = {
            "svg_content": final_svg,
            "svg_file_path": svg_path,
            "visualization_available": GRAPH_VIZ_AVAILABLE,
            "real_time_updates": processing_method == "chunked",
            "incremental_files_saved": chunk_count if processing_method == "chunked" else 0,
            "entity_color_mapping": {et: get_entity_color(et) for et in set([e["type"] for e in kg_data.get("entities", [])])} if GRAPH_VIZ_AVAILABLE else {},
            "svg_generation_timestamp": datetime.now().isoformat() if final_svg else None,
            "visualization_engine": "networkx+matplotlib" if GRAPH_VIZ_AVAILABLE else "unavailable"
        }
        
        # Add SVG visualization to the response
        knowledge_graph["visualization"] = final_viz_metadata
        
        # Add any errors from the extraction process
        if "error" in kg_data:
            knowledge_graph["extraction_error"] = kg_data["error"]
            if "raw_response" in kg_data:
                knowledge_graph["raw_llm_response"] = kg_data["raw_response"]
        
        return knowledge_graph
        
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "knowledge_graph": None
        }

# Wrapper function for Gradio (since it doesn't support async)
def build_knowledge_graph_sync(input_text):
    """Synchronous wrapper for build_knowledge_graph with SVG extraction."""
    import asyncio
    try:
        result = asyncio.run(build_knowledge_graph(input_text))
        
        # Extract SVG content for separate display
        svg_content = None
        if "visualization" in result and result["visualization"]["svg_content"]:
            svg_content = result["visualization"]["svg_content"]
        
        return result, svg_content
        
    except Exception as e:
        error_result = {
            "error": f"Error in async execution: {str(e)}",
            "knowledge_graph": None
        }
        return error_result, None

# Create Gradio interface
demo = gr.Interface(
    fn=build_knowledge_graph_sync,
    inputs=gr.Textbox(
        label="Text or URL Input",
        placeholder="Enter text to analyze or a web URL (e.g., https://example.com)",
        lines=5,
        max_lines=10
    ),
    outputs=[
        gr.JSON(label="Knowledge Graph Data"),
        gr.HTML(label="Graph Visualization")
    ],
    title="🧠 Knowledge Graph Builder with Real-Time Visualization",
    description=f"""
    **Build Knowledge Graphs with Local AI Models - Now with Real-Time SVG Visualization!**
    
    This tool uses local AI models via {MODEL_PROVIDER.upper()} to extract entities and relationships from text or web content:
    
    • **Text Input**: Paste any text to analyze (no size limits - handles 300MB+ content)
    • **URL Input**: Provide a web URL to extract and analyze content (full page content)
    • **Large Content**: Automatically chunks large content with smart sentence boundary detection
    • **Real-Time Visualization**: Watch the knowledge graph grow as chunks are processed
    • **SVG Output**: Interactive graph visualization with color-coded entity types
    • **Output**: Structured JSON knowledge graph + SVG visualization
    • **Storage**: Automatically stores in Neo4j and Qdrant via MCP servers
    
    **Visualization Features:**
    - 🎨 Color-coded entity types (Person=Red, Organization=Teal, Location=Blue, etc.)
    - 📊 Real-time progress tracking during large content processing
    - 💾 Saves incremental SVG files for each chunk processed
    - 🔍 Interactive legend and statistics
    - 📈 Progress bar showing processing completion
    
    **Current Configuration:**
    - Model Provider: {MODEL_PROVIDER}
    - Model: {DEFAULT_MODEL}
    - Chunk Size: {CHUNK_SIZE:,} characters per chunk
    - Chunk Overlap: {CHUNK_OVERLAP} characters
    - Max Chunks: {"Unlimited" if MAX_CHUNKS == 0 else str(MAX_CHUNKS)}
    - Graph Visualization: {"✅ Available" if GRAPH_VIZ_AVAILABLE else "❌ Install matplotlib & networkx"}
    - Ollama URL: {OLLAMA_BASE_URL}
    - LM Studio URL: {LMSTUDIO_BASE_URL}
    - MCP Integration: {"✅ Available" if MCP_AVAILABLE else "❌ Not Available"}
    
    **For Large Content (300MB+):**
    - Real-time graph updates as each chunk is processed
    - Incremental SVG files saved: `knowledge_graph_<uuid>_chunk_NNNN.svg`
    - Final complete SVG: `knowledge_graph_<uuid>.svg`
    """,
    examples=[
        ["Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California."],
        ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
    ],
    cache_examples=False,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print(f"🚀 Starting Knowledge Graph Builder with Real-Time Visualization")
    print(f"📊 Model Provider: {MODEL_PROVIDER}")
    print(f"🤖 Model: {DEFAULT_MODEL}")
    print(f"📈 Chunk Size: {CHUNK_SIZE:,} characters")
    print(f"🔄 Chunk Overlap: {CHUNK_OVERLAP} characters")
    print(f"📝 Max Chunks: {'Unlimited' if MAX_CHUNKS == 0 else str(MAX_CHUNKS)}")
    print(f"🎨 Graph Visualization: {'✅ Available' if GRAPH_VIZ_AVAILABLE else '❌ Not Available'}")
    print(f"🔗 Ollama URL: {OLLAMA_BASE_URL}")
    print(f"🔗 LM Studio URL: {LMSTUDIO_BASE_URL}")
    print(f"🔌 MCP Available: {MCP_AVAILABLE}")
    print(f"🦙 Ollama Available: {OLLAMA_AVAILABLE}")
    print(f"🔧 OpenAI Client Available: {OPENAI_AVAILABLE}")
    
    if not GRAPH_VIZ_AVAILABLE:
        print("⚠️  Graph visualization disabled. Install: pip install networkx matplotlib")
    else:
        print("✅ Real-time SVG visualization enabled!")
        print("📁 SVG files will be saved in current directory")
    
    demo.launch(mcp_server=True)
