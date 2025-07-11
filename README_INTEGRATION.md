# KGB-MCP Integration Guide

## Overview

This enhanced version of KGB-MCP has been modified to:
1. Use local AI models (Ollama/LM Studio) instead of HuggingFace/Mistral hosted models
2. Generate UUIDv8 for unified entity tracking
3. Integrate with Neo4j and Qdrant via MCP servers
4. Maintain backward compatibility with the original JSON output format

## Architecture Changes

### Local Model Integration

The system now supports two local model providers:

#### Ollama Integration
- Uses the `ollama` Python package
- Connects to local Ollama server (default: `http://localhost:11434`)
- Configurable via `OLLAMA_BASE_URL` environment variable

#### LM Studio Integration  
- Uses the `openai` Python package with LM Studio's OpenAI-compatible API
- Connects to local LM Studio server (default: `http://localhost:1234`)
- Configurable via `LMSTUDIO_BASE_URL` environment variable

### Configuration

Set these environment variables to configure the system:

```bash
# Model provider: "ollama" or "lmstudio"
export MODEL_PROVIDER=ollama

# Model name to use
export LOCAL_MODEL=llama3.2:latest

# Custom URLs for dedicated inference servers
export OLLAMA_BASE_URL=http://your-inference-server:11434
export LMSTUDIO_BASE_URL=http://your-inference-server:1234
```

### UUIDv8 Generation

The system now generates UUIDv8 identifiers for each knowledge graph extraction:
- Includes timestamp and namespace components
- Provides unified tracking across Neo4j and Qdrant
- Format: `xxxxxxxx-xxxx-8xxx-xxxx-xxxxxxxxxxxx`

### MCP Integration

#### Neo4j Storage
- Entities are stored with their type, description, and UUID
- Relationships are created between entities
- Each entry includes extraction timestamp and metadata

#### Qdrant Vector Storage
- Knowledge graphs are vectorized and stored for similarity search
- Includes full content summary, entity names, and relationships
- Metadata includes UUID, counts, and timestamps

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up local model server:
   - For Ollama: Install and run Ollama with your chosen model
   - For LM Studio: Install and configure LM Studio with OpenAI-compatible API

3. Configure MCP servers (in Claude Code environment):
   - Neo4j MCP server for graph storage
   - Qdrant MCP server for vector storage

## Usage

### Basic Usage
```python
# The main function maintains the original interface
result = build_knowledge_graph_sync("Your text or URL here")

# Result structure (maintains backward compatibility):
{
    "source": {
        "type": "text|url",
        "value": "input_value",
        "content_preview": "First 200 chars..."
    },
    "knowledge_graph": {
        "entities": [...],
        "relationships": [...],
        "entity_count": N,
        "relationship_count": M
    },
    "metadata": {
        "model": "ollama:llama3.2:latest",
        "content_length": N,
        "uuid": "generated-uuid-v8",
        "neo4j_stored": true,
        "qdrant_stored": true,
        "timestamp": "2025-01-10T..."
    }
}
```

### Running the Application

```bash
python app.py
```

The Gradio interface will show:
- Current model provider and model
- Server URLs for Ollama and LM Studio
- MCP integration status
- Real-time storage confirmation

## Testing

Run the core functionality tests:
```bash
python test_core.py
```

This tests:
- UUIDv8 generation
- Local model availability
- Entity extraction (with fallback to mock data)
- JSON structure compatibility

## Key Features

### 1. Flexible Model Support
- Switch between Ollama and LM Studio seamlessly
- Support for custom inference server URLs
- Automatic fallback and error handling

### 2. Unified Entity Tracking
- UUIDv8 generation for consistent tracking
- Cross-database identifier alignment
- Temporal and namespace components

### 3. MCP Integration
- Automatic storage in Neo4j knowledge graph
- Vector storage in Qdrant for similarity search
- Real-time storage status reporting

### 4. Backward Compatibility
- Original JSON structure maintained
- Enhanced metadata with new features
- Drop-in replacement for existing integrations

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `ollama` | Model provider: "ollama" or "lmstudio" |
| `LOCAL_MODEL` | `llama3.2:latest` | Model name to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234` | LM Studio server URL |

## Troubleshooting

### Model Not Available
If you see "model not found" errors:
1. Check if your model server is running
2. Verify the model name exists
3. For Ollama: `ollama pull llama3.2:latest`
4. For LM Studio: Load the model in the interface

### MCP Integration Issues
- Ensure Neo4j and Qdrant MCP servers are configured in Claude Code
- Check console output for storage confirmation messages
- MCP integration gracefully degrades if servers unavailable

### Performance Optimization
- Use dedicated inference servers for better performance
- Configure appropriate model sizes for your hardware
- Monitor model response times and adjust timeout settings

## Migration from Original

To migrate from the original HuggingFace version:

1. Remove `HF_TOKEN` environment variable dependency
2. Set up local model server (Ollama or LM Studio)
3. Configure `MODEL_PROVIDER` and `LOCAL_MODEL` variables
4. The JSON output format remains the same with additional metadata

## Future Enhancements

Planned improvements:
- Support for additional local model providers
- Enhanced vector embedding options
- GraphQL API for knowledge graph queries
- Batch processing capabilities
- Model-specific prompt optimization