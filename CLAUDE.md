# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KGB-mcp is a Knowledge Graph Builder MCP (Model Context Protocol) server that transforms text or web content into structured knowledge graphs using AI-powered entity extraction and relationship mapping. The project is built as a Gradio application designed for the MCP Hackathon 2025.

## Core Architecture

- **Entry Point**: `app.py` - Main application file containing the Gradio interface and MCP server
- **AI Model**: Uses Mistral AI (`mistralai/Mistral-Small-24B-Instruct-2501`) via HuggingFace Inference Client
- **Web Scraping**: BeautifulSoup for extracting text content from URLs
- **Output Format**: Structured JSON knowledge graphs containing entities and relationships

## Key Functions

- `extract_text_from_url()` - Scrapes and cleans text from web URLs (app.py:15)
- `extract_entities_and_relationships()` - Uses Mistral AI to extract structured knowledge graphs (app.py:42)
- `build_knowledge_graph()` - Main orchestration function that handles both text and URL inputs (app.py:134)

## Environment Setup

**Required Environment Variables:**
- `HF_TOKEN`: HuggingFace API token for accessing Mistral AI through the inference client

**Dependencies Installation:**
```bash
pip install -r requirements.txt
```

## Running the Application

**Development:**
```bash
python app.py
```

The application launches a Gradio interface with MCP server capabilities enabled (`mcp_server=True`).

## Input/Output Format

**Input**: Text content or web URLs
**Output**: JSON structure containing:
- `source`: Information about the input (type, value, content preview)
- `knowledge_graph`: Extracted entities and relationships with counts
- `metadata`: Model information and content length

**Entity Types**: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, OTHER
**Relationship Types**: Custom relationship types extracted by the AI model

## Content Limits

- URL content: Limited to first 5000 characters
- AI analysis: Uses first 3000 characters of content
- Content preview: First 200 characters in output

## Error Handling

The application includes comprehensive error handling for:
- Invalid URLs or network failures
- Missing API tokens
- JSON parsing errors from LLM responses
- Malformed or empty inputs