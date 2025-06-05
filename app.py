import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize Mistral client
client = InferenceClient(
    provider="together",
    api_key=os.environ.get("HF_TOKEN"),
)

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
        
        return text[:5000]  # Limit to first 5000 characters
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

def extract_entities_and_relationships(text):
    """Use Mistral to extract entities and relationships from text."""
    
    if not os.environ.get("HF_TOKEN"):
        return {
            "entities": [],
            "relationships": [],
            "error": "HF_TOKEN environment variable not set"
        }
    
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
    {text[:3000]}
    
    Please provide only the JSON response without any additional text or formatting.
    """
    
    try:
        completion = client.chat.completions.create(
            model="mistralai/Mistral-Small-24B-Instruct-2501",
            messages=[
                {
                    "role": "user",
                    "content": entity_prompt
                }
            ],
            max_tokens=4000,
            temperature=0.2
        )
        
        if not completion.choices or not completion.choices[0].message:
            return {
                "entities": [],
                "relationships": [],
                "error": "Empty response from Mistral API"
            }
        
        response_text = completion.choices[0].message.content.strip()
        
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
            "error": f"Error calling Mistral API: {str(e)}"
        }

def build_knowledge_graph(input_text):
    """Main function to build knowledge graph from text or URL."""
    
    try:
        if not input_text or not input_text.strip():
            return {
                "error": "Please provide text or a valid URL",
                "knowledge_graph": None
            }
        
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
        
        # Extract entities and relationships using Mistral
        kg_data = extract_entities_and_relationships(content)
        
        # Build the final knowledge graph structure
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
                "model": "mistralai/Mistral-Small-24B-Instruct-2501",
                "content_length": len(content)
            }
        }
        
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

# Create Gradio interface
demo = gr.Interface(
    fn=build_knowledge_graph,
    inputs=gr.Textbox(
        label="Text or URL Input",
        placeholder="Enter text to analyze or a web URL (e.g., https://example.com)",
        lines=5,
        max_lines=10
    ),
    outputs=gr.JSON(label="Knowledge Graph"),
    title="🧠 Knowledge Graph Builder",
    description="""
    **Build Knowledge Graphs with AI**
    
    This tool uses Mistral AI to extract entities and relationships from text or web content:
    
    • **Text Input**: Paste any text to analyze
    • **URL Input**: Provide a web URL to extract and analyze content
    • **Output**: Structured JSON knowledge graph for LLM agents
    
    The output includes entities (people, organizations, locations, concepts) and their relationships, formatted for easy consumption by AI agents.
    """,
    examples=[
        ["Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California."],
        ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
    ],
    cache_examples=False,
    theme=gr.themes.Soft()
)

demo.launch(mcp_server=True)
