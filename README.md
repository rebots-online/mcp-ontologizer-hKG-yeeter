---
title: Knowledge Graph Builder-MCP Server
emoji: 🦀
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
short_description: Transforms any text or webpage into knowledge graphs
tags:
- mcp-server-track
- agent-demo-track
- knowledge-graph
- entity-extraction
- nlp
- visualization
- semantic-analysis
- clustering
- embeddings
---
# Knowledge Graph Builder-MCP Server

A Knowledge Graph Builder Model Context Protocol (MCP) server that transforms any text or webpage into interactive knowledge graphs using AI-powered entity extraction and relationship mapping.


## 📊 What It Extracts

- **🏢 Organizations**: Companies, institutions, groups
- **👥 People**: Names, roles, relationships
- **📍 Locations**: Places, countries, regions  
- **💡 Concepts**: Ideas, technologies, events
- **🔗 Relationships**: Connections between entities

## 🎬 Demo Video

This space demonstrates the MCP server in action. You can watch a video recording of the server running below:

Website URL Test:
<video controls src="https://huggingface.co/spaces/Agents-MCP-Hackathon/KGB-mcp/resolve/main/Screen%20Recording%202025-06-05%20at%2023.29.31.mov" style="width: 80%; max-width: 80%; height: auto;">
</video>
**Input**:
```
{
  "input_text": "https://huggingface.co/Agents-MCP-Hackathon"
}
```

**Returned results**:
```
root={'source': {'type': 'url', 'value': 'https://huggingface.co/Agents-MCP-Hackathon', 'content_preview': 'Agents-MCP-Hackathon (Agents-MCP-Hackathon) Hugging Face Models Datasets Spaces Community Docs Enterprise Pricing Log In Sign Up Agents-MCP-Hackathon community https://www.gradio.app/ gradio gradio-ap...'}, 'knowledge_graph': {'entities': [{'name': 'Agents-MCP-Hackathon', 'type': 'EVENT', 'description': 'A hackathon focused on Model Context Protocol (MCP) and AI Agents.'}, {'name': 'Hugging Face', 'type': 'ORGANIZATION', 'description': 'A company providing models, datasets, and spaces for AI and ML.'}, {'name': 'Gradio', 'type': 'ORGANIZATION', 'description': 'A platform for creating and sharing machine learning models.'}, {'name': 'Model Context Protocol (MCP)', 'type': 'CONCEPT', 'description': 'An open protocol that standardizes how applications provide context to LLMs.'}, {'name': 'TouradAi', 'type': 'PERSON', 'description': 'A participant who updated a Space.'}, {'name': 'kingabzpro', 'type': 'PERSON', 'description': 'A participant who updated a Space.'}, {'name': 'VirtualOasis', 'type': 'PERSON', 'description': 'A participant who updated a Space.'}, {'name': 'Modal Labs', 'type': 'ORGANIZATION', 'description': 'A sponsor providing GPU/CPU Compute credits.'}, {'name': 'Nebius', 'type': 'ORGANIZATION', 'description': 'A sponsor providing API credits.'}, {'name': 'Anthropic', 'type': 'ORGANIZATION', 'description': 'A sponsor providing API credits.'}, {'name': 'OpenAI', 'type': 'ORGANIZATION', 'description': 'A sponsor providing API credits.'}, {'name': 'Hyperbolic Labs', 'type': 'ORGANIZATION', 'description': 'A sponsor providing API credits.'}, {'name': 'MistralAI', 'type': 'ORGANIZATION', 'description': 'A sponsor providing API credits.'}, {'name': 'Sambanova.AI', 'type': 'ORGANIZATION', 'description': 'A sponsor providing API credits.'}], 'relationships': [{'source': 'Agents-MCP-Hackathon', 'target': 'Hugging Face', 'relationship': 'ASSOCIATED_WITH', 'description': 'Hugging Face is associated with the Agents-MCP-Hackathon.'}, {'source': 'Agents-MCP-Hackathon', 'target': 'Gradio', 'relationship': 'ASSOCIATED_WITH', 'description': 'Gradio is associated with the Agents-MCP-Hackathon.'}, {'source': 'Agents-MCP-Hackathon', 'target': 'Model Context Protocol (MCP)', 'relationship': 'FOCUSED_ON', 'description': 'The hackathon focuses on Model Context Protocol (MCP).'}, {'source': 'TouradAi', 'target': 'Agents-MCP-Hackathon', 'relationship': 'PARTICIPATED_IN', 'description': 'TouradAi is a participant in the Agents-MCP-Hackathon.'}, {'source': 'kingabzpro', 'target': 'Agents-MCP-Hackathon', 'relationship': 'PARTICIPATED_IN', 'description': 'kingabzpro is a participant in the Agents-MCP-Hackathon.'}, {'source': 'VirtualOasis', 'target': 'Agents-MCP-Hackathon', 'relationship': 'PARTICIPATED_IN', 'description': 'VirtualOasis is a participant in the Agents-MCP-Hackathon.'}, {'source': 'Modal Labs', 'target': 'Agents-MCP-Hackathon', 'relationship': 'SPONSOR_OF', 'description': 'Modal Labs is a sponsor of the Agents-MCP-Hackathon.'}, {'source': 'Nebius', 'target': 'Agents-MCP-Hackathon', 'relationship': 'SPONSOR_OF', 'description': 'Nebius is a sponsor of the Agents-MCP-Hackathon.'}, {'source': 'Anthropic', 'target': 'Agents-MCP-Hackathon', 'relationship': 'SPONSOR_OF', 'description': 'Anthropic is a sponsor of the Agents-MCP-Hackathon.'}, {'source': 'OpenAI', 'target': 'Agents-MCP-Hackathon', 'relationship': 'SPONSOR_OF', 'description': 'OpenAI is a sponsor of the Agents-MCP-Hackathon.'}, {'source': 'Hyperbolic Labs', 'target': 'Agents-MCP-Hackathon', 'relationship': 'SPONSOR_OF', 'description': 'Hyperbolic Labs is a sponsor of the Agents-MCP-Hackathon.'}, {'source': 'MistralAI', 'target': 'Agents-MCP-Hackathon', 'relationship': 'SPONSOR_OF', 'description': 'MistralAI is a sponsor of the Agents-MCP-Hackathon.'}, {'source': 'Sambanova.AI', 'target': 'Agents-MCP-Hackathon', 'relationship': 'SPONSOR_OF', 'description': 'Sambanova.AI is a sponsor of the Agents-MCP-Hackathon.'}], 'entity_count': 14, 'relationship_count': 13}, 'metadata': {'model': 'mistralai/Mistral-Small-24B-Instruct-2501', 'content_length': 5000}}
```
<video controls src="https://huggingface.co/spaces/Agents-MCP-Hackathon/KGB-mcp/resolve/main/Screen%20Recording%202025-06-05%20at%2023.30.44.mov" style="width: 80%; max-width: 80%; height: auto;">
</video>

**Input**:
```
{
  "input_text": "OpenAI released GPT-4 in March 2023, which was a significant advancement in large language models. The model was developed by Sam Altman's team at OpenAI and showed remarkable improvements in reasoning capabilities compared to GPT-3.5. Meanwhile, Google launched Bard AI as a competitor, powered by their LaMDA technology. Anthropic, founded by former OpenAI researchers Dario and Daniela Amodei, released Claude AI as another alternative. These AI companies are competing in the rapidly growing artificial intelligence market, with Microsoft investing heavily in OpenAI through Azure integration. Meta has also entered the space with their LLaMA models, while Elon Musk founded xAI to develop Grok. The AI safety concerns raised by researchers like Geoffrey Hinton and Yoshua Bengio have sparked debates about responsible AI development across the industry."
}
```

**Returned results**:
```
root={'source': {'type': 'text', 'value': 'direct_input', 'content_preview': "OpenAI released GPT-4 in March 2023, which was a significant advancement in large language models. The model was developed by Sam Altman's team at OpenAI and showed remarkable improvements in reasonin..."}, 'knowledge_graph': {'entities': [{'name': 'OpenAI', 'type': 'ORGANIZATION', 'description': 'A company that released GPT-4.'}, {'name': 'GPT-4', 'type': 'CONCEPT', 'description': 'A large language model released by OpenAI.'}, {'name': 'Sam Altman', 'type': 'PERSON', 'description': 'Leader of the team that developed GPT-4 at OpenAI.'}, {'name': 'GPT-3.5', 'type': 'CONCEPT', 'description': "A previous version of OpenAI's large language model."}, {'name': 'Google', 'type': 'ORGANIZATION', 'description': 'A company that launched Bard AI.'}, {'name': 'Bard AI', 'type': 'CONCEPT', 'description': 'An AI model launched by Google.'}, {'name': 'LaMDA', 'type': 'CONCEPT', 'description': 'Technology used to power Bard AI.'}, {'name': 'Anthropic', 'type': 'ORGANIZATION', 'description': 'A company that released Claude AI.'}, {'name': 'Dario Amodei', 'type': 'PERSON', 'description': 'Founder of Anthropic.'}, {'name': 'Daniela Amodei', 'type': 'PERSON', 'description': 'Founder of Anthropic.'}, {'name': 'Claude AI', 'type': 'CONCEPT', 'description': 'An AI model released by Anthropic.'}, {'name': 'Microsoft', 'type': 'ORGANIZATION', 'description': 'A company investing in OpenAI through Azure integration.'}, {'name': 'Azure', 'type': 'CONCEPT', 'description': "A platform used for OpenAI's integration with Microsoft."}, {'name': 'Meta', 'type': 'ORGANIZATION', 'description': 'A company that entered the AI space with LLaMA models.'}, {'name': 'LLaMA', 'type': 'CONCEPT', 'description': 'AI models developed by Meta.'}, {'name': 'Elon Musk', 'type': 'PERSON', 'description': 'Founder of xAI.'}, {'name': 'xAI', 'type': 'ORGANIZATION', 'description': 'A company founded by Elon Musk.'}, {'name': 'Grok', 'type': 'CONCEPT', 'description': 'An AI model developed by xAI.'}, {'name': 'Geoffrey Hinton', 'type': 'PERSON', 'description': 'A researcher who raised AI safety concerns.'}, {'name': 'Yoshua Bengio', 'type': 'PERSON', 'description': 'A researcher who raised AI safety concerns.'}, {'name': 'Artificial Intelligence Market', 'type': 'CONCEPT', 'description': 'The market where AI companies are competing.'}, {'name': 'Responsible AI Development', 'type': 'CONCEPT', 'description': 'The topic of debate sparked by AI safety concerns.'}], 'relationships': [{'source': 'OpenAI', 'target': 'GPT-4', 'relationship': 'DEVELOPED', 'description': 'OpenAI developed GPT-4.'}, {'source': 'Sam Altman', 'target': 'OpenAI', 'relationship': 'LEADS', 'description': 'Sam Altman leads the team at OpenAI.'}, {'source': 'OpenAI', 'target': 'GPT-3.5', 'relationship': 'PREVIOUS_VERSION', 'description': "GPT-3.5 is a previous version of OpenAI's model."}, {'source': 'Google', 'target': 'Bard AI', 'relationship': 'LAUNCHED', 'description': 'Google launched Bard AI.'}, {'source': 'Google', 'target': 'LaMDA', 'relationship': 'POWERED_BY', 'description': 'Bard AI is powered by LaMDA.'}, {'source': 'Anthropic', 'target': 'Claude AI', 'relationship': 'RELEASED', 'description': 'Anthropic released Claude AI.'}, {'source': 'Dario Amodei', 'target': 'Anthropic', 'relationship': 'FOUNDED', 'description': 'Dario Amodei founded Anthropic.'}, {'source': 'Daniela Amodei', 'target': 'Anthropic', 'relationship': 'FOUNDED', 'description': 'Daniela Amodei founded Anthropic.'}, {'source': 'Microsoft', 'target': 'OpenAI', 'relationship': 'INVESTED_IN', 'description': 'Microsoft invested in OpenAI.'}, {'source': 'Microsoft', 'target': 'Azure', 'relationship': 'INTEGRATED_WITH', 'description': 'Microsoft integrated OpenAI through Azure.'}, {'source': 'Meta', 'target': 'LLaMA', 'relationship': 'DEVELOPED', 'description': 'Meta developed LLaMA models.'}, {'source': 'Elon Musk', 'target': 'xAI', 'relationship': 'FOUNDED', 'description': 'Elon Musk founded xAI.'}, {'source': 'xAI', 'target': 'Grok', 'relationship': 'DEVELOPED', 'description': 'xAI developed Grok.'}, {'source': 'Geoffrey Hinton', 'target': 'Responsible AI Development', 'relationship': 'RAISED_CONCERNS', 'description': 'Geoffrey Hinton raised concerns about responsible AI development.'}, {'source': 'Yoshua Bengio', 'target': 'Responsible AI Development', 'relationship': 'RAISED_CONCERNS', 'description': 'Yoshua Bengio raised concerns about responsible AI development.'}, {'source': 'OpenAI', 'target': 'Artificial Intelligence Market', 'relationship': 'COMPETES_IN', 'description': 'OpenAI competes in the AI market.'}, {'source': 'Google', 'target': 'Artificial Intelligence Market', 'relationship': 'COMPETES_IN', 'description': 'Google competes in the AI market.'}, {'source': 'Anthropic', 'target': 'Artificial Intelligence Market', 'relationship': 'COMPETES_IN', 'description': 'Anthropic competes in the AI market.'}, {'source': 'Microsoft', 'target': 'Artificial Intelligence Market', 'relationship': 'COMPETES_IN', 'description': 'Microsoft competes in the AI market.'}, {'source': 'Meta', 'target': 'Artificial Intelligence Market', 'relationship': 'COMPETES_IN', 'description': 'Meta competes in the AI market.'}, {'source': 'xAI', 'target': 'Artificial Intelligence Market', 'relationship': 'COMPETES_IN', 'description': 'xAI competes in the AI market.'}], 'entity_count': 22, 'relationship_count': 21}, 'metadata': {'model': 'mistralai/Mistral-Small-24B-Instruct-2501', 'content_length': 858}}
```

**Built with ❤️ for the Gradio MCP Hackathon 2025**

*Transform any content into knowledge graphs with the power of AI and MCP integration!* 

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference