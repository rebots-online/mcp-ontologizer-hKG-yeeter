#!/usr/bin/env python3
"""
Test script for the integrated KGB-MCP system.
Tests local model integration, UUIDv8 generation, and MCP storage simulation.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import (
    generate_uuidv8,
    call_local_model,
    extract_entities_and_relationships,
    build_knowledge_graph_sync,
    store_in_neo4j_sync,
    store_in_qdrant_sync,
    MODEL_PROVIDER,
    DEFAULT_MODEL,
    OLLAMA_AVAILABLE,
    OPENAI_AVAILABLE,
    MCP_AVAILABLE
)

def test_uuidv8_generation():
    """Test UUIDv8 generation."""
    print("🧪 Testing UUIDv8 generation...")
    uuid1 = generate_uuidv8()
    uuid2 = generate_uuidv8()
    
    print(f"UUID1: {uuid1}")
    print(f"UUID2: {uuid2}")
    
    # Check format
    assert len(uuid1) == 36, "UUID should be 36 characters"
    assert uuid1.count('-') == 4, "UUID should have 4 hyphens"
    assert uuid1 != uuid2, "UUIDs should be unique"
    assert uuid1[14] == '8', "Should be UUIDv8 (version 8)"
    
    print("✅ UUIDv8 generation test passed!")
    return True

def test_local_model_integration():
    """Test local model integration."""
    print("🧪 Testing local model integration...")
    print(f"Model Provider: {MODEL_PROVIDER}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Ollama Available: {OLLAMA_AVAILABLE}")
    print(f"OpenAI Client Available: {OPENAI_AVAILABLE}")
    
    # Test with a simple prompt
    test_prompt = "What is artificial intelligence? Give a brief answer."
    response = call_local_model(test_prompt)
    
    print(f"Response: {response[:200]}...")
    
    if response.startswith("Error:"):
        print("⚠️  Local model not available, but integration code is working")
        return False
    else:
        print("✅ Local model integration test passed!")
        return True

def test_entity_extraction():
    """Test entity and relationship extraction."""
    print("🧪 Testing entity extraction...")
    
    test_text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. 
    The company is headquartered in Cupertino, California and is known for creating 
    innovative products like the iPhone and MacBook.
    """
    
    result = extract_entities_and_relationships(test_text)
    
    print(f"Extracted entities: {len(result.get('entities', []))}")
    print(f"Extracted relationships: {len(result.get('relationships', []))}")
    
    if "error" in result:
        print(f"⚠️  Error in extraction: {result['error']}")
        return False
    
    print("✅ Entity extraction test passed!")
    return True

def test_knowledge_graph_building():
    """Test the complete knowledge graph building process."""
    print("🧪 Testing knowledge graph building...")
    
    test_text = """
    Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975. 
    The company develops software products including Windows operating system 
    and Office productivity suite. Microsoft is headquartered in Redmond, Washington.
    """
    
    result = build_knowledge_graph_sync(test_text)
    
    print(f"Result keys: {list(result.keys())}")
    
    if "error" in result:
        print(f"⚠️  Error in knowledge graph building: {result['error']}")
        return False
    
    # Check structure
    assert "source" in result, "Should have source information"
    assert "knowledge_graph" in result, "Should have knowledge_graph section"
    assert "metadata" in result, "Should have metadata section"
    
    # Check UUID is present
    assert "uuid" in result["metadata"], "Should have UUID in metadata"
    
    print("✅ Knowledge graph building test passed!")
    return True

def test_mcp_integration():
    """Test MCP integration (simulation)."""
    print("🧪 Testing MCP integration...")
    print(f"MCP Available: {MCP_AVAILABLE}")
    
    # Test data
    test_entities = [
        {"name": "Test Company", "type": "ORGANIZATION", "description": "A test company"}
    ]
    test_relationships = [
        {"source": "Test Person", "target": "Test Company", "relationship": "WORKS_FOR"}
    ]
    test_uuid = generate_uuidv8()
    test_content = "Test content for vector storage"
    
    # Test Neo4j storage
    neo4j_result = store_in_neo4j_sync(test_entities, test_relationships, test_uuid)
    print(f"Neo4j storage result: {neo4j_result}")
    
    # Test Qdrant storage
    qdrant_result = store_in_qdrant_sync(test_content, test_entities, test_relationships, test_uuid)
    print(f"Qdrant storage result: {qdrant_result}")
    
    print("✅ MCP integration test completed!")
    return True

def main():
    """Run all tests."""
    print("🚀 Starting KGB-MCP Integration Tests")
    print("=" * 50)
    
    tests = [
        test_uuidv8_generation,
        test_local_model_integration,
        test_entity_extraction,
        test_knowledge_graph_building,
        test_mcp_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
        print("-" * 30)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check configuration")

if __name__ == "__main__":
    main()