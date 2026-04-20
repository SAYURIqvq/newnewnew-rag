"""Test API connections for Qwen (DashScope) and local BGE-large embeddings."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


def test_qwen():
    """Test Qwen via DashScope compatible-mode."""
    print("Testing Qwen (DashScope compatible-mode) ...")
    try:
        llm = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.getenv("LLM_MODEL", "qwen-plus"),
            temperature=0,
        )
        response = llm.invoke("Say 'Hello, RAG!'")
        print(f"✅ Qwen works! Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Qwen failed: {e}")
        return False


def test_bge():
    """Test local BGE-large embeddings."""
    print("\nTesting BGE-large (Sentence-Transformers) ...")
    try:
        model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        st_model = SentenceTransformer(model)
        embedding = st_model.encode(["Hello, world!"], normalize_embeddings=True)[0]
        print(f"✅ BGE works! Embedding dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"❌ BGE failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("API CONNECTION TESTS - Phase 1 Day 1")
    print("=" * 60)
    
    qwen_ok = test_qwen()
    bge_ok = test_bge()
    
    print("\n" + "=" * 60)
    if qwen_ok and bge_ok:
        print("✅ ALL TESTS PASSED! Ready to build RAG.")
        print("\nNext steps:")
        print("  1. Implement PDF loading")
        print("  2. Add text chunking")
        print("  3. Generate embeddings")
    else:
        print("❌ Some tests failed. Check your API keys in .env file")
    print("=" * 60)