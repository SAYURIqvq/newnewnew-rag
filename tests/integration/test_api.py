"""Test API connections for DeepSeek LLM and local BGE-large embeddings."""

import os
from dotenv import load_dotenv

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

load_dotenv()


def test_deepseek_llm():
    """Test DeepSeek via Anthropic-compatible API."""
    print("Testing DeepSeek (Anthropic-compatible API) ...")
    try:
        from src.config import get_settings
        from src.llm.chat_model import create_chat_model

        settings = get_settings()
        if not settings.anthropic_auth_token:
            print("❌ ANTHROPIC_AUTH_TOKEN not set in .env")
            return False
        llm = create_chat_model(settings)
        response = llm.invoke("Say 'Hello, RAG!'")
        content = response.content if hasattr(response, "content") else str(response)
        print(f"✅ DeepSeek works! Response: {content}")
        return True
    except Exception as e:
        print(f"❌ DeepSeek failed: {e}")
        return False


def test_bge():
    """Test local BGE-large embeddings."""
    from sentence_transformers import SentenceTransformer

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
    print("API CONNECTION TESTS")
    print("=" * 60)

    llm_ok = test_deepseek_llm()
    bge_ok = test_bge()

    print("\n" + "=" * 60)
    if llm_ok and bge_ok:
        print("✅ ALL TESTS PASSED! Ready to build RAG.")
    else:
        print("❌ Some tests failed. Check ANTHROPIC_AUTH_TOKEN in .env")
    print("=" * 60)
