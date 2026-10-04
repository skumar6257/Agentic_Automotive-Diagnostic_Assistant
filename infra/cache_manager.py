import os
import datetime
from vertexai.preview import caching
import vertexai

def create_vertex_cache(project_id: str, location: str, massive_text: str) -> str:
    """
    Creates a KV Context Cache in Vertex AI for massive OEM manuals.
    Returns the Cache ID string.
    """

    print("--- [Cloud Cache] Uploading massive manual to Vertex AI KV Cache ---")

    try:
        vertexai.init(project=project_id, location=location)

        # Freeze the text in Google's cache for 60 minutes
        cached_content = caching.CachedContent.create(
            model_name="gemini-1.5-pro-001",
            system_instruction="You are an expert Automotive Diagnostic AI.",
            contents=[massive_text],
            ttl=datetime.timedelta(minutes=60),
        )

        print(f"✅ Context Cache created! Cache ID: {cached_content.name}")
        return cached_content.name
    
    except ImportError:
        print("Vertex AI SDK not installed. Skipping cache.")
        return None
    
    except Exception as e:
        print(f"Failed to create Vertex Cache: {e}")
        return None


