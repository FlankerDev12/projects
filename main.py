from fastapi import FastAPI, HTTPException
from huggingface_hub import InferenceClient
from RAG_engine import retrieve_answer as retrieve_info
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize HuggingFace client
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",  # Using a more reliable model
    token="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"
)

@app.get("/")
def read_root():
    return {"message": "Quantum Mechanics Troubleshoot Online", "status": "running"}


@app.get("/rag")
def rag(query: str):  
    """RAG mode: Retrieve context from PDF documents"""
    try:
        logger.info(f"RAG query received: {query}")
        context = retrieve_info(query)
        
        if not context:
            return {"context": "No relevant information found in documents."}
        
        logger.info(f"Context retrieved successfully, length: {len(context)}")
        return {"context": context}
    
    except Exception as e:
        logger.error(f"RAG error: {str(e)}")
        return {"error": str(e), "context": ""}


@app.get("/troubleshoot")
def troubleshoot(issue: str):
    """AI mode: Generate response using LLM without RAG"""
    try:
        logger.info(f"AI query received: {issue}")
        
        # Create a focused prompt
        prompt = f"""You are an expert in quantum mechanics. Answer the following question clearly and concisely:

Question: {issue}

Provide a detailed explanation with key concepts, formulas (in LaTeX), and examples where appropriate."""

        # Call HuggingFace API
        response = client.chat_completion(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional quantum physics assistant with deep knowledge of quantum mechanics, quantum theory, and related mathematical concepts."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Extract the answer
        if hasattr(response, 'choices') and len(response.choices) > 0:
            # New API format
            answer = response.choices[0].message.content
        elif isinstance(response, dict):
            # Dict response format
            if 'choices' in response and len(response['choices']) > 0:
                answer = response['choices'][0].get('message', {}).get('content', '')
            else:
                answer = response.get('generated_text', '')
        else:
            answer = str(response)
        
        if not answer or answer.strip() == "":
            logger.warning("Empty response from model")
            answer = "⚠️ The AI model returned an empty response. Please try rephrasing your question."
        
        logger.info(f"Response generated successfully, length: {len(answer)}")
        
        # Return with the expected key
        return {
            "response": answer,
            "troubleshooting_guide": answer,  # Keep for backward compatibility
            "answer": answer  # Additional key for flexibility
        }
    
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "response": f"⚠️ {error_msg}",
            "troubleshooting_guide": f"⚠️ {error_msg}"
        }


# Health check endpoint
@app.get("/health")
def health_check():
    """Check if the API is running and model is accessible"""
    try:
        # Try a simple test
        test_response = client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
        return {
            "status": "healthy",
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "api": "operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)