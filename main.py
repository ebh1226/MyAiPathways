import os
import json
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from flask import jsonify # For proper JSON response

# --- Configuration (MUST MATCH YOUR PHASE A SETUP) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
VECTOR_DIMENSION = 384
INDEX_NAME = "kelleher-hvac-parts-free"
PINECONE_REGION = 'us-east-1' 

# --- Global Initialization (Runs once when the function container starts) ---

# Initialize Pinecone and the Index globally
try:
    PINECECONE_KEY = os.environ.get("PINECONE_KEY")
    if not PINECECONE_KEY:
        raise ValueError("PINECONE_KEY environment variable not set.")
        
    pc = Pinecone(api_key=PINECECONE_KEY)
    index = pc.Index(INDEX_NAME)
    
    # Load the Sentence Transformer model globally
    # This keeps the model in memory for subsequent quick calls (low latency)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Initialization complete: Pinecone and SentenceTransformer loaded.")

except Exception as e:
    # Log initialization errors to Cloud Logging
    print(f"FATAL INITIALIZATION ERROR: {e}")
    pc = None
    embedding_model = None

# --- Core Logic Function ---
def find_matching_parts(technician_description: str, top_k: int = 3):
    """Embeds the description and queries Pinecone."""
    
    if not embedding_model or not pc:
        return "Service not initialized.", []

    # 1. Query Embedding
    query_vector = embedding_model.encode(
        [technician_description], 
        convert_to_numpy=True
    ).tolist()[0]
    
    # 2. Vector DB Query
    query_results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # 3. Process matches
    matches = []
    for match in query_results['matches']:
        matches.append({
            "part_number": match['id'],
            "description": match['metadata']['description'],
            "score": round(match['score'], 4)
        })
        
    return "Success", matches


# --- Cloud Function HTTP Entrypoint (Step B & C) ---
def query_part_finder(request):
    """
    HTTP Cloud Function entrypoint.
    Accepts POST requests with a 'description' field in JSON body.
    """
    
    # Set CORS headers for security and browser access (optional but good practice)
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    
    # Check if initialization failed
    if not embedding_model or not pc:
        return jsonify({"error": "Service initialization failed. Check server logs."}), 500

    # 1. Parse Request Data
    request_json = request.get_json(silent=True)
    if not request_json or 'description' not in request_json:
        return jsonify({"error": "Missing 'description' in request body."}), 400

    description = request_json['description']
    top_k = request_json.get('top_k', 3) # Allow caller to override top_k
    
    # 2. Run Core Logic
    try:
        status, matches = find_matching_parts(description, top_k)
        
        # 3. Return JSON Response (Endpoint)
        response_data = {
            "query_description": description,
            "status": status,
            "matches": matches
        }
        return jsonify(response_data), 200

    except Exception as e:
        # Return a clean error message if search fails
        return jsonify({"error": f"An unexpected search error occurred: {str(e)}"}), 500
