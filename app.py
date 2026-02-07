from flask import Flask, request, jsonify
from flask_cors import CORS
from hugface2 import process_message  # Import the workflow function from your HuggingFace integration
app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS with credentials for cookies


import uuid
from threading import Lock

SESSIONS = {}
LOCK = Lock()

def get_session(session_id):
    with LOCK:
        return SESSIONS.get(session_id)

def create_session():
    session_id = str(uuid.uuid4())
    with LOCK:
        SESSIONS[session_id] = {
            "items": []
        }
    return session_id


from flask import request, jsonify, make_response

@app.route("/api/connect", methods=["POST"])
def connect():
    data = request.get_json() or {}
    items = data.get("items", [])

    session_id = request.cookies.get("session_id")

    # Create session if missing / invalid
    if not session_id or not get_session(session_id):
        session_id = create_session()

    session = get_session(session_id)

    # Append objects
    if isinstance(items, list):
        session["items"].extend(items)

    response = make_response(jsonify({
        "stored_items_count": len(session["items"])
    }))

    # 🍪 Set cookie
    response.set_cookie(
        "session_id",
        session_id,
        httponly=True,
        samesite="Lax"
        # secure=True  # enable in HTTPS
    )

    return response, 200



@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text")
    items = data.get("items", [])

    session_id = request.cookies.get("session_id")

    if not session_id:
        return jsonify({"error": "No session"}), 401

    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Invalid session"}), 401

    # Optional append
    if isinstance(items, list):
        session["items"].extend(items)

    result = process_message(text)

    return jsonify({
        "result": result,
        "stored_items_count": len(session["items"])
    }), 200


@app.route("/api/detect", methods=["POST"])
def detect():
    """Detect hate speech in user input and provide rewrites if needed"""
    data = request.get_json()
    text = data.get("text", "")

    if not text or text.strip() == "":
        return jsonify({
            "is_hate": False,
            "score": 0.0,
            "category": None,
            "sentiment": "neutral",
            "message": None,
            "rewrites": []
        }), 200

    # Get or create session
    session_id = request.cookies.get("session_id")
    if not session_id or not get_session(session_id):
        session_id = create_session()

    # Process the message through HuggingFace pipeline
    result = process_message(text)

    # Extract results
    sentiment = result.get("sentiment", "neutral")
    category = result.get("category")
    replacement = result.get("replacement", "")
    severity = result.get("severity", "medium")
    
    # Calculate hate speech score (0.0 to 1.0)
    is_hate = sentiment == "negative" and category is not None
    
    # Score mapping based on category and severity
    if not is_hate:
        score = 0.0
    elif category == "Bullying":
        # Bullying has severity levels
        severity_scores = {"low": 0.4, "medium": 0.6, "high": 0.85}
        score = severity_scores.get(severity, 0.6)
    else:
        # Serious hate categories (Racism, Sexism, etc.)
        score = 0.95
    
    # Generate warning message
    message = None
    if is_hate:
        if category in ["Racism", "Sexism", "Xenophobia", "Ableism", "Religious hate", "Cultural discrimination"]:
            message = f"⚠️ This might come out as hurtful. Your message contains {category.lower()} and may seriously offend others."
        elif category == "Bullying":
            if severity == "high":
                message = "⚠️ This might come out as hurtful. Your message contains severe bullying language."
            elif severity == "medium":
                message = "⚠️ This might come out as hurtful. Your message may be perceived as offensive."
            else:
                message = "⚠️ This might come out as hurtful. Consider rephrasing to be more respectful."
    
    # Generate rewrites (provide 1-3 alternatives)
    rewrites = []
    if is_hate and replacement and replacement != text:
        rewrites.append(replacement)
    
    response = make_response(jsonify({
        "is_hate": is_hate,
        "score": round(score, 2),
        "category": category,
        "sentiment": sentiment,
        "message": message,
        "rewrites": rewrites,
        "severity": severity if category == "Bullying" else None
    }))
    
    # Set session cookie if new
    if request.cookies.get("session_id") != session_id:
        response.set_cookie(
            "session_id",
            session_id,
            httponly=True,
            samesite="None",
            secure=False  # Set to True if using HTTPS
        )
    
    return response, 200


if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)