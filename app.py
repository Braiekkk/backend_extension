from flask import Flask, request, jsonify
from flask_cors import CORS
from hugface2 import process_message  # Import the workflow function from your HuggingFace integration
app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Resource Sharing for frontend calls


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

    result = process_message(text, session["items"])

    return jsonify({
        "result": result,
        "stored_items_count": len(session["items"])
    }), 200


if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)