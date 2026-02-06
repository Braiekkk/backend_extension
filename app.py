from flask import Flask, request, jsonify
from flask_cors import CORS
from backend_extension.hugface2 import process_message  # Import the workflow function from your HuggingFace integration
app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Resource Sharing for frontend calls

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    user_input = data['text']
    
    try:
        # Call your LangChain workflow
        result = process_message(user_input)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)