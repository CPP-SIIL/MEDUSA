"""
Simple test Flask server to verify basic functionality
"""
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running'
    })

@app.route('/api/getModels', methods=['GET'])
def get_models():
    """Return some test models"""
    return jsonify({
        'models': [
            {'name': 'test_positive_1.stl', 'type': 'positive'},
            {'name': 'test_negative_1.stl', 'type': 'negative'}
        ]
    })

@app.route('/')
def index():
    return """
    <h1>Simple Test Server</h1>
    <p>Endpoints:</p>
    <ul>
        <li><a href="/api/ping">/api/ping</a></li>
        <li><a href="/api/getModels">/api/getModels</a></li>
    </ul>
    """

if __name__ == '__main__':
    print("Starting simple test server...")
    app.run(host='0.0.0.0', port=5000, debug=False)