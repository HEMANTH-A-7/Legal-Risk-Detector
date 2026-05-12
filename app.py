import os
import sys
import secrets
import tempfile
import traceback

from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from utils.llm_explainer import explain_clause
from utils.nlp_utils import (
    analyze_risks,
    extract_text_from_pdf,
    generate_explanation,
    segment_sentences,
)


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# Configuration for file uploads
# Use tempfile.gettempdir() so this works on both Linux (HF Spaces) and Windows (local dev)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'lexguard_uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# Ensure upload directory exists
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception:
    pass

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint for analyzing uploaded files or pasted text.

    Runs the full CCIC pipeline (Tier 1 → Tier 2 → Tier 3) and returns:
    - risks: list of detected risk clauses with HSE severity scores
    - ccic: Confidence-Calibrated Inference Cascade metadata for paper figures
    - summary: aggregate statistics including severity distribution
    """
    # Check if a file was uploaded
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            suffix = os.path.splitext(filename)[1].lower()
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp_path = tmp.name
                    file.save(tmp_path)

                if suffix == '.pdf':
                    text = extract_text_from_pdf(tmp_path)
                else:
                    with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
            finally:
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        else:
            return jsonify({'error': 'Invalid file type. Only .txt and .pdf are supported.'}), 400

    # If no file, check for pasted text
    elif 'text' in request.form and request.form['text'].strip() != '':
        text = request.form['text']

    else:
        return jsonify({'error': 'No file or text provided.'}), 400

    # Analyze the text
    try:
        sentences = segment_sentences(text)
        risks = analyze_risks(sentences)

        # Calculate summary statistics
        total_sentences = len(sentences)
        risky_count = len(risks)

        llm_available = bool(os.getenv("OPENAI_API_KEY"))

        # ── Build summary with HSE severity distribution (for paper's analysis) ──
        severity_dist = {'High': 0, 'Medium': 0, 'Low': 0}
        detector_dist = {'transformer': 0, 'ml': 0, 'keyword': 0}
        for r in risks:
            sev = r.get('severity', 'Low')
            det = r.get('detector', 'keyword')
            severity_dist[sev] = severity_dist.get(sev, 0) + 1
            detector_dist[det] = detector_dist.get(det, 0) + 1

        return jsonify({
            'risks': risks,
            'llm_available': llm_available,
            # CCIC metadata — exposed for evaluation and paper figures
            'ccic': {
                'mode': os.getenv('RISK_DETECTOR', 'auto'),
                'temperature': float(os.getenv('CCIC_TEMPERATURE', '1.5')),
                'transformer_threshold': float(os.getenv('TRANSFORMER_THRESHOLD', '0.6')),
                'ml_threshold': float(os.getenv('ML_RISK_THRESHOLD', '0.5')),
                'detector_distribution': detector_dist,
            },
            'summary': {
                'total_sentences': total_sentences,
                'risky_count': risky_count,
                'risk_percentage': round((risky_count / total_sentences * 100), 2) if total_sentences > 0 else 0,
                'severity_distribution': severity_dist,
            }
        })
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[ANALYZE ERROR] {exc}\n{tb}", file=sys.stderr, flush=True)
        return jsonify({'error': f'Analysis failed: {str(exc)}', 'traceback': tb}), 500


@app.route('/explain', methods=['POST'])
def explain():
    """
    LLM explanation endpoint — calls GPT-4o-mini if OPENAI_API_KEY is set,
    otherwise falls back to the template-based generator in nlp_utils.py.
    """
    payload = request.get_json(silent=True) or {}
    sentence = (payload.get("sentence") or "").strip()
    risk_type = (payload.get("risk_type") or "").strip()
    severity = (payload.get("severity") or "").strip()

    if not sentence or not risk_type or not severity:
        return jsonify({"error": "Missing required fields: sentence, risk_type, severity"}), 400

    if len(sentence) > 8000:
        return jsonify({"error": "Sentence too long"}), 400

    try:
        explanation = explain_clause(sentence=sentence, risk_type=risk_type, severity=severity)
        return jsonify({"explanation": explanation, "source": "llm"})
    except Exception:
        explanation = generate_explanation(sentence, risk_type, severity)
        return jsonify({"explanation": explanation, "source": "template"})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
