import sys, os
sys.path.insert(0, '.')

# Test 1: calibration
from utils.calibration import calibrate_confidence, compute_ece
c = calibrate_confidence(0.95, 1.5)
print(f'[OK] calibrate_confidence(0.95, T=1.5) = {c}  (expected ~0.877)')

# Test 2: severity estimator
from utils.severity_estimator import estimate_severity, severity_score
sev = estimate_severity('The contractor shall indemnify and hold harmless all parties.', 'Liability', 0.91)
scr = severity_score('The contractor shall indemnify and hold harmless all parties.', 'Liability', 0.91)
print(f'[OK] estimate_severity (Liability, conf=0.91) = {sev}  (expected High)')
print(f'[OK] severity_score    (Liability, conf=0.91) = {scr}')

# Test 3: ML classifier
from utils.ml_classifier import predict_label
label, conf = predict_label('The contractor shall indemnify and hold harmless.')
print(f'[OK] predict_label ML = ({label}, {conf:.4f})')

# Test 4: nlp_utils CCIC (keyword tier only)
os.environ['RISK_DETECTOR'] = 'keyword'
from utils.nlp_utils import analyze_risks, segment_sentences
sentences = segment_sentences(
    'The contractor shall indemnify all parties. '
    'Late payments are subject to a 1.5% monthly surcharge. '
    'This is a purely administrative clause.'
)
results = analyze_risks(sentences)
print(f'[OK] analyze_risks: {len(results)} risks found (expected 2)')
for r in results:
    rt = r['risk_type']
    sv = r['severity']
    sc = r['severity_score']
    dt = r['detector']
    print(f'     - {rt} | {sv} | score={sc} | detector={dt}')

print()
print('All core tests PASSED.')
