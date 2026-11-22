"""Validate report schema and hashes without granting any modeling gate."""
import hashlib
import json
from pathlib import Path
import jsonschema

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT/'schemas/report-binding.schema.json'
binding_path = ROOT/'planning/literature/related_paper_analysis.json'
binding = json.loads(binding_path.read_text(encoding='utf-8'))
jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding='utf-8'))).validate(binding)
references = [binding['report']]+binding['inputs']
errors = []
for record in references:
    path = ROOT/record['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        errors.append(record['path'])
assert not errors, errors
for relative in ('results/q3_cost_rollout/development/inputs_before_run.json',
                 'results/q3_cost_rollout/repair_v2/review.json'):
    data = json.loads((ROOT/relative).read_text(encoding='utf-8'))
    refs = data if isinstance(data, list) else data['inputs']
    for record in refs:
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest() == record['sha256'], record['path']
gate = json.loads((ROOT/'planning/gates/literature_6pro_20260911_staleness.json').read_text(encoding='utf-8'))
result = {'report_binding_schema_valid': True, 'direct_reference_hashes_current': True,
          'checked_direct_references': len(references), 'v1_and_v2_evidence_hashes_current': True,
          'standard_R6_gate_status': gate['mechanical_status'], 'standard_R6_gate_errors': gate['errors'],
          'limitation': 'The generic R6 gate rejects the separate literature reviewer custom artifact type; this direct check does not replace or override that gate and grants no method approval.',
          'method_approved': False,
          'binding_sha256': hashlib.sha256(binding_path.read_bytes()).hexdigest()}
out = ROOT/'results/literature_transfer/report_binding_validation.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
