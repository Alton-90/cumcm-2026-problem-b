"""Read-only, stepwise public-log audit independent of the campaign's counters."""
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/q3_practice_100_20'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    plan = load(OUT/'plan.json')
    errors, rows, pending = [], [], []
    for item in plan['inputs']:
        if sha(ROOT/item['path']) != item['sha256']:
            errors.append('Input hash changed: '+item['path'])
    correction = load(OUT/'observer_correction.json')
    if sha(ROOT/'code/shared/q3_practice_campaign_run.py') != correction['corrected_runner_sha256']:
        errors.append('Correction runner hash changed')
    selection = load(OUT/'selection.json') if (OUT/'selection.json').exists() else None
    seen_cases = set()
    for journal_path in sorted((OUT/'trials').glob('*/journal.json')):
        journal = load(journal_path)
        if journal['stage'] != 'exported':
            pending.append({'folder': str(journal_path.parent), 'stage': journal['stage']})
            continue
        folder = journal_path.parent
        summary = load(folder/'summary.json')
        end = load(folder/'public_ui_after_exit.json')
        case = summary['case_id']
        local_errors = []
        def require(condition, message):
            if not condition:
                local_errors.append(message)
        require(case not in seen_cases, 'duplicate GUI case')
        seen_cases.add(case)
        require(summary['exited'] and end['test_ended'], 'post-exit evidence missing')
        require(end['completion_titles']==['问题3演练测试完成'] and end['case_ids']==[case], 'wrong completion case/question')
        counts = set()
        for text in end['source_texts']:
            m = re.search(r'(?:共|本次案例含干扰源)\s*(\d+).*?全向\s*(\d+).*?定向\s*(\d+)', text)
            if m: counts.add(tuple(map(int, m.groups())))
        require(counts == {(summary['true_source_count'], summary['true_source_count'], 0)}, 'source count differs from public UI')
        require(sha(folder/'completion.png')==summary['completion_screenshot_sha256'], 'screenshot hash mismatch')
        require(sha(ROOT/summary['export_path'])==summary['export_sha256'], 'export hash mismatch')
        require(summary['plan_sha256']==sha(OUT/'plan.json'), 'summary plan hash')
        if summary['batch']<=20:
            require(plan['tuning_batches'][summary['batch']-1]['profile']==summary['profile']['id'], 'tuning order/profile mismatch')
        else:
            require(selection is not None and summary['profile']==selection['profile'], 'validation winner changed')
        request_ids = {}
        position, measured_channel, last_clock = (0.,0.), 1, 0.
        cleared, step_errors, enters, exits = set(), [], 0, 0
        move_cost, measure_cost, switch_cost, clear_success_cost, clear_failure_cost = 0., 0., 0., 0., 0.
        for line in (folder/'client.jsonl').read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            request, response = event['request'], event.get('response', {})
            if response.get('accepted') is not True:
                continue
            rid = request['request_id']
            if rid in request_ids:
                require(request_ids[rid] == request, 'retry changed request payload')
                continue
            request_ids[rid] = request
            action = event['path']
            require(action in ('/enter','/measure','/clear','/exit'), 'unexpected interface action')
            if action=='/enter':
                enters += 1
                require(response['virtual_time_s']==0, 'enter clock nonzero')
            elif action=='/exit':
                exits += 1
                require(abs(response['virtual_time_s']-last_clock)<1e-8, 'exit changes virtual time')
            else:
                next_position = (request['position']['x'], request['position']['y'])
                movement = math.dist(position, next_position)/5
                move_cost += movement
                delta = movement
                if action=='/measure':
                    measure_cost += 5
                    change = int(request['channel']!=measured_channel)
                    switch_cost += change
                    delta += 5+change
                    measured_channel = request['channel']
                else:
                    if response['clear_result']=='success':
                        clear_success_cost += 5
                        delta += 5
                        require(request['channel'] not in cleared, 'duplicate successful clear')
                        cleared.add(request['channel'])
                    else:
                        clear_failure_cost += 3
                        delta += 3
                step_errors.append(abs(response['virtual_time_s']-last_clock-delta))
                position = next_position
            last_clock = response['virtual_time_s']
        require(enters==1 and exits==1, 'enter/exit count')
        require(not step_errors or max(step_errors)<2e-6, 'per-action fee mismatch beyond public timestamp rounding')
        require(abs(last_clock-summary['virtual_time_s'])<1e-8, 'summary time differs from simulator')
        require(len(cleared)==summary['cleared_channel_count'], 'clear count mismatch')
        require(set(summary['policy']['cleared_channels'])==cleared, 'policy and public clear sets differ')
        n = summary['true_source_count']
        require(abs(summary['s_per_true_source']-last_clock/n)<1e-9, 'mean denominator')
        components = {'movement_s':move_cost, 'measurement_s':measure_cost, 'switching_s':switch_cost, 'successful_clear_s':clear_success_cost, 'failed_clear_s':clear_failure_cost}
        require(abs(sum(components.values())-last_clock)<max(1e-5, len(step_errors)*5.1e-7), 'total reconstructed cost')
        row = {'case_id':case, 'batch':summary['batch'], 'stage':summary['stage'], 'profile':summary['profile']['id'],
               'true_sources':n, 'cleared_sources':len(cleared), 'all_cleared':summary['all_cleared'], 'virtual_time_s':last_clock,
               's_per_source':last_clock/n, **components, 'max_action_rounding_error_s':max(step_errors, default=0),
               'independent_checks_pass':not local_errors}
        rows.append(row)
        errors.extend(case+': '+message for message in local_errors)
    tuning = [r for r in rows if r['stage']=='tuning']
    holdout = [r for r in rows if r['stage']=='validation']
    if selection:
        if set(selection['selection_case_ids'])!={r['case_id'] for r in tuning}:
            errors.append('Selection case set mismatch')
        if set(selection['selection_case_ids']) & {r['case_id'] for r in holdout}:
            errors.append('Tuning/validation case overlap')
    result = {'scope':'PUBLIC_LOG_INDEPENDENT_ACCOUNTING', 'updated_at':datetime.now().astimezone().isoformat(),
              'audited_trials':len(rows), 'tuning_trials':len(tuning), 'validation_trials':len(holdout),
              'all_checks_pass':not errors, 'campaign_complete':len(tuning)==100 and len(holdout)==20 and not pending,
              'errors':errors, 'pending_attempts':pending, 'formal_tests':0,
              'first_case_observer_error_retained_and_corrected':correction, 'rows':rows}
    (OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if rows:
        with (OUT/'all_trials.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ('audited_trials','tuning_trials','validation_trials','all_checks_pass','campaign_complete','errors','pending_attempts')},ensure_ascii=False))
    return result


if __name__=='__main__':
    audit()
