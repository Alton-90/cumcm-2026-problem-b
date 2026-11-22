"""User-authorized Q3 practice tuning: 100 selection cases then 20 fresh cases.

No formal-test control, source-state access, or encrypted-log parsing is present.
The simulator's public completion counts are read only after the robot exits.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / 'code/shared'
OUT = ROOT / 'results/q3_practice_100_20'
UI = ROOT / 'results/desktop_control'
PWSH = os.environ.get('PWSH') or shutil.which('pwsh') or 'pwsh'
PROFILES = [
    {'id': 'G0', 'label': '原冻结方案', 'module': 'search_policy_q3_combined', 'parameters': {}},
    {'id': 'G1', 'label': '前瞻默认', 'module': 'search_policy_q3_cost_rollout_v2', 'parameters': {'MOVE_SLACK': 100.0, 'MIN_PREDICTED_SAVING': 5.0}},
    {'id': 'G2', 'label': '前瞻低门槛', 'module': 'search_policy_q3_cost_rollout_v2', 'parameters': {'MOVE_SLACK': 100.0, 'MIN_PREDICTED_SAVING': 0.0}},
    {'id': 'G3', 'label': '前瞻高门槛', 'module': 'search_policy_q3_cost_rollout_v2', 'parameters': {'MOVE_SLACK': 100.0, 'MIN_PREDICTED_SAVING': 15.0}},
    {'id': 'G4', 'label': '前瞻短移动', 'module': 'search_policy_q3_cost_rollout_v2', 'parameters': {'MOVE_SLACK': 50.0, 'MIN_PREDICTED_SAVING': 5.0}},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def now():
    return datetime.now().astimezone().isoformat()


def dependency_paths(names):
    found = set()
    def visit(name):
        path = SHARED / (name + '.py')
        if path in found or not path.is_file():
            return
        found.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    visit(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                visit(node.module.split('.')[0])
    for name in names:
        visit(name)
    return sorted(found)


def init():
    path = OUT / 'plan.json'
    if path.exists():
        raise RuntimeError('Existing prospectively frozen plan will not be overwritten')
    order = []
    rng = random.Random(20260911)
    for block in range(4):
        ids = [p['id'] for p in PROFILES]
        rng.shuffle(ids)
        order.extend({'batch': len(order) + i + 1, 'round': block + 1, 'profile': ident, 'count': 5, 'stage': 'tuning'} for i, ident in enumerate(ids))
    # list.extend consumes a generator while changing len(order); assign indices explicitly.
    for index, batch in enumerate(order, 1):
        batch['batch'] = index
    paths = dependency_paths([p['module'] for p in PROFILES] + ['simulator_client', 'practice_runner', 'q3_practice_campaign'])
    paths += [UI / x for x in ('practice_ui.ps1', 'capture_window.py', 'save_export_dialog.ps1')]
    inputs = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in paths]
    fixed = {k: getattr(importlib.import_module('q3_cost_rollout_v2'), k) for k in ('OFFSETS', 'BACKOFFS', 'RECEPTION_LIMIT', 'MAX_HYPOTHESES', 'BIASES')}
    plan = {'schema_version': 1, 'scope': 'Q3_PRACTICE_TUNING_ONLY', 'created_at': now(),
            'authorization': '用户明确要求仅第三问进行100次调参、每五次记录一次，选定参数后再20次模拟求平均值；本轮练习调参授权替代旧开发筛选未过不得练习的流程约束，不追认旧门槛通过。',
            'question': 3, 'tuning_trials': 100, 'validation_trials': 20, 'batch_size': 5,
            'profiles': PROFILES, 'fixed_rollout_parameters': fixed, 'tuning_batches': order,
            'selection_rule': ['fewest non-all-clear trials out of 20', 'fewest uncleared sources', 'lowest arithmetic mean of per-case virtual seconds / public true sources', 'lowest worst-case seconds/source', 'profile id'],
            'no_outcome_replacement': True, 'validation_rule': 'Freeze winner after all 100; next 20 new GUI cases use it unchanged; do not reselect from validation.',
            'primary_metric': 'arithmetic mean across cases of virtual_time_s / true_source_count; not ratio of pooled totals',
            'randomization': 'four rounds, each profile one consecutive five-case block, order shuffled with seed 20260911',
            'limitations': 'GUI produces new layouts; profile comparisons are unpaired. Selection averages are optimistically selected. Final 20 are prospective new cases, with no claim that server randomization proves IID.',
            'paper_frozen_results_unchanged': True, 'formal_tests': 0, 'inputs': inputs}
    save(path, plan)
    report()
    print(json.dumps({'plan': str(path), 'batches': order}, ensure_ascii=False), flush=True)


def validate():
    plan = load(OUT / 'plan.json')
    assert plan['scope'] == 'Q3_PRACTICE_TUNING_ONLY' and plan['question'] == 3
    for item in plan['inputs']:
        path = (ROOT / item['path']).resolve()
        if ROOT not in path.parents or not path.is_file() or sha(path) != item['sha256']:
            raise RuntimeError('Frozen input changed: ' + item['path'])
    return plan


def configure(profile):
    helper = importlib.import_module('q3_cost_rollout_v2')
    helper.MOVE_SLACK, helper.MIN_PREDICTED_SAVING = 100.0, 5.0
    for key, value in profile['parameters'].items():
        if key not in ('MOVE_SLACK', 'MIN_PREDICTED_SAVING'):
            raise ValueError('Unsupported tuning parameter')
        setattr(helper, key, value)
    return importlib.import_module(profile['module']).run_search


def preflight():
    plan = validate()
    from evaluate_q3_robust_parallax_development import run, contains_point
    cases = load(ROOT / 'results/feasible_q3/cases.json')[:36]
    rows = []
    for profile in plan['profiles']:
        policy = configure(profile)
        for case in cases:
            metrics, state, actions = run(policy, case)
            errors = []
            truths = {s['channel']: (s['x'], s['y']) for s in case['sources']}
            for event in state.get('cost_rollout_events', []):
                if not contains_point(event['before_polygon'], truths[event['channel']], tolerance=1e-5):
                    errors.append('true source absent from pre-measure polygon')
                if event['used_alternative']:
                    if max(math.dist(event['selected_point'], v) for v in event['before_polygon']) > 999.9 + 1e-7:
                        errors.append('reception certificate')
                    if event['response'].get('measure_result') == 'no_signal':
                        errors.append('guaranteed reception lost')
                    if math.dist(event['selected_point'], event['current_position']) > math.dist(event['original_center'], event['current_position']) + profile['parameters']['MOVE_SLACK'] + 1e-7:
                        errors.append('movement slack')
                    if event['original_score_s'] - event['selected_score_s'] < profile['parameters']['MIN_PREDICTED_SAVING'] - 1e-7:
                        errors.append('adoption threshold')
            rows.append({'profile': profile['id'], 'case_id': case['case_id'], 'pass': metrics['all_checks_pass'] and not errors, 'errors': errors, 'metrics': metrics})
        print(json.dumps({'preflight_profile': profile['id'], 'passed': all(r['pass'] for r in rows if r['profile'] == profile['id'])}), flush=True)
    validate()
    result = {'scope': 'KNOWN_SYNTHETIC_CASES_SAFETY_PREFLIGHT_NOT_120_LIVE_RUNS', 'plan_sha256': sha(OUT/'plan.json'), 'all_pass': all(r['pass'] for r in rows), 'rows': rows, 'created_at': now()}
    save(OUT/'preflight.json', result)
    if not result['all_pass']:
        raise RuntimeError('Preflight safety check failed')


def command(args):
    process = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace', creationflags=subprocess.CREATE_NO_WINDOW)
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    return process.stdout.strip()


def ui(action, case=None):
    if action not in ('read', 'start-q3', 'confirm', 'return', 'export'):
        raise ValueError('Only Q3 practice actions permitted')
    args = [PWSH, '-NoProfile', '-File', str(UI/'practice_ui.ps1'), '-Action', action]
    if case:
        args += ['-CaseId', case]
    return json.loads(command(args))


def wait_ui(predicate, timeout=45):
    deadline = time.monotonic() + timeout
    while True:
        state = ui('read')
        if predicate(state):
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError('Practice UI state timeout; no automatic replacement')
        time.sleep(.3)


def log_metrics(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    unique = set()
    point, channel = (0.0, 0.0), 1
    distance, measures, switches, failed, cleared = 0.0, 0, 0, 0, set()
    for row in records:
        reply, request = row.get('response', {}), row.get('request', {})
        request_id = request.get('request_id')
        if reply.get('accepted') is not True or request_id in unique:
            continue
        unique.add(request_id)
        if row['path'] not in ('/measure', '/clear'):
            continue
        target = (request['position']['x'], request['position']['y'])
        distance += math.dist(point, target)
        point = target
        if row['path'] == '/measure':
            measures += 1
            switches += request['channel'] != channel
            channel = request['channel']
        elif reply.get('clear_result') == 'success':
            cleared.add(request['channel'])
        else:
            failed += 1
    return {'movement_m': distance, 'measure_count': measures, 'switch_count': switches, 'failed_clear_count': failed, 'cleared_channels': sorted(cleared),
            'reconstructed_virtual_time_s': distance/5 + 5*measures + 2*switches + 3*failed + 5*len(cleared)}


def execute_policy(folder, profile, robot):
    from simulator_client import SimulatorClient
    from practice_runner import RetryingClient, _call_with_pending_retries
    policy = configure(profile)
    client = SimulatorClient(robot, port=2026, log_path=folder/'client.jsonl')
    state = None
    error = exit_error = None
    started = time.monotonic()
    try:
        _call_with_pending_retries(client, 'enter')
        state = policy(RetryingClient(client), question=3, method='main')
    except Exception as caught:
        error = f'{type(caught).__name__}: {caught}'
    finally:
        if client.entered and not client.ended:
            try:
                _call_with_pending_retries(client, 'force_exit' if client.uncertain else 'exit')
            except Exception as caught:
                exit_error = f'{type(caught).__name__}: {caught}'
    metrics = log_metrics(client.log_path)
    clock_error = None if client.virtual_time_s is None else metrics['reconstructed_virtual_time_s'] - client.virtual_time_s
    completed = bool(state and state.get('completed') and not error and not exit_error)
    summary = {'scope': 'Q3_PRACTICE_TUNING_ONLY', 'question': 3, 'practice_only': True, 'official_result': False,
               'profile': profile, 'policy_sha256': sha(SHARED/(profile['module']+'.py')), 'plan_sha256': sha(OUT/'plan.json'),
               'status': 'completed' if completed else 'failed_or_unresolved', 'virtual_time_s': client.virtual_time_s,
               'cleared_channel_count': len(metrics['cleared_channels']), 'true_source_count': None, 'policy': state,
               'client_log_path': str(client.log_path), 'log_metrics': metrics, 'clock_reconstruction_error_s': clock_error,
               'local_elapsed_s': time.monotonic()-started, 'error': error, 'exit_error': exit_error, 'exited': client.ended}
    save(folder/'summary.json', summary)
    return summary


def trial(batch, index, profile, robot):
    folder = OUT/'trials'/f"b{batch['batch']:02}_r{index:02}"
    journal = folder/'journal.json'
    if journal.exists():
        history = load(journal)
        if history['stage'] == 'exported':
            return load(folder/'summary.json')
        raise RuntimeError('An unfinished attempt exists; inspect and resume that case, do not replace: '+str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    history = {'batch': batch, 'index': index, 'profile': profile['id'], 'created_at': now(), 'stage': 'start_requested'}
    save(journal, history)
    ui('start-q3')
    ready = wait_ui(lambda s: s['waiting_for_enter'] and len(s['case_ids']) == 1 and s['case_ids'][0] != 'XXXX-XXXX-XXXX-XXXX')
    case = ready['case_ids'][0]
    used = [load(p).get('case_id') for p in (OUT/'trials').glob('*/journal.json') if p != journal]
    if case in used:
        raise RuntimeError('Duplicate GUI case; no enter sent')
    history.update(stage='ready', case_id=case)
    save(journal, history)
    summary = execute_policy(folder, profile, robot)
    summary.update(batch=batch['batch'], stage=batch['stage'], trial_index=index, case_id=case)
    save(folder/'summary.json', summary)
    history['stage'] = 'executed'
    save(journal, history)
    if not summary['exited']:
        raise RuntimeError('Robot exit unresolved; recorded attempt retained')
    end = wait_ui(lambda s: s['case_ids'] == [case] and s['test_ended'] and s['completion_titles'] == ['问题3演练测试完成'])
    save(folder/'public_ui_after_exit.json', end)
    counts = set()
    for text in end['source_texts']:
        match = re.search(r'(?:共|本次案例含干扰源)\s*(\d+).*?全向\s*(\d+).*?定向\s*(\d+)', text)
        if match:
            counts.add(tuple(map(int, match.groups())))
    if len(counts) != 1:
        raise RuntimeError('Ambiguous public completion source count')
    total, omni, directional = counts.pop()
    if total != omni or directional != 0 or total <= 0:
        raise RuntimeError('Completion is not an all-omnidirectional Q3 case')
    command([sys.executable, '-X', 'utf8', str(UI/'capture_window.py')])
    shutil.copyfile(UI/'window.png', folder/'completion.png')
    summary.update(true_source_count=total, true_omnidirectional_count=omni, true_directional_count=directional,
                   s_per_true_source=summary['virtual_time_s']/total, all_cleared=summary['status']=='completed' and summary['cleared_channel_count']==total,
                   source_count_observed_after_exit=True, completion_screenshot_sha256=sha(folder/'completion.png'))
    save(folder/'summary.json', summary)
    history['stage'] = 'recorded'
    save(journal, history)
    ui('confirm')
    time.sleep(.25)
    ui('return')
    time.sleep(.4)
    ui('export', case)
    command([PWSH, '-NoProfile', '-File', str(UI/'save_export_dialog.ps1')])
    exports = list((ROOT/'results/official_exports').glob('*'+case+'.jlog'))
    if len(exports) != 1 or exports[0].stat().st_size < 8:
        raise RuntimeError('Practice log export missing or ambiguous')
    summary.update(export_path=exports[0].relative_to(ROOT).as_posix(), export_sha256=sha(exports[0]))
    save(folder/'summary.json', summary)
    history.update(stage='exported', completed_at=now())
    save(journal, history)
    print(json.dumps({'batch': batch['batch'], 'run': index, 'case': case, 'profile': profile['id'], 'cleared': summary['cleared_channel_count'], 'total': total, 's_per_source': summary['s_per_true_source'], 'status': summary['status']}, ensure_ascii=False), flush=True)
    if abs(summary['clock_reconstruction_error_s']) > 1e-5:
        raise RuntimeError('Clock reconstruction mismatch; attempt retained')
    return summary


def statistics_for(rows):
    if not rows:
        return {'n': 0}
    values = [s['s_per_true_source'] for s in rows]
    sd = statistics.stdev(values) if len(values) > 1 else None
    mean = statistics.mean(values)
    half = 2.093024054 * sd / math.sqrt(20) if len(values) == 20 else None
    return {'n': len(rows), 'all_clear_trials': sum(s['all_cleared'] for s in rows),
            'cleared_sources': sum(s['cleared_channel_count'] for s in rows), 'true_sources': sum(s['true_source_count'] for s in rows),
            'mean_s_per_source': mean, 'median_s_per_source': statistics.median(values), 'sample_sd': sd, 'worst_s_per_source': max(values),
            'pooled_s_per_source': sum(s['virtual_time_s'] for s in rows)/sum(s['true_source_count'] for s in rows),
            'descriptive_t95_ci_under_iid_assumption': None if half is None else [mean-half, mean+half]}


def all_rows():
    return [load(p.parent/'summary.json') for p in sorted((OUT/'trials').glob('*/journal.json')) if load(p)['stage'] == 'exported']


def report():
    plan = load(OUT/'plan.json')
    rows = all_rows()
    tuning = [r for r in rows if r['stage']=='tuning']
    validation = [r for r in rows if r['stage']=='validation']
    groups = {p['id']: statistics_for([r for r in tuning if r['profile']['id']==p['id']]) for p in plan['profiles']}
    data = {'updated_at': now(), 'scope': plan['scope'], 'tuning_completed': len(tuning), 'validation_completed': len(validation),
            'groups': groups, 'validation': statistics_for(validation), 'formal_tests': 0,
            'winner': load(OUT/'selection.json') if (OUT/'selection.json').exists() else None}
    save(OUT/'progress.json', data)
    lines = ['# 第三问：100 次调参＋20 次新案例验证', '', f"更新时间：{data['updated_at']}", '',
             f"已完成调参 {len(tuning)}/100 次，验证 {len(validation)}/20 次。每五次一批，全部为练习测试。", '',
             '| 组别 | 参数 | 已跑 | 全清场次 | 平均秒/源 | 最慢秒/源 |', '|---|---|---:|---:|---:|---:|']
    for p in plan['profiles']:
        s = groups[p['id']]
        params = '原冻结方案' if not p['parameters'] else f"额外移动≤{p['parameters']['MOVE_SLACK']:g} m；预测节省≥{p['parameters']['MIN_PREDICTED_SAVING']:g} s"
        lines.append(f"| {p['id']} {p['label']} | {params} | {s['n']} | {s.get('all_clear_trials', 0)} | {s.get('mean_s_per_source', 0):.2f} | {s.get('worst_s_per_source', 0):.2f} |")
    lines += ['', '先比较全清场次和未清源数，再比较逐场 T/N 的算术平均。调参结果存在优胜者偏差，最终 20 场单独报告；各组为不同的新案例，不当作同场配对比较。', '', '## 每五次记录', '', '| 批次 | 阶段 | 参数 | 场次 | 全清 | 平均秒/源 |', '|---|---|---|---:|---:|---:|']
    for number in sorted(set(r['batch'] for r in rows)):
        part = [r for r in rows if r['batch']==number]
        s = statistics_for(part)
        if len(part)==5:
            save(OUT/'batches'/f'b{number:02}.json', {'batch': number, 'stage': part[0]['stage'], 'profile': part[0]['profile'], 'statistics': s,
                'trials': [{k:r[k] for k in ('case_id','trial_index','true_source_count','cleared_channel_count','virtual_time_s','s_per_true_source','all_cleared','export_path','export_sha256')} for r in part]})
        lines.append(f"| {number} | {part[0]['stage']} | {part[0]['profile']['id']} | {len(part)} | {s['all_clear_trials']} | {s['mean_s_per_source']:.2f} |")
    if data['winner']:
        lines += ['', '选定参数：'+data['winner']['profile']['id']+' '+data['winner']['profile']['label']+'。选择后不再调整。']
    if validation:
        s = data['validation']
        lines += ['', f"验证：{s['all_clear_trials']}/{s['n']} 场全清，平均 **{s['mean_s_per_source']:.2f} 秒/源**，标准差 {s['sample_sd'] or 0:.2f}，最慢 {s['worst_s_per_source']:.2f}。"]
        if s['descriptive_t95_ci_under_iid_assumption']:
            lo, hi = s['descriptive_t95_ci_under_iid_assumption']
            lines += [f"假定案例独立同分布时的描述性 t 区间：[{lo:.2f}, {hi:.2f}] 秒/源；服务器生成机制未经独立验证。"]
    lines += ['', '原论文冻结结果保持为 244.67 秒/源；本实验记录独立保存。案例编码、公开回执、完成截图、参数及代码哈希、导出的 .jlog 均在本目录或 export_path 指定位置可追溯。不读取 .jlog 内部真值。', '']
    (OUT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return data


def select():
    plan = validate()
    if (OUT/'selection.json').exists():
        return load(OUT/'selection.json')
    rows = [r for r in all_rows() if r['stage']=='tuning']
    if len(rows)!=100 or any(sum(r['profile']['id']==p['id'] for r in rows)!=20 for p in plan['profiles']):
        raise RuntimeError('Selection requires all 100 cases and 20 per profile')
    ranking = []
    for p in plan['profiles']:
        part = [r for r in rows if r['profile']['id']==p['id']]
        stats = statistics_for(part)
        key = [20-stats['all_clear_trials'], stats['true_sources']-stats['cleared_sources'], stats['mean_s_per_source'], stats['worst_s_per_source'], p['id']]
        ranking.append({'profile': p, 'statistics': stats, 'ranking_key': key})
    ranking.sort(key=lambda x:x['ranking_key'])
    result = {'selected_at': now(), 'plan_sha256': sha(OUT/'plan.json'), 'profile': ranking[0]['profile'], 'ranking': ranking,
              'selection_case_ids': [r['case_id'] for r in rows], 'validation_cases_observed': 0, 'validation_trials': 20,
              'selection_rule': plan['selection_rule']}
    save(OUT/'selection.json', result)
    print(json.dumps({'selection': result['profile'], 'tuning_statistics': ranking[0]['statistics']}, ensure_ascii=False), flush=True)
    return result


def run_campaign(robot, max_batches=None):
    plan = validate()
    check = load(OUT/'preflight.json')
    if not check['all_pass'] or check['plan_sha256'] != sha(OUT/'plan.json'):
        raise RuntimeError('Current safety preflight missing')
    ran = 0
    for number in range(1,25):
        validate()
        if number <= 20:
            batch = plan['tuning_batches'][number-1]
            profile = next(p for p in plan['profiles'] if p['id']==batch['profile'])
        else:
            winner = select()
            profile = winner['profile']
            batch = {'batch': number, 'profile': profile['id'], 'stage': 'validation', 'count': 5}
        if all((OUT/'trials'/f'b{number:02}_r{i:02}'/'journal.json').exists() and load(OUT/'trials'/f'b{number:02}_r{i:02}'/'journal.json')['stage']=='exported' for i in range(1,6)):
            continue
        print(json.dumps({'batch_start': number, 'profile': profile, 'stage': batch['stage']}, ensure_ascii=False), flush=True)
        for index in range(1,6):
            trial(batch, index, profile, robot)
        status = report()
        print(json.dumps({'batch_complete': number, 'statistics': load(OUT/'batches'/f'b{number:02}.json')['statistics'],
                          'tuning_completed': status['tuning_completed'], 'validation_completed': status['validation_completed']}, ensure_ascii=False), flush=True)
        ran += 1
        if max_batches and ran >= max_batches:
            return
    print(json.dumps({'campaign_complete': True, 'result': report()}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init','preflight','run','report','select'])
    parser.add_argument('--practice-confirmed', action='store_true')
    parser.add_argument('--robot-id')
    parser.add_argument('--max-batches', type=int)
    args = parser.parse_args()
    if args.action=='init': init()
    elif args.action=='preflight': preflight()
    elif args.action=='report': print(json.dumps(report(), ensure_ascii=False))
    elif args.action=='select': select()
    elif args.action=='run':
        if not args.practice_confirmed or not args.robot_id:
            parser.error('Live practice requires explicit --practice-confirmed and --robot-id')
        try:
            run_campaign(args.robot_id, args.max_batches)
        except Exception as error:
            save(OUT/'last_stop.json', {'at': now(), 'error': f'{type(error).__name__}: {error}', 'replacement_trials': 0})
            report()
            raise


if __name__=='__main__':
    main()
