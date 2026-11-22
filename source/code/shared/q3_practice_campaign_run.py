"""Observation-only correction for the prospectively frozen Q3 campaign.

The simulator charges one second per measurement-channel change. The original
campaign's independent cost checker used two; public simulator times, policies,
profile parameters, order and selection rule are unchanged. Keep original code
and first-case evidence intact and record this correction separately.
"""
from pathlib import Path
import sys
import q3_practice_campaign as campaign

original_log_metrics = campaign.log_metrics


def corrected_log_metrics(path):
    metrics = original_log_metrics(path)
    metrics['reconstructed_virtual_time_s'] -= metrics['switch_count']
    metrics['switch_unit_cost_s'] = 1
    metrics['observer_correction'] = 'observer_cost_v2_switch_unit_1'
    return metrics


campaign.log_metrics = corrected_log_metrics


def main():
    campaign.validate()
    path = campaign.OUT/'observer_correction.json'
    evidence = {
        'scope': 'OBSERVATION_ONLY_CORRECTION', 'created_at': campaign.now(),
        'reason': 'Independent checker used two seconds per channel change; public consecutive origin measurements go from 5 to 11 seconds, establishing 5 seconds measurement plus 1 second switching.',
        'original_code_sha256': campaign.sha(Path(campaign.__file__)),
        'corrected_runner_sha256': campaign.sha(Path(__file__)),
        'frozen_policy_and_parameters_unchanged': True, 'plan_sha256_unchanged': campaign.sha(campaign.OUT/'plan.json'),
        'first_case': '6XGH-YM2R-E5SD-RCCJ', 'first_case_preserved_in_tuning': True,
        'original_first_summary_sha256': campaign.sha(campaign.OUT/'trials/b01_r01/summary.json'),
        'corrected_first_metrics': corrected_log_metrics(campaign.OUT/'trials/b01_r01/client.jsonl'),
    }
    first = campaign.load(campaign.OUT/'trials/b01_r01/summary.json')
    evidence['corrected_first_clock_error_s'] = evidence['corrected_first_metrics']['reconstructed_virtual_time_s']-first['virtual_time_s']
    if path.exists():
        previous = campaign.load(path)
        for key in ('corrected_runner_sha256','original_code_sha256','plan_sha256_unchanged','original_first_summary_sha256'):
            if previous[key] != evidence[key]:
                raise RuntimeError('Observer correction binding changed: '+key)
    else:
        campaign.save(path, evidence)
    campaign.main()


if __name__ == '__main__':
    main()
