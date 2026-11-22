"""Synthetic black-box diagnostic arena; not an official simulator replica."""
import hashlib
import math
import random
from dataclasses import dataclass

from simulator_client import HttpResult


@dataclass(frozen=True)
class Source:
    channel: int
    x: float
    y: float
    radius: float
    heading: float | None = None


class OfflineArena:
    """Evaluator owns truth. Policies receive only the normal SimulatorClient."""

    def __init__(self, sources, seed=0, error_mode='fixed_hash'):
        if len({s.channel for s in sources}) != len(sources):
            raise ValueError('source channels must be unique')
        self.sources = {s.channel: s for s in sources}
        self.seed, self.error_mode = seed, error_mode
        self.cleared = set()
        self.position, self.channel = (0.0, 0.0), 1
        self.virtual_time_s = 0.0
        self.entered = self.ended = False
        self.measure_count = self.clear_count = 0
        self.distance_m = self.switch_count = 0
        self.cache = {}

    def _error(self, channel, x, y):
        if self.error_mode == 'positive_edge':
            return 1.0
        if self.error_mode == 'negative_edge':
            return -1.0
        key = f'{self.seed}:{channel}:{x.hex()}:{y.hex()}'.encode()
        integer = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big')
        return 2 * integer / (2**64 - 1) - 1

    def post(self, path, payload):
        request_id = payload['request_id']
        if request_id in self.cache:
            old_path, old_payload, response = self.cache[request_id]
            return response if (path, payload) == (old_path, old_payload) else HttpResult(409, {})
        response = self._execute(path, payload)
        self.cache[request_id] = (path, payload.copy(), response)
        return response

    def _execute(self, path, payload):
        def answer(**extra):
            return HttpResult(200, dict(accepted=True, virtual_time_s=self.virtual_time_s,
                                       real_timestamp_ms=0, **extra))
        if path == '/enter':
            if self.entered:
                return HttpResult(200, dict(accepted=False, virtual_time_s=0))
            self.entered = True
            return answer(remaining_real_duration_s=1200, max_real_duration_s=1200,
                          max_virtual_duration_s=360000)
        if not self.entered or self.ended:
            return HttpResult(200, dict(accepted=False, virtual_time_s=0))
        if path == '/exit':
            self.ended = True
            return answer(exit_reason='user_exit')
        if path not in ('/measure', '/clear'):
            return HttpResult(404, {})
        position = payload['position']
        x, y, channel = float(position['x']), float(position['y']), payload['channel']
        if not (math.isfinite(x) and math.isfinite(y) and max(abs(x), abs(y)) <= 2000000
                and isinstance(channel, int) and not isinstance(channel, bool) and 1 <= channel <= 20):
            return HttpResult(400, {})
        distance = math.dist(self.position, (x, y))
        self.distance_m += distance
        self.virtual_time_s += distance / 5
        self.position = (x, y)
        source = self.sources.get(channel) if channel not in self.cleared else None
        target_distance = math.hypot(x-source.x, y-source.y) if source else math.inf
        if path == '/clear':
            self.clear_count += 1
            success = target_distance <= 20
            self.virtual_time_s += 5 if success else 3
            if success:
                self.cleared.add(channel)
            return answer(clear_result='success' if success else 'no_target_in_range')
        self.measure_count += 1
        switched = channel != self.channel
        self.switch_count += switched
        self.virtual_time_s += 5 + switched
        self.channel = channel
        visible = source is not None and target_distance <= source.radius
        if visible and source.heading is not None:
            heading = math.radians(source.heading)
            visible = math.cos(heading)*(x-source.x) + math.sin(heading)*(y-source.y) >= -1e-10
        if not visible:
            return answer(measure_result='no_signal')
        if target_distance <= 5:
            return answer(measure_result='near')
        angle = math.degrees(math.atan2(source.y-y, source.x-x))
        bearing = round((angle + self._error(channel, x, y)) % 360, 2) % 360
        return answer(measure_result='direction', svd_deg=bearing)


def make_cases(question):
    """Predeclared 12 cases per question; no result-based exclusions or tuning."""
    cases = []
    for index in range(12):
        seed = question * 10000 + index
        rng = random.Random(seed)
        count = (10, 13, 16)[index % 3]
        channels = rng.sample(range(1, 21), count)
        sources = []
        for j, channel in enumerate(channels):
            a = rng.uniform(0, 2*math.pi)
            radius = 1800 * math.sqrt(rng.random())
            x, y = radius*math.cos(a), radius*math.sin(a)
            if index in (3, 4, 5):
                a = 2*math.pi*j/count
                x, y = 1800*math.cos(a), 1800*math.sin(a)
            elif index in (6, 7, 8):
                grid = [(xx, yy) for xx in range(-1800, 1801, 600)
                        for yy in range(-1800, 1801, 600) if xx*xx+yy*yy <= 1800**2]
                x, y = map(float, grid[j % len(grid)])
            elif index in (9, 10, 11) and j == 0:
                x, y = (0.0, 0.0) if index == 9 else (5.0, 0.0) if index == 10 else (20.0, 0.0)
                if channel != 1 and 1 not in channels:
                    channels[j] = channel = 1
            reception = 1000.0 if index % 3 == 0 else 1500.0 if index % 3 == 1 else rng.uniform(1000, 1500)
            # Q4 always has both kinds, including predominantly directional cases.
            directional = question == 4 and (j % 2 == 1 if index % 2 == 0 else j > 0)
            heading = (math.degrees(a) if index in (3, 4, 5) else rng.uniform(0, 360)) if directional else None
            sources.append(Source(channel, float(x), float(y), reception, heading))
        mode = ('fixed_hash', 'positive_edge', 'negative_edge')[index % 3]
        cases.append(dict(case_id=f'synthetic_q{question}_{index:02}', seed=seed,
                          error_mode=mode, sources=sources))
    return cases
