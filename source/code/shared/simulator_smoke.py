"""Practice-only connectivity check; no search strategy or formal-test runner."""
import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, ProxyHandler, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--port', type=int, default=2026)
    parser.add_argument('--practice-confirmed', action='store_true', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    folder = root / 'results' / 'connectivity'
    folder.mkdir(parents=True, exist_ok=True)
    logfile = folder / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.jsonl')
    opener = build_opener(ProxyHandler({}))

    def call(endpoint, **fields):
        payload = dict(arena_id='default', robot_id=args.robot_id,
                       request_id=uuid.uuid4().hex, **fields)
        encoded = json.dumps(payload).encode('utf-8')
        for attempt in range(3):
            request = Request(f'http://127.0.0.1:{args.port}/{endpoint}',
                              data=encoded, headers={'Content-Type': 'application/json'}, method='POST')
            record = dict(endpoint=endpoint, request=payload, attempt=attempt + 1)
            try:
                with opener.open(request, timeout=5) as response:
                    record['http_status'] = response.status
                    body = json.loads(response.read().decode('utf-8'))
                record['response'] = body
            except HTTPError as error:
                record['http_status'] = error.code
                record['error'] = error.read().decode('utf-8', errors='replace')
                raise
            except (URLError, TimeoutError, ConnectionError) as error:
                record['error'] = str(error)
                if attempt == 2:
                    raise
                time.sleep(0.5)
                continue
            finally:
                with logfile.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            if body.get('accepted') is not True:
                raise RuntimeError(f'{endpoint} rejected: {body}')
            print(endpoint, json.dumps(body, ensure_ascii=False), flush=True)
            return body

    entered = False
    try:
        call('enter')
        entered = True
        call('measure', position={'x': 0, 'y': 0}, channel=1)
    finally:
        if entered:
            call('exit')
        print('Log:', logfile, flush=True)


if __name__ == '__main__':
    main()
