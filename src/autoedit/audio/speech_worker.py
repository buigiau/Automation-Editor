"""Owned, persistent ASR subprocess. The parent can enforce a real deadline."""
import contextlib
import json
import sys


def main():
    from .source_speech import transcribe_source
    from autoedit.runtime import release_inference
    output = sys.stdout
    for line in sys.stdin:
        request = json.loads(line)
        if request['command'] == 'close':
            break
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if request['command'] == 'release':
                    release_inference({})
                    result = None
                else:
                    request['arguments']['ranges'] = [tuple(r) for r in request['arguments']['ranges']]
                    result = transcribe_source(**request['arguments'])
            response = {'result': result}
        except Exception as exc:
            response = {'error': type(exc).__name__ + ': ' + str(exc)}
        output.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + '\n')
        output.flush()


if __name__ == '__main__':
    main()
