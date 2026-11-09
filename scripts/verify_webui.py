"""Non-destructive smoke checks; no account creation or model downloads."""
import json
import subprocess
import urllib.request

code = r'''
import json
import urllib.request
base = 'http://llama:8080'
def request(path, data=None):
    return urllib.request.urlopen(urllib.request.Request(base + path,
        data=json.dumps(data).encode() if data else None,
        headers={'Content-Type': 'application/json'}), timeout=120)
models = json.load(request('/v1/models'))['data']
assert models, 'No models discovered'
model = models[0]['id']
print('Model discovery:', model)
payload = {'model': model, 'messages': [{'role':'user','content':'Reply with the word hello.'}], 'max_tokens': 24, 'temperature': 0}
result = json.load(request('/v1/chat/completions', payload))
assert result['choices'][0]['message']['content'], result
print('Chat completion:', result['choices'][0]['message']['content'])
payload['stream'] = True
payload['stream_options'] = {'include_usage': True}
with request('/v1/chat/completions', payload) as response:
    assert 'text/event-stream' in response.headers['Content-Type']
    chunks = 0
    usage = None
    done = False
    for line in response:
        if line.startswith(b'data: '):
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            chunks += 1
    assert done and chunks, 'Incomplete SSE stream'
    assert usage and usage['prompt_tokens'] > 0 and usage['completion_tokens'] > 0
    print('Streaming:', chunks, 'SSE events and [DONE]')
    print('Token usage:', json.dumps(usage))
'''
subprocess.run(['docker', 'compose', 'exec', '-T', 'open-webui', 'python', '-c', code], check=True)
for path in ('/health', '/'):
    with urllib.request.urlopen('http://127.0.0.1:3000' + path, timeout=30) as response:
        assert response.status == 200
        body = response.read()
        if path == '/':
            assert b'<html' in body.lower()
        print('Open WebUI', path, 'HTTP', response.status)
print('Browser login/chat still requires your Open WebUI account; no account was created by this check.')
