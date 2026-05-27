import requests
import json
resp = requests.post('http://localhost:8000/message', data={'message': 'dime que es latencia, en teoria', 'session_id': 'test-evrag-5', 'rag_mode': 'video', 'evrag_mode': 'hybrid'}, stream=True)
for line in resp.iter_lines():
    if line:
        try:
            data = line.decode('utf-8')
            if data.startswith('data: ') and data != 'data: [DONE]':
                j = json.loads(data[6:])
                print(j['role'], j['name'])
                if j['name'] == 'unifier':
                    print(j['content'])
        except Exception as e:
            pass
