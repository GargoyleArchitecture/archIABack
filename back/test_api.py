import requests
import json
resp = requests.post('http://localhost:8000/message', data={'message': 'explicame latencia usando RAG y EVRAG', 'session_id': 'test-evrag-4', 'rag_mode': 'both', 'evrag_mode': 'visual'}, stream=True)
for line in resp.iter_lines():
    if line:
        try:
            data = line.decode('utf-8')
            if data.startswith('data: ') and data != 'data: [DONE]':
                j = json.loads(data[6:])
                print(j)
        except Exception as e:
            print(e)
