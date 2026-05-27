#!/usr/bin/env python3
import requests

response = requests.post(
    "http://localhost:8000/message",
    data={"message": "What is CQRS?", "session_id": "quick_test"},
    timeout=60
)

result = response.json()
end_msg = result.get("endMessage", "")

print("=" * 70)
print("RESPUESTA:")
print("=" * 70)
print(end_msg[:1000])
print("\n" + "=" * 70)

if "Video Results" in end_msg or ".mp4" in end_msg:
    print("✓ EVRAG FUNCIONANDO - Se encontraron resultados de video!")
else:
    print("✗ EVRAG NO ACTIVO - No hay resultados de video")

print("=" * 70)
