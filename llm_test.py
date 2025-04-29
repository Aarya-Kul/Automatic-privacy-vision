import requests

test_prompt = """
You are determining the privacy sensitivity of regions detected in images.

Region class: license_plate
Detected text: ["ABC123"]

Output a single number between 0.0 and 1.0.
Only output the number, nothing else.
"""

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3",
        "messages": [{"role": "user", "content": test_prompt}],
        "options": {"temperature": 0.0},
        "stream": False
    },
    timeout=10,
)

output = response.json()["message"]["content"].strip()
print(f"LLama 3 replied with: {output}")