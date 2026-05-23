"""
End-to-end drift detection demo.
Run: make seed && make serve  (in separate terminal)  then: make demo
"""
from __future__ import annotations
import requests
import json
import time
import numpy as np

BASE = "http://localhost:8081"


def post(path, payload):
    return requests.post(f"{BASE}{path}", json=payload).json()


def get(path):
    return requests.get(f"{BASE}{path}").json()


print("\n=== Drift Monitor Demo ===\n")

# 1. Health check
health = get("/health")
print(f"1. Health: {health}")

# 2. Serve 50 predictions from normal distribution
print("\n2. Sending 50 predictions (normal distribution)...")
rng = np.random.default_rng(42)
for i in range(50):
    features = {f"f{j}": round(float(rng.normal(0, 1)), 4) for j in range(8)}
    post("/predict", {"entity_id": f"user_{i:03d}", "features": features})
print("   Done.")

# 3. Inject drifted data — shift mean significantly
print("\n3. Injecting drifted data (shifted distribution)...")
for i in range(50):
    features = {f"f{j}": round(float(rng.normal(5, 1)), 4) for j in range(8)}
    post("/predict", {"entity_id": f"drifted_{i:03d}", "features": features})

time.sleep(1)  # allow background tasks to flush

# 4. Check drift events
print("\n4. Drift events detected:")
events = get("/drift/events?limit=10")
if events:
    for e in events[:5]:
        print(f"   feature={e['feature']}  score={e['drift_score']}  at={e['detected_at']}")
else:
    print("   None yet — background tasks may still be processing")

# 5. Drift summary
print("\n5. Drift summary:")
summary = get("/drift/summary")
print(f"   {json.dumps(summary, indent=4)}")

# 6. Trigger retraining
print("\n6. Triggering retraining...")
result = post("/model/retrain", {})
print(f"   {result}")
time.sleep(3)

# 7. Model versions
print("\n7. Model versions:")
versions = get("/model/versions")
for v in versions:
    active = "← active" if v["is_active"] else ""
    print(f"   {v['version']}  metrics={v['metrics']}  {active}")

print(f"\nAPI docs → http://localhost:8081/docs")
print("\nDone.")
