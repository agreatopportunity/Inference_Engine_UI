# Step 1: Install Gradio
pip install gradio

# Step 2: The "UI" Script (app.py)Here is the Universal Inference UI.

# How to use this on ANY machine:
# Scenario A: The RTX 4060 Ti / 4090 / 5090 (Modern) Just run it. It defaults to GPU 0.


```
python3 app_universal.py
Result: It will auto-detect "Ampere/Ada", enable bfloat16, enable Flash Attention, and run at max speed.
```
# Scenario B: The Titan V (Legacy) If the Titan V is the second GPU (like in your current setup):


```
python3 app_universal.py --device_id 1
Result: It will auto-detect "Volta", switch to float16, disable Flash Attention, and run safely.
```
# Scenario C: A Single-GPU Titan V Machine If you move the Titan V to its own box where it is the only GPU:


```
python3 app_universal.py
Result: It detects GPU 0 is Volta, and auto-downgrades settings to work perfectly.
```
