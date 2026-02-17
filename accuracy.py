import json
import random
from gliner import GLiNER

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
# Path to your trained model (The specific checkpoint)
MODEL_PATH = "/home/cdac/Documents/GLiNER-main/models/checkpoint-7500" 

# Path to your data
DATA_PATH = "nuner_train.json"

# How many examples to test? (Testing 1 million takes too long)
TEST_SIZE = 1000 

# ---------------------------------------------------------
# 2. LOAD THE MODEL
# ---------------------------------------------------------
print(f"Loading model from {MODEL_PATH}...")
# We load_tokenizer=True because we trained it from scratch
model = GLiNER.from_pretrained(MODEL_PATH, load_tokenizer=True)
model.eval() # Switch to "Test Mode"

# ---------------------------------------------------------
# 3. PREPARE THE TEST EXAM
# ---------------------------------------------------------
print(f"Loading data from {DATA_PATH}...")
with open(DATA_PATH, "r") as f:
    full_data = json.load(f)

# We randomly pick 1000 sentences to test
# (In a real scenario, you should use data the model has NEVER seen)
test_data = full_data[:TEST_SIZE]
print(f"Running evaluation on {len(test_data)} examples...")

# ---------------------------------------------------------
# 4. THE EXAM (Evaluation)
# ---------------------------------------------------------
# This function compares the model's predictions vs. the actual answers
results = model.evaluate(
    test_data, 
    flat_ner=True,      # Standard NER (no overlapping entities)
    threshold=0.5,      # Confidence level (50%)
    batch_size=8
)

# ---------------------------------------------------------
# 5. THE REPORT CARD
# ---------------------------------------------------------
print("\n" + "="*30)
print("FINAL RESULTS")
print("="*30)
print(results)