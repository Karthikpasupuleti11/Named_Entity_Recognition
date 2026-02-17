from gliner import GLiNER

# 1. Load the model from your checkpoint
# This path comes from the folder structure you showed in the image
model_path = "models/checkpoint-15000"

print(f"Loading trained model from {model_path}...")
model = GLiNER.from_pretrained(model_path, load_tokenizer=True)

# 2. Test it with a sentence
# Since you trained on NuNER, it should be good at general entities
text = "Kyrie Irving scored 40 points for the Dallas Mavericks in Texas."

# 3. Ask for specific labels
labels = ["Person", "BasketballTeam", "Location", "Score"]

entities = model.predict_entities(text, labels)

print("\n--- PREDICTIONS ---")
for entity in entities:
    print(f"{entity['text']} => {entity['label']}")