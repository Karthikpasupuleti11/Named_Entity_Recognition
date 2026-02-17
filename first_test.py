from gliner import GLiNER

# 1. Load the model (it downloads automatically from the internet)
# We use the 'small' version because it's fast and easy for beginners.
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# 2. Define your text
text = "Tony Stark built the first Iron Man suit in a cave in Afghanistan."

# 3. Define what you want to find (You can invent ANY label here!)
labels = ["Superhero", "Invention", "Location", "Material"]

# 4. Run the prediction
entities = model.predict_entities(text, labels)

# 5. Print the results
for entity in entities:
    print(f"{entity['text']} => {entity['label']}")