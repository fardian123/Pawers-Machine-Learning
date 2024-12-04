import json

# Load the JSON file
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Remove duplicate patterns
for intent in data["intents"]:
    intent["patterns"] = list(set(intent["patterns"]))

# Save the cleaned JSON back to a file
with open("intents_cleaned.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("Duplikasi pada patterns berhasil dihilangkan dan disimpan di intents_cleaned.json")
