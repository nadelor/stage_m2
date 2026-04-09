import json

filename = "2_merging_cleaning_vital_signs_and_ioa_file_22pel.ipynb"
print(f"⏳ Lecture du fichier {filename}...")

try:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # On parcourt chaque cellule pour vider les sorties (outputs)
    for cell in data.get('cells', []):
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None

    # On réenregistre le fichier "nettoyé"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)

    print("✨ SUCCÈS : Ton notebook est maintenant léger et prêt à être ouvert !")

except Exception as e:
    print(f"❌ Erreur : {e}")