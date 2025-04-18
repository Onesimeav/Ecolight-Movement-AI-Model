
import pandas as pd

def data_cleaning_process(file_path):
    cleaned_output_path = "data/cleaned_casas_data.csv"

    # Lire les lignes et corriger celles qui ont trop ou trop peu de champs
    corrected_lines = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                corrected_lines.append(parts)
            elif len(parts) > 4:
                # Cas où un champ a une tabulation en trop : on regroupe les champs du milieu
                corrected_lines.append([parts[0], parts[1], " ".join(parts[2:-1]), parts[-1]])
            elif len(parts) < 4:
                # Lignes corrompues (trop courtes), on les ignore
                continue

    # Convertir en DataFrame
    df = pd.DataFrame(corrected_lines, columns=['timestamp', 'location', 'status', 'device'])

    #Etape 2: Retrait des capteurs inutiles
    included_types = ["Control4-Motion", "Control4-LightSensor"]
    filtered_df = df[df['device'].isin(included_types)].reset_index(drop=True)

    # Sauvegarder en CSV
    filtered_df.to_csv(cleaned_output_path, index=False)

data_cleaning_process("data/dirty_data.txt")