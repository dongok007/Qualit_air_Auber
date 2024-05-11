import numpy as np
import pandas as pd

# Charger les données à partir du fichier dataset.csv
df = pd.read_csv('data/clean_dataset.csv')


# Fonction de calcul de l'AQI
def calculate_AQI(NO, NO2, PM10, PM25, CO2):
    # Fonctions de calcul des sous-indices pour chaque polluant
    def calculate_NO_subindex(NO):
        # Calcul du sous-indice pour le NO
        if NO <= 50:
            return NO * 50 / 50
        elif NO <= 100:
            return 50 + (NO - 50) * 50 / 50
        elif NO <= 150:
            return 100 + (NO - 100) * 100 / 50
        elif NO <= 200:
            return 200 + (NO - 150) * 100 / 50
        elif NO <= 300:
            return 300 + (NO - 200) * 100 / 100
        elif NO > 300:
            return 400 + (NO - 300) * 100 / 100
        else:
            return 0

    # Autres fonctions de calcul des sous-indices pour les autres polluants (NO2, PM10, PM25, CO2)...

    # Calcul des sous-indices pour chaque polluant
    NO_subindex = calculate_NO_subindex(NO)
    # Calcul des sous-indices pour les autres polluants (NO2, PM10, PM25, CO2)...

    # Calcul de l'AQI
    AQI = max(NO_subindex, NO2_subindex, PM10_subindex, PM25_subindex, CO2_subindex)

    return AQI


# Création d'une liste pour stocker les AQI calculés
AQI_list = []

# Boucle for pour calculer l'AQI pour chaque ligne du DataFrame
for index, row in df.iterrows():
    NO = row["NO"]
    NO2 = row["NO2"]
    PM10 = row["PM10"]
    PM25 = row["PM2.5"]
    CO2 = row["CO2"]

    AQI = calculate_AQI(NO, NO2, PM10, PM25, CO2)
    AQI_list.append(AQI)

# Ajout de la liste AQI à DataFrame en tant que nouvelle colonne
df["AQI"] = AQI_list

# Afficher les premières lignes du DataFrame avec l'AQI calculé
print(df.head())

# Sauvegarder le DataFrame avec l'AQI calculé dans le fichier supervisé.csv
df.to_csv('data_supervisé', index=False)
