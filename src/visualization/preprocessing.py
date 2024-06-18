import pandas as pd

def pre_processing(fusion):
    
    # Fusionner les colonnes description et designation
    fusion['descriptif'] = fusion['description'].astype(str).replace("nan", "") + " " + fusion['designation'].astype(str)
    fusion = fusion.drop(['designation', 'description'], axis=1)

    return None