def exploration_donnee(fusion):

    # Afficher les premières lignes du DataFrame
    print(fusion.head())

    # Afficher les informations du DataFrame
    print(fusion.info())

    # Afficher les statistiques descriptives des variables quantitatives
    print(fusion.describe())

    # Afficher les valeurs uniques prises par prdtypecode
    print(fusion.prdtypecode.nunique())

    # Vérifier les doublons
    print(fusion.duplicated().sum())

    # Vérifier les valeurs nulles
    print(fusion.isna().sum())

    return None
    