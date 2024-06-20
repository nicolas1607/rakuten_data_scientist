import seaborn as sns
import matplotlib.pyplot as plt

def data_visualisation(fusion):

    # Heatmap : corrélation entre toutes les variables quantitatives
    fusion = fusion.drop(['description'], axis=1)
    fusion = fusion.drop(['designation'], axis=1)
    fig, ax = plt.subplots(figsize = (8,8))
    sns.heatmap(fusion.corr(), ax = ax, cmap = "coolwarm", annot=True)
    plt.title('Corrélation entre toutes les variables quantitatives')
    plt.savefig("reports/figures/heatmap.png", bbox_inches='tight')

    # Histogramme avec estimation de la densité : prdtypecode
    sns.displot(fusion.prdtypecode, bins=20, kde = True, rug=True, color="red")
    plt.title('Répartition des valeurs de prdtypecode avec estimation de la densité')
    plt.savefig("reports/figures/histogramme_avec_estimation_densite.png", bbox_inches='tight')
    
    # Histogramme : prdtypecode

    distribution = fusion['prdtypecode'].value_counts()
    distribution_df = distribution.reset_index()
    distribution_df.columns = ['prdtypecode', 'count']
    distribution_df.plot(kind='bar', x='prdtypecode', y='count')
    plt.xlabel('prdtypecode')
    plt.ylabel('Nombre d\'occurrences')
    plt.title('Répartition des valeurs de prdtypecode')
    plt.savefig("reports/figures/historamme.png", bbox_inches='tight')

    # Nuage de point : productid et prdtypecode
    sns.relplot(x=fusion.productid, y=fusion.prdtypecode)
    plt.title('Catégorie du produit en fonction du productid')
    plt.savefig("reports/figures/scatterplot.png", bbox_inches='tight')

    # Pairplot : corrélation entre productid et prdtypecode
    sns.pairplot(fusion[['productid', 'prdtypecode']], hue="prdtypecode",diag_kind="kde")
    plt.title('Catégorie du produit en fonction du productid')
    plt.savefig("reports/figures/pairplot.png", bbox_inches='tight')


