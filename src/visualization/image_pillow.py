def is_valid_image_pillow(file_name):
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False

def get_image_pillow(productid, imageid):
    file_name = "../../data/images/image_train/image_" + imageid.astype('str') + "_product_" + productid.astype('str') + ".jpg"
    if is_valid_image_pillow(file_name):
        return Image.open(file_name)
    else:
        return None

# Fonction de test à supprimer à l'avenir
def get_all_images():
    for product in range(len(X_train)):
        image = get_image_pillow(X_train.loc[product, 'productid'], X_train.loc[product, 'imageid'])
        if image is not None:
            print(image)