import os
import pandas as pd

#Variables Globales
#Especies ya almacenadas en el dataset, para evitar duplicados
stored_species = []

base_dir = os.path.dirname(__file__)
dataset_version = 'FungiTastic-Mini'
#Rutas de Csv y de imagenes
route_csv = os.path.join(base_dir,'FungiTastic' ,'metadata',dataset_version)
route_pic = os.path.join(base_dir,'FungiTastic' ,dataset_version)
#Rutas de test
test_csv = os.path.join(route_csv, dataset_version+'-Test.csv')
test_pic = os.path.join(route_pic, 'test', '300p')
#Rutas de train
train_csv = os.path.join(route_csv, dataset_version+'-Train.csv')
train_pic = os.path.join(route_pic, 'train', '300p')
#Rutas de val
val_csv = os.path.join(route_csv, dataset_version+'-Val.csv')
val_pic = os.path.join(route_pic, 'val', '300p')


#Estructura de carpetas
def mk_folders():
    mk_dir(base_dir, 'dataset')
    mk_dir(os.path.join(base_dir, 'dataset'), 'uppa')
    mk_dir(os.path.join(base_dir, 'dataset', 'uppa'),'images')
    mk_dir(os.path.join(base_dir, 'dataset', 'uppa'),'labels')
    mk_dir(os.path.join(base_dir, 'dataset', 'uppa','images'),'test')
    mk_dir(os.path.join(base_dir, 'dataset', 'uppa','labels'),'test')
    mk_dir(os.path.join(base_dir, 'dataset', 'uppa','images'),'train')
    mk_dir(os.path.join(base_dir, 'dataset', 'uppa','labels'),'train')
    mk_dir(os.path.join(base_dir, 'dataset', 'uppa','images'),'val')
    mk_dir(os.path.join(base_dir, 'dataset', 'uppa','labels'),'val')

#Funcion para crear carpetas
def mk_dir(root,name):
    target_dir = os.path.join(root, name)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

#Funcion para agregar datos a stored_species
def add_species(species):
    for sp in species:
        if sp not in stored_species:
            stored_species.append(sp)

#Funcion para escribir archivos
def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

#Funcion para la creacion de archivo Yaml para el dataset, con la estructura requerida por Ultralytics
def make_yaml():
    #Datos base para el archivo Yaml
    content_yaml = 'path: uppa\n' \
                'train: images/train\n' \
                'val: images/val\n' \
                'test: images/test\n' \
                'names:\n'
    #Agregando las especies al archivo Yaml, con su indice correspondiente
    for sp in stored_species:
        content_yaml += f"{stored_species.index(sp)}: {sp}\n"
        print(f"{stored_species.index(sp)}: {sp}")
    #Escribir el archivo Yaml en la carpeta del dataset
    write_file(os.path.join(base_dir, 'dataset', 'uppa','uppa.yaml'), content_yaml)

#Funcion para la creacion de archivos Txt para cada imagen, con su respectiva etiqueta, con la estructura requerida por Ultralytics
def make_txt(path, species):
    content_txt = f"{stored_species.index(species)} 0.5 0.5 1 1"
    write_file(path, content_txt)

#Funcion para la creacion de archivos Txt para cada imagen, con su respectiva etiqueta, con la estructura requerida por Ultralytics, para cada conjunto de datos (test, train, val)
def make_txts(data, path_img, path_label):
    for index, row in data.iterrows():
        img_name = row['filename'].replace('.JPG', '')

        species = row['species']
        img_path = os.path.join(path_img, img_name+".JPG")
        label_path = os.path.join(path_label, img_name + '.txt')
        if os.path.exists(img_path):
            make_txt(label_path, species)
        else:
            print(f"Imagen no encontrada: {img_path}")


#Funcion para mover las imagenes a la carpeta del dataset, con la estructura requerida por Ultralytics
def move_images(path_source, path_dst):
    for img in os.listdir(path_source):
        if img.endswith('.JPG'):
            src = os.path.join(path_source, img)
            dst = os.path.join(path_dst, img)
            os.rename(src, dst)
#main
mk_folders()
#inicializar Csv's 
test_data = pd.read_csv(test_csv)
train_data = pd.read_csv(train_csv)
val_data = pd.read_csv(val_csv)
#Agregar especies a stored_species
add_species(test_data['species'].unique())
add_species(train_data['species'].unique())
add_species(val_data['species'].unique())
#Creacion de archivo Yaml
make_yaml()
#Creacion de archivos Txt para cada imagen, con su respectiva etiqueta
make_txts(test_data, test_pic, os.path.join(base_dir, 'dataset', 'uppa','labels','test'))
make_txts(train_data, train_pic, os.path.join(base_dir, 'dataset', 'uppa','labels','train'))
make_txts(val_data, val_pic, os.path.join(base_dir, 'dataset', 'uppa','labels','val'))
#Mover las imagenes a la carpeta del dataset
move_images(test_pic, os.path.join(base_dir, 'dataset', 'uppa','images','test'))
move_images(train_pic, os.path.join(base_dir, 'dataset', 'uppa','images','train'))
move_images(val_pic, os.path.join(base_dir, 'dataset', 'uppa','images','val'))