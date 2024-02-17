import pickle

if __name__ == '__main__':
    config_path = input("Por favor, escribe la ruta relativa del archivo de configuración (.pickle):\n")
    with open(config_path, 'rb') as f_in:
        config = pickle.load(f_in)

    new_path = input("Por favor, escribe la ruta (pesos) tomando como carpeta base 'model/':\n")

    old_path = config.weights_output_path
    config.weights_output_path = new_path

    # sobrescribe el archivo
    with open(config_path, 'wb') as config_f:
        pickle.dump(config, config_f)

    print("Se corrigió la ruta, antigua: {}, nueva: {}".format(old_path, new_path))
