# -------------------------------------------------------------------------------------------------
#   Universidad Rey Juan Carlos - Grado en Ingeniería Informática - Visión Artificial
#
#       Práctica 1 - Detección de Objetos
#
#       Desarrollado por:
#             - Alberto Pérez Pérez (GII + GIS)
#             - Daniel Tolosa Oropesa (GII)
# -------------------------------------------------------------------------------------------------


# -------------------------------------
#                IMPORTS
# -------------------------------------

import source
import argparse

# --------------------------------------
#              MAIN PROGRAM
# --------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="MSER_7_200_2000_0.15",
        help='Detector string name (default: MSER_15_200_2000_0.6)')
    parser.add_argument(
        '--train_path', default="train_jpg", help='Select the training data dir (default: train_jpg)')
    parser.add_argument(
        '--test_path', default="test_alumnos_jpg", help='Select the testing data dir (default: test_alumnos_jpg)')

    args = parser.parse_args()

    mserValues = args.detector.split('_')

    if len(mserValues) != 0 and len(mserValues) == 5 and mserValues[0] == 'MSER' and 0 < int(mserValues[1]) <= 40 and 0 < int(
            mserValues[2]) <= 20000 and 0 < int(mserValues[
                                                    3]) <= 20000 and int(mserValues[2]) <= \
            int(mserValues[3]) and 0 < float(mserValues[4]) <= 1:
        source.test(args.train_path.replace('\\', '/'), args.test_path.replace('\\', '/'),
                    (int(mserValues[1]), int(mserValues[2]), int(mserValues[3]), float(mserValues[4])))
    else:
        print("\nNo se ha especificado el nombre de un detector existente")
        print("        Especificaciones sobre\n"
              "        detectores:.........................................")
        print(
            "                          Todos los detectores usan MSER para encontrar las señales de tráfico. Para especificar\n"
            "                          un detector siga esta nomenclatura:\n"
            "                                 NOMBRE -> 'MSER'\n"
            "                                  DELTA -> Numero entero entre 1 y 40\n"
            "                               MIN AREA -> Numero entero entre 1 y 20000\n"
            "                               MAX AREA -> Numero entero entre 1 y 20000\n"
            "                          MAX VARIATION -> Numero decimal mayor que 0 entre 0 y 1\n"
            "                          (Cada uno de los parámetros irá separado por '_')\n"
            "                          (Tenga en cuenta que MIN AREA >= MAX AREA)\n\n"
            "         Ejemplo de uso: MSER_5_200_3000_0.45")
