# -------------------------------------------------------------------------------------------------
#   Universidad Rey Juan Carlos - Grado en Ingeniería Informática - Visión Artificial
#
#       Práctica 2 - Reconocimiento de Objetos
#
#       Desarrollado por:
#             - Alberto Pérez Pérez (GII + GIS)
#             - Daniel Tolosa Oropesa (GII)
# -------------------------------------------------------------------------------------------------


# --------------------------------------
#                IMPORTS
# --------------------------------------

import constants
import source
import argparse


# --------------------------------------
#           SUPPORT FUNCTIONS
# --------------------------------------

def checkParams(mserParams, classifierParams):
    mserCheckIn = len(mserParams) != 0 and len(mserParams) == 5 and mserParams[0] == 'MSER' and 0 < int(mserParams[1]) <= 40 and 0 < int(
            mserParams[2]) <= 20000 and 0 < int(mserParams[3]) <= 20000 and int(mserParams[2]) <= int(mserParams[3]) and 0 < float(mserParams[4]) <= 1
    classifierCheckIn = len(classifierParams) != 0 and classifierParams[0] in constants.FEATURE_DESCRIPTORS and classifierParams[1] in constants.DIM_REDUCERS and classifierParams[2] in constants.CLASSIFIERS
    return mserCheckIn and classifierCheckIn


# --------------------------------------
#              MAIN PROGRAM
# --------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Entrena clasificador sobre train y ejecuta test de clasificación sobre imágenes de test')
    parser.add_argument(
        '--train_path', type=str, default="./train_jpg", help='Path al directorio de imágenes de train (default: ./train_jpg)')
    parser.add_argument(
        '--test_path', type=str, default="./test_alumnos_jpg", help='Path al directorio de imágenes de test (default: ./test_alumnos_jpg)')
    parser.add_argument(
        '--detector', type=str, default="MSER_7_200_2000_1",
        help='String con el nombre del detector MSER (default: MSER_15_200_2000_1)')
    parser.add_argument(
        '--classifier', type=str, default="HOG_LDA_BAYES", help='String con el nombre del clasificador')

    args = parser.parse_args()

    mserParams = args.detector.split('_')
    classifierParams = args.classifier.split('_')

    if checkParams(mserParams, classifierParams):
        mserValues = int(mserParams[1]), int(mserParams[2]), int(mserParams[3]), float(mserParams[4])
        print("\nDesea realizar validación sobre el conjunto de datos de entrenamiento...")
        print("SI - s")
        print("NO - n")
        answer = input()
        if answer == "s":
            source.testValidation(args.train_path.replace('\\', '/'), mserValues, classifierParams, 0.1, 0.5)
        elif answer == "n" or answer == "s":
            print("\nNo se ha reconocido una respuesta correcta. Se ejecutará sin validación.")
        # source.test(args.train_path.replace('\\', '/'), mserValues, classifierParams, 0.1, 0.5)
    else:
        print("\nNo se ha especificado el nombre de un detector y/o clasificador existentes")
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
        print("\n        Especificaciones sobre\n"
              "        clasificadores:.....................................")
        print(
            "                          Todos los clasificadores usan:\n"
            "                               -> Algoritmo de descripción de características\n"
            "                                      -> Disponibles:", constants.FEATURE_DESCRIPTORS, "\n"
            "                               -> Algoritmo de reducción de dimensionalidad\n"
            "                                      -> Disponibles:", constants.DIM_REDUCERS, "\n"
            "                               -> Algoritmo de clasificación\n"
            "                                      -> Disponibles:", constants.CLASSIFIERS, "\n"
            "                          (Cada uno de los parámetros irá separado por '_')\n"
            "                          (Hay que usar uno y solo un tipo de algoritmo para especificar\n"
            "                           un clasificador y en el orden anteriormente indicado)\n\n"
            "         Ejemplo de uso: HOG_LDA_KNN")


# trainPath = 'train_jpg'
# testPath = 'test_alumnos_jpg'
# mserParams = (7, 200, 2000, 1)
# classifierParams = ('HOG', 'LDA', 'LDABAYES'), ('GRAY', 'LDA', 'LDABAYES'), ('HOG', 'LDA', 'KNN'), ('GRAY', 'LDA', 'KNN')
# for cp in classifierParams:
#     source.testValidation(trainPath, mserParams, cp, 0.1, 0.5)

# source.testValidation(trainPath, mserParams, ('GRAY', 'LDA', 'KNN'), 0.1, 0.5)
