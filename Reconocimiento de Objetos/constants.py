SIGN_NAMES = ['NoSeñal', 'Prohibicion', 'Peligro', 'Stop', 'DirProhibida', 'Ceda Paso', 'DirObligatoria']

PROHIBICION = ['00', '01', '02', '03', '04', '05', '07', '08', '09', '10', '15', '16']
PELIGRO = ['11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
STOP = ['14']
DIRECCIONPROHIBIDA = ['17']
CEDAPASO = ['13']
DIRECCIONOBLIGATORIA = ['38']

FEATURE_DESCRIPTORS = ['HOG', 'GRAY']
DIM_REDUCERS = ['LDA']
CLASSIFIERS = ['LDABAYES', 'KNN']

HOG_FEATURE_DESCRIPTORS_PARAMS = [((32, 32), (16, 16), (8, 8), (8, 8), 9, True), ("WINDOW SIZE", "BLOCK SIZE", "BLOCK STRIDE", "CELL SIZE", "N BINS", "SIGNED GRADIENT")]

TRAIN_PATH = ''
TRAIN_PATH_REAL_RESULTS = ''
TEST_PATH = ''
TEST_PATH_REAL_RESULTS = ''
