# Parameters
BATCH_SIZE = 256
EPOCHS = 30
REG_CONST = 0.001
LEARNING_RATE = 0.001
MOMENTUM = 0.9

HIDDEN_CNN_LAYERS = [{
    'numFilters': 64,
    'kernelSize': (7, 7)
}, {
    'numFilters': 64,
    'kernelSize': (5, 5)
}, {
    'numFilters': 64,
    'kernelSize': (5, 5)
}, {
    'numFilters': 64,
    'kernelSize': (3, 3)
}, {
    'numFilters': 64,
    'kernelSize': (3, 3)
}, {
    'numFilters': 64,
    'kernelSize': (3, 3)
}]
