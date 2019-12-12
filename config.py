# Training
BATCH_SIZE = 256
EPOCHS = 8
REG_CONST = 0.001
LEARNING_RATE = 0.1
MOMENTUM = 0.9


HIDDEN_CNN_LAYERS = [{
    'numFilters': 64,
    'kernelSize': (3, 3)
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

# HIDDEN_CNN_LAYERS = [{
#     'numFilters': 64,
#     'kernelSize': (3, 3)
# }, {
#     'numFilters': 64,
#     'kernelSize': (3, 3)
# }, {
#     'numFilters': 64,
#     'kernelSize': (3, 3)
# }, {
#     'numFilters': 64,
#     'kernelSize': (3, 3)
# }]
