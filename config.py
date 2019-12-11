# Training
BATCH_SIZE = 256
EPOCHS = 3
REG_CONST = 0.001
LEARNING_RATE = 0.1
MOMENTUM = 0.9


HIDDEN_CNN_LAYERS = [{
     'numFilters': 75,
     'kernelSize': (3, 3)}]


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
}, {
    'numFilters': 64,
    'kernelSize': (3, 3)
}, {
    'numFilters': 64,
    'kernelSize': (3, 3)
}]

# HIDDEN_CNN_LAYERS = [{
#     'filters': 75,
#     'kernel_size': (3, 3)
# }, {
#     'filters': 75,
#     'kernel_size': (3, 3)
# }, {
#     'filters': 75,
#     'kernel_size': (3, 3)
# }, {
#     'filters': 75,
#     'kernel_size': (3, 3)
# }, {
#     'filters': 75,
#     'kernel_size': (3, 3)
# }, {
#     'filters': 75,
#     'kernel_size': (3, 3)
# }]
