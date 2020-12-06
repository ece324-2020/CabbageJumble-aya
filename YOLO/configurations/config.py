from easydict import EasyDict
''' Configuration Values for data augmentation
The config values for training are in the notebook'''

Cfg = EasyDict()

Cfg.momentum = 0.949

Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1

Cfg.learning_rate = 0.00261
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

Cfg.cutmix = 0
Cfg.mosaic = 1
Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 80
Cfg.track = 0
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # box num


if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

Cfg.TRAIN_TENSORBOARD_DIR = 'log'
