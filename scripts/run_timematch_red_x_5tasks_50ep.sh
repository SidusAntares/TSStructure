#!/bin/bash

RUN_NAME=timematch_red_x_5tasks_50ep
OUTPUT_ROOT=outputs/$RUN_NAME
TB_ROOT=runs/$RUN_NAME

SOURCE_MODEL=pseltae_32VNH
SOURCE_TILE=32VNH
SOURCE=denmark/$SOURCE_TILE/2017

# Source-only (red-cross config: no ShiftAug)
python train.py -e $SOURCE_MODEL --source $SOURCE --target $SOURCE --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false --epochs 50

# TimeMatch
TARGET_TILE=30TXT
TARGET=france/$TARGET_TILE/2017
python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false --eval
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false timematch --weights $OUTPUT_ROOT/pseltae_$SOURCE_TILE --epochs 50

TARGET_TILE=31TCJ
TARGET=france/$TARGET_TILE/2017
python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false --eval
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false timematch --weights $OUTPUT_ROOT/pseltae_$SOURCE_TILE --epochs 50

TARGET_TILE=33UVP
TARGET=austria/$TARGET_TILE/2017
python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false --eval
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false timematch --weights $OUTPUT_ROOT/pseltae_$SOURCE_TILE --epochs 50


SOURCE_MODEL=pseltae_30TXT
SOURCE_TILE=30TXT
SOURCE=france/$SOURCE_TILE/2017

# Source-only (red-cross config: no ShiftAug)
python train.py -e $SOURCE_MODEL --source $SOURCE --target $SOURCE --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false --epochs 50

# TimeMatch
TARGET_TILE=32VNH
TARGET=denmark/$TARGET_TILE/2017
python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false --eval
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false timematch --weights $OUTPUT_ROOT/pseltae_$SOURCE_TILE --epochs 50

TARGET_TILE=31TCJ
TARGET=france/$TARGET_TILE/2017
python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false --eval
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --output_dir $OUTPUT_ROOT --tensorboard_log_dir $TB_ROOT --with_shift_aug false timematch --weights $OUTPUT_ROOT/pseltae_$SOURCE_TILE --epochs 50
