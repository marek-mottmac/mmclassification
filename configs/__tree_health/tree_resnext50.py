_base_ = [
    '../_base_/models/resnext50_32x4d.py',
    # '../_base_/datasets/imagenet_bs32.py', #OBSOLETE DATA PIPELINE
    '../__tree_health/tree_data_loader.py',

    '../_base_/schedules/imagenet_bs256.py', 
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=6,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 3),
    ))
