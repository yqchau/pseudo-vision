import timm


def TIMM(
    arch: str = "mobilenetv3_small_050",
    transfer_learning: bool = False,
    classes: int = 1000,
    drop_rate: float = 0.1,
    final_pooling: str = "avg",
    in_chans: int = 3,
):

    model = timm.create_model(
        arch, pretrained=transfer_learning, drop_rate=drop_rate, in_chans=in_chans
    )

    if model.num_classes != classes:
        model.reset_classifier(num_classes=classes, global_pool=final_pooling)

    return model
