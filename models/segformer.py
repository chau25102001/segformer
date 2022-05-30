from models.modules import *
from utils.generals import init_weights


class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone)
        self.num_classes = num_classes
        self.decode_head = UnetDecoder(
            embed_dims=self.backbone.embed_dims,
            img_size_reductions=self.backbone.img_size_reductions,
            n_classes=num_classes
        )
        self.apply(init_weights)

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y[::-1])
        return y


if __name__ == '__main__':
    model = SegFormer('MiT-B2', 1)
    inp = torch.zeros(1, 3, 512, 512)
    oup = model(inp)
    print(oup.shape)
