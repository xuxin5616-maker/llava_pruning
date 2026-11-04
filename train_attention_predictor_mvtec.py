"""MVTec-AD ALL images, no validation; otherwise the existing student recipe."""

from pathlib import Path
from train_attention_predictor import main as train


def main(argv=None):
    return train(argv,defaults=dict(dataset="mvtec",data_root="/home/yz/xxy/data/datasets/MVTec-AD",
        validation_fraction=0.,output_dir=Path("output/attention_predictor_mvtec_train_01")))


if __name__=="__main__": main()
