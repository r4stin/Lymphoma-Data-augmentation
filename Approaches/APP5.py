# The APPs are imported from the respective files (APP1, APP3) and stored in the APP5 list.
import Approaches.APP1 as APP1
import Approaches.APP3 as APP3


class APP5:

    APP5 = [
        APP1.APP1.random_horizontal_flip,
        APP1.APP1.random_vertical_flip,
        APP1.APP1.random_affine,
        APP3.APP3.apply_pca_perturbation
    ]


def __getattr__():
    return APP5.APP5
