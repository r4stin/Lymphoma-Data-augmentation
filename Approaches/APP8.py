# The APPs are imported from the respective files (APP1, APP3, APP4) and stored in the APP8 list.
import Approaches.APP1 as APP1
import Approaches.APP3 as APP3
import Approaches.APP4 as APP4


class APP8:

    APP8 = [APP1.APP1.random_horizontal_flip,
            APP1.APP1.random_vertical_flip,
            APP1.APP1.random_affine,
            APP4.APP4.apply_dct,
            APP3.APP3.apply_pca_perturbation
            ]


def __getattr__():
    return APP8.APP8
