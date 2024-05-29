# The APPs are imported from the respective files (APP2, APP3) and stored in the APP7 list.
import Approaches.APP4 as APP4
import Approaches.APP3 as APP3


class APP7:

    APP7 = [APP4.APP4.apply_dct, APP3.APP3.apply_pca_perturbation]


def __getattr__():
    return APP7.APP7
