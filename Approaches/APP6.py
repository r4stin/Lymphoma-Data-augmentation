# The APPs are imported from the respective files (APP2, APP3) and stored in the APP6 list.
import Approaches.APP2 as APP2
import Approaches.APP3 as APP3


class APP6:
    APP6 = [APP3.APP3.apply_pca_perturbation, APP2.APP2.gaussian_noisy_1, APP2.APP2.gaussian_noisy_2]


def __getattr__():
    return APP6.APP6
