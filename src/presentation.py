from manim import *
from manim_slides import *

import scene1
from energy_minimisation import VarMethod, EnergyMinimisation, ConjGradient
from ansatz import PfaffianIntro, PfaffianAnsatz, Gutzwiller, Jastrow
from encodage import Projecteurs

class Presentation(ThreeDSlide):
    skip_reversing = True
    def construct(self):
        all_slides = [
                VarMethod,
                EnergyMinimisation,
                ConjGradient,
                PfaffianAnsatz,
                PfaffianIntro,
                Gutzwiller,
                Jastrow,
                Projecteurs,
        ]

        for slides in all_slides:
            slides.setup(self)
            slides.construct(self)
            self.next_slide()
            self.play(*[FadeOut(mob) for mob in self.mobjects])
