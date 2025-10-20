from manim import *
from manim_slides import *

import scene1
from energy_minimisation import VarMethod
from ansatz import PfaffianIntro, PfaffianAnsatz, Gutzwiller

class Presentation(ThreeDSlide):
    def construct(self):
        all_slides = [
                VarMethod,
                PfaffianAnsatz,
                PfaffianIntro,
                Gutzwiller,
        ]

        for slides in all_slides:
            slides.setup(self)
            slides.construct(self)
            self.next_slide()
            self.play(*[FadeOut(mob) for mob in self.mobjects])
