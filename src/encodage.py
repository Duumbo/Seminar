from manim import *
from manim_slides import *


class Projecteurs(Scene):
    def setup(self):
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{physics}")
        self.gutz = MathTex(r"\mathcal{P}_G \ket{x_\sigma}=", tex_template=myTemplate)
        self.gvar = MathTex(
                r"""
                \vec{g}=
                \begin{pmatrix}
                g_0 & g_1 & g_2 & g_3 & g_4 & g_5 & g_6 & g_7
                \end{pmatrix}
        """, tex_template=myTemplate).to_edge(DOWN)
        side = 1
        bit_array = []
        bit_opparray = []
        bit_resarray = []
        self.ket = MathTex(r"\ket{x_\sigma}=", tex_template=myTemplate)
        for i in range(8):
            square = Square(side_length=side).set_fill(WHITE)
            text_inside = MathTex(rf"n_{{{i}\sigma}}").move_to(square.get_center())
            bit = VGroup(square, text_inside)
            bit_array.append(bit)
        for i in range(8):
            square = Square(side_length=side).set_fill(WHITE)
            text_inside = MathTex(rf"n_{{{i}\bar{{\sigma}}}}").move_to(square.get_center())
            bit = VGroup(square, text_inside)
            bit_opparray.append(bit)
        for i in range(8):
            square = Square(side_length=side).set_fill(WHITE)
            text_inside = MathTex(rf"{i%2}").move_to(square.get_center())
            bit = VGroup(square, text_inside)
            bit_resarray.append(bit)
        bits = VGroup(*bit_array).arrange(buff=0)
        oppbits = VGroup(*bit_opparray).arrange(buff=0)
        resbits = VGroup(*bit_resarray).arrange(buff=0)
        self.bits = bits
        self.oppbits = oppbits
        self.resbits = resbits
        self.state = VGroup(self.ket, self.bits.copy()).arrange(buff=0.2)
        self.resstate = VGroup(self.gutz, self.resbits).arrange(buff=0.2)

    def construct(self):
        self.play(
                Create(self.state)
        )
        self.wait()
        stacked = VGroup(self.bits, self.oppbits).arrange(DOWN, buff=1)
        gutzstate = VGroup(self.gutz, stacked).arrange(buff=0.2)
        and_sign = Text("&")
        newstate = VGroup(gutzstate, and_sign).arrange(buff=0.1)
        self.play(
                TransformMatchingShapes(self.state, newstate)
        )
        self.wait()
        self.play(
                TransformMatchingShapes(newstate, self.resstate)
        )
        self.wait()
        self.play(
                Write(self.gvar)
        )
        self.wait()
