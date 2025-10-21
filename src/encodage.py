from manim import *
from manim_slides import *


class Projecteurs(Slide):
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
        bit_jastrow = []
        self.ket = MathTex(r"\ket{x_\sigma}=", tex_template=myTemplate)
        self.jket = MathTex(r"\mathcal{P}_J\ket{x_\sigma}=", tex_template=myTemplate)
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
        for i in range(8):
            square = Square(side_length=side).set_fill(WHITE)
            text_inside = MathTex(rf"{(i+1)%2}").move_to(square.get_center())
            bit = VGroup(square, text_inside)
            bit_jastrow.append(bit)
        bits = VGroup(*bit_array).arrange(buff=0)
        oppbits = VGroup(*bit_opparray).arrange(buff=0)
        resbits = VGroup(*bit_resarray).arrange(buff=0)
        jastrow = VGroup(*bit_jastrow).arrange(buff=0)
        self.bits = bits
        self.oppbits = oppbits
        self.resbits = resbits
        self.jastrow = jastrow
        self.state = VGroup(self.ket, self.bits.copy()).arrange(buff=0.2)
        self.resstate = VGroup(self.gutz, self.resbits).arrange(buff=0.2)
        self.pj = MathTex(r"\ln\mathcal{P}_J=\frac12\sum_{i\neq j}v_{ij}(n_i-1)(n_j-1)", tex_template=myTemplate).to_edge(UP)

    def construct(self):
        self.play(
                Create(self.state)
        )
        self.wait()
        self.next_slide()
        stacked = VGroup(self.bits, self.oppbits).arrange(DOWN, buff=1)
        gutzstate = VGroup(self.gutz, stacked).arrange(buff=0.2)
        and_sign = Text("&")
        newstate = VGroup(gutzstate, and_sign).arrange(buff=0.1)
        self.play(
                TransformMatchingShapes(self.state, newstate)
        )
        self.wait()
        self.next_slide()
        self.play(
                TransformMatchingShapes(newstate, self.resstate),
                Write(self.gvar)
        )
        self.wait()
        self.next_slide()
        self.play(
                FadeOut(self.gvar)
        )
        new_state = VGroup(self.resbits, self.jastrow).arrange(DOWN, buff=1)
        new_state = VGroup(self.jket, new_state).arrange(buff=0.1)
        self.play(
                TransformMatchingShapes(self.resstate, new_state),
                Write(self.pj)
        )
        self.wait()
        self.next_slide()

