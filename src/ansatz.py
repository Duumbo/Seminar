from manim import *
from manim_slides import *


class PfaffianIntro(Slide):
    def setup(self):
        self.xmat = Matrix(
            [[ 0         , r"a_{12}" , r"a_{13}" , r"a_{14}"],
             [ r"-a_{12}", 0         , r"a_{23}" , r"a_{24}"],
             [ r"-a_{13}", r"-a_{23}", 0         , r"a_{34}"],
             [ r"-a_{14}", r"-a_{24}", r"-a_{34}", 0        ]],
            left_bracket="(",
            right_bracket=")"
        )

        coeff_12 = MathTex(r"{{ a_{12} }} Pf")
        self.submat12 = Matrix(
            [
             [ 0         , r"a_{34}"],
             [ r"-a_{34}", 0        ]],
            left_bracket="(",
            right_bracket=")"
        )
        coeff_13 = MathTex(r"- {{ a_{13} }} Pf")
        self.submat13 = Matrix(
            [
             [ 0         , r"a_{24}"],
             [ r"-a_{24}", 0        ]],
            left_bracket="(",
            right_bracket=")"
        )
        coeff_14 = MathTex(r"+ {{ a_{14} }} Pf")
        self.submat14 = Matrix(
            [
             [ 0         , r"a_{23}"],
             [ r"-a_{23}", 0        ],
             ],
            left_bracket="(",
            right_bracket=")"
        )

        self.pfaff = Group(coeff_12, self.submat12, coeff_13, self.submat13, coeff_14, self.submat14).arrange(RIGHT, buff=0.5)

    def construct(self):
        self.play(
                Write(self.xmat)
        )
        self.next_slide()
        rows = self.xmat.get_rows()
        cols = self.xmat.get_columns()
        elements = self.xmat.get_entries()
        rows_ani = []
        cols_ani = []
        for i in range(4):
            rows_ani.append(Circumscribe(rows[i], fade_out=True))
            cols_ani.append(Circumscribe(cols[i], fade_out=True))

        self.play(
            elements[1].animate.set_color(YELLOW),
            )
        self.next_slide()
        self.play(
            rows_ani[0],
            rows_ani[1],
            cols_ani[0],
            cols_ani[1],
        )
        self.next_slide()

        coeff_copy1 = elements[1].copy()
        matrix_copy = Group(*[
                 elements[2 + 4*2].copy(), elements[3 + 4*2].copy(),
                 elements[2 + 4*3].copy(), elements[3 + 4*3].copy(),
                ])
        self.add(coeff_copy1, matrix_copy)
        self.play(
                coeff_copy1.animate.to_corner(UL)
        )
        self.play(
                matrix_copy.animate.next_to(coeff_copy1, RIGHT)
        )
        self.submat12.next_to(coeff_copy1, RIGHT)
        self.play(
                TransformMatchingShapes(matrix_copy, self.submat12)
        )
        self.next_slide()

        self.play(
            elements[2].animate.set_color(YELLOW),
            )
        self.wait()
        self.play(
            rows_ani[0],
            rows_ani[2],
            cols_ani[0],
            cols_ani[2],
        )
        self.wait()

        coeff_copy2 = elements[2].copy()
        matrix_copy = Group(*[
                 elements[1 + 4*1].copy(), elements[1 + 4*3].copy(),
                 elements[3 + 4*1].copy(), elements[3 + 4*3].copy(),
                ])
        self.add(coeff_copy2, matrix_copy)
        minus = MathTex(r"-").next_to(self.submat12, RIGHT)
        self.play(
                Create(minus),
                coeff_copy2.animate.next_to(minus, RIGHT)
        )
        self.play(
                matrix_copy.animate.next_to(coeff_copy2, RIGHT)
        )
        self.submat13.next_to(coeff_copy2, RIGHT)
        self.play(
                TransformMatchingShapes(matrix_copy, self.submat13)
        )
        self.wait()

        self.play(
            elements[3].animate.set_color(YELLOW),
            )
        self.wait()
        self.play(
            rows_ani[0],
            rows_ani[3],
            cols_ani[0],
            cols_ani[3],
        )
        self.wait()

        coeff_copy3 = elements[3].copy()
        matrix_copy = Group(*[
                 elements[1 + 4*1].copy(), elements[1 + 4*2].copy(),
                 elements[2 + 4*1].copy(), elements[2 + 4*2].copy(),
                ])
        self.add(coeff_copy3, matrix_copy)
        plus = MathTex(r"+").next_to(self.submat13, RIGHT)
        self.play(
                Create(plus),
                coeff_copy3.animate.next_to(plus, RIGHT)
        )
        self.play(
                matrix_copy.animate.next_to(coeff_copy3, RIGHT)
        )
        self.submat14.next_to(coeff_copy3, RIGHT)
        self.play(
                TransformMatchingShapes(matrix_copy, self.submat14)
        )
        self.next_slide()
        pfaff_final = MathTex(r"{\rm Pf}(A)=a_{12}a_{34}-a_{13}a_{24}+a_{14}a_{23}")
        self.play(
                FadeOut(self.xmat)
                )
        self.wait()
        self.play(
                TransformMatchingShapes(
                    Group(
                        coeff_copy1,
                        self.submat12,
                        minus,
                        coeff_copy2,
                        self.submat13,
                        plus,
                        coeff_copy3,
                        self.submat14,
                    ),
                    pfaff_final
                )
        )
        self.next_slide()


class PfaffianAnsatz(Slide):
    def setup(self):
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{physics}")
        self.ansatz = MathTex(r"\ket{\psi}=\qty[\sum_{i,j=1}^N\sum_{\sigma_i,\sigma_j}F_{i\sigma_i,j\sigma_j}c^\dagger_{i\sigma_i}c_{j\sigma_j}]^{N_e/2}\ket{0}", tex_template=myTemplate)
        self.slater = MathTex(
                r"""
                \psi(x_1,x_2,x_3,\cdots,x_n)
                =
                \frac1{\sqrt{n!}}\det\begin{pmatrix}
                \psi_1(x_1) & \psi_1(x_2) & \psi_1(x_3) & \cdots & \psi_1(x_n)\\
                \psi_2(x_1) & \psi_2(x_2) & \psi_2(x_3) & \cdots & \psi_2(x_n)\\
                \psi_3(x_1) & \psi_3(x_2) & \psi_3(x_3) & \cdots & \psi_3(x_n)\\
                \vdots & \vdots & \vdots & \ddots & \vdots\\
                \psi_n(x_1) & \psi_n(x_2) & \psi_n(x_3) & \cdots & \psi_n(x_n)
                \end{pmatrix}
        """, font_size=35)
        self.slater_comment = Text("Pour un fonction d'onde séparable:").to_edge(UP)
        self.motivation = MathTex(
                r"""
                \psi(x_1,x_2,x_3,x_4)&=\frac{1}{2}\det
                \begin{pmatrix}
                \psi_1(x_1,x_2) & \psi_1(x_3,x_4)\\
                \psi_2(x_1,x_2) & \psi_2(x_3,x_4)
                \end{pmatrix}\\
                &=
                \frac{1}{2}{\rm Pf}
                \begin{pmatrix}
                0 & 0 & \psi_1(x_1,x_2) & \psi_1(x_3,x_4)\\
                0 & 0 & \psi_2(x_1,x_2) & \psi_2(x_3,x_4)\\
                -\psi_1(x_1,x_2) & -\psi_2(x_1,x_2) & 0 & 0\\
                -\psi_1(x_3,x_4) & -\psi_2(x_3,x_4) & 0 & 0
                \end{pmatrix}
        """, font_size=35)

    def construct(self):
        self.play(
                Write(self.slater_comment)
        )
        self.play(
                Write(self.slater)
        )
        self.next_slide()
        self.wait()
        self.play(
                FadeOut(self.slater_comment),
                FadeOut(self.slater)
        )
        self.wait()
        self.next_slide()
        self.play(
                Write(self.motivation)
        )
        self.wait()
        self.next_slide()
        self.play(
                FadeOut(self.motivation)
        )
        self.wait()
        self.next_slide()
        self.play(
                Write(self.ansatz)
        )
        self.wait()
        self.next_slide()


class Gutzwiller(ThreeDSlide):
    def setup(self):
        self.proj = MathTex(r"\mathcal{P}_G=e^{\sum_i g_in_{i\uparrow}n_{i\downarrow}}").set_opacity(0)
        self.add_fixed_in_frame_mobjects(self.proj)
        self.set_camera_orientation(phi=PI/3)
        xran, yran = (3, 3)
        self.axes = ThreeDAxes(x_range=[-xran, xran], y_range=[-yran,yran]).set_opacity(0)
        self.add(self.axes)
        dots = []
        dashedlines = []
        spin_ups = []
        spin_downs = []
        for i in range(4):
            for j in range(4):
                dots.append(Dot3D(self.axes.c2p(*(i*RIGHT + j*UP + 2*LEFT + 2*DOWN))))
                origin = self.axes.c2p(0,0,0)
                up = self.axes.c2p(0,0,1)
                down = self.axes.c2p(0,0,-1)
                eps = 0.1
                spin_ups.append(
                        Arrow3D(start=(origin), end=up).move_to(
                            self.axes.c2p(*(i*RIGHT + j*UP + 2*LEFT + 2*DOWN + eps * RIGHT))
                        ).set_color(BLUE)
                )
                spin_downs.append(
                        Arrow3D(start=(origin), end=down).move_to(
                            self.axes.c2p(*(i*RIGHT + j*UP + 2*LEFT + 2*DOWN + eps * LEFT))
                        ).set_color(RED)
                )
            startx = self.axes.c2p(*(i*RIGHT + 2*LEFT - 4*UP))
            endx   = self.axes.c2p(*(i*RIGHT + 2*LEFT + 4*UP))
            starty = self.axes.c2p(*(i*UP + 2*DOWN - 4*RIGHT))
            endy   = self.axes.c2p(*(i*UP + 2*DOWN + 4*RIGHT))
            dashedlines.append(DashedLine(start=(startx), end=(endx)))
            dashedlines.append(DashedLine(start=(starty), end=(endy)))


        self.dots = dots
        self.lattice = VGroup(*dots)
        self.dashedlines = dashedlines
        self.checker = VGroup(*dashedlines)
        self.spins = (spin_ups, spin_downs)
        self.spingroup = VGroup(*spin_ups, *spin_downs)

        # Bar Charts
        bcharts = []
        bvalues = []
        for i in range(4):
            bvalues.append([1,1,1,1])
            bcharts.append(BarChart(values=[1, 1, 1, 1], bar_names=[
                rf"$g_{{{i}0}}$",
                rf"$g_{{{i}1}}$",
                rf"$g_{{{i}2}}$",
                rf"$g_{{{i}3}}$",
                ]
            ))
        self.subscene = VGroup(self.checker, self.lattice, self.spingroup, self.axes)
        self.barcharts = VGroup(*bcharts).arrange(UP).to_edge(RIGHT)
        self.barcharts.set_opacity(0).scale(0.4)
        self.add_fixed_in_frame_mobjects(self.barcharts)

        def update_opacity(g_vec):
            animations = []
            for i in range(4):
                animations.append(bcharts[i].animate.change_bar_values(g_vec[4*i:4*i+4]))
                for j in range(4):
                    animations.append(self.spins[0][i*4+j].animate.set_opacity(g_vec[i*4+j]))
                    animations.append(self.spins[1][i*4+j].animate.set_opacity(g_vec[i*4+j]))
            return animations

        self.update_opacity = update_opacity


    def construct(self):
        self.begin_ambient_camera_rotation(rate=0.1, about='theta')
        self.proj.set_opacity(1)
        self.play(
                Write(self.proj)
        )
        self.next_slide()
        self.wait()
        self.play(
                self.proj.animate.to_edge(UP)
        )
        self.wait()
        self.next_slide()
        #self.play(
        #        LaggedStart(*(Create(dot) for dot in self.dots)),
        #        LaggedStart(*(Create(line) for line in self.dashedlines)),
        #        LaggedStart(*(Create(sup) for sup in self.spins[0])),
        #        LaggedStart(*(Create(sdo) for sdo in self.spins[1])),
        #)
        self.play(
                Create(self.subscene)
        )
        self.wait()
        self.next_slide()
        self.play(
                self.subscene.animate.scale(0.8).to_edge(LEFT),
        )
        self.wait()
        self.next_slide()
        self.barcharts.set_opacity(1)
        self.play(
                Create(self.barcharts)
        )
        self.next_slide()
        self.wait()

        new_g = [
                0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0,
        ]
        animations = self.update_opacity(new_g)
        self.play(*animations)
        self.wait()
        self.next_slide()
        self.play(
                FadeOut(self.proj),
                FadeOut(self.subscene),
                FadeOut(self.barcharts)
        )


class Jastrow(ThreeDScene):
    def setup(self):
        self.proj = MathTex(r"\mathcal{P}_J=e^{\sum_{i\neq j} v_{ij}(n_{i\uparrow} + n_{i\downarrow} - 1)(n_{j\uparrow} + n_{j\downarrow} - 1)").set_opacity(0)
        self.add_fixed_in_frame_mobjects(self.proj)
        self.set_camera_orientation(phi=PI/3)
        xran, yran = (3, 3)
        self.axes = ThreeDAxes(x_range=[-xran, xran], y_range=[-yran,yran]).set_opacity(0)
        self.add(self.axes)
        dots = []
        dashedlines = []
        spin_ups = []
        spin_downs = []
        for i in range(4):
            for j in range(4):
                dots.append(Dot3D(self.axes.c2p(*(i*RIGHT + j*UP + 2*LEFT + 2*DOWN))))
                origin = self.axes.c2p(0,0,0)
                up = self.axes.c2p(0,0,1)
                down = self.axes.c2p(0,0,-1)
                eps = 0.1
                spin_ups.append(
                        Arrow3D(start=(origin), end=up).move_to(
                            self.axes.c2p(*(i*RIGHT + j*UP + 2*LEFT + 2*DOWN + eps * RIGHT))
                        ).set_color(BLUE)
                )
                spin_downs.append(
                        Arrow3D(start=(origin), end=down).move_to(
                            self.axes.c2p(*(i*RIGHT + j*UP + 2*LEFT + 2*DOWN + eps * LEFT))
                        ).set_color(RED)
                )
            startx = self.axes.c2p(*(i*RIGHT + 2*LEFT - 4*UP))
            endx   = self.axes.c2p(*(i*RIGHT + 2*LEFT + 4*UP))
            starty = self.axes.c2p(*(i*UP + 2*DOWN - 4*RIGHT))
            endy   = self.axes.c2p(*(i*UP + 2*DOWN + 4*RIGHT))
            dashedlines.append(DashedLine(start=(startx), end=(endx)))
            dashedlines.append(DashedLine(start=(starty), end=(endy)))


        self.dots = dots
        self.lattice = VGroup(*dots)
        self.dashedlines = dashedlines
        self.checker = VGroup(*dashedlines)
        self.spins = (spin_ups, spin_downs)
        self.spingroup = VGroup(*spin_ups, *spin_downs)

        # Bar Charts
        bcharts = []
        bvalues = []
        for i in range(4):
            bvalues.append([1,1,1,1])
            bcharts.append(BarChart(values=[1, 1, 1, 1], bar_names=[
                rf"$g_{{{i}0}}$",
                rf"$g_{{{i}1}}$",
                rf"$g_{{{i}2}}$",
                rf"$g_{{{i}3}}$",
                ]
            ))
        self.subscene = VGroup(self.checker, self.lattice, self.spingroup, self.axes)
        self.barcharts = VGroup(*bcharts).arrange(UP).to_edge(RIGHT)
        self.barcharts.set_opacity(0).scale(0.4)
        self.add_fixed_in_frame_mobjects(self.barcharts)

        def update_opacity(g_vec):
            animations = []
            for i in range(4):
                animations.append(bcharts[i].animate.change_bar_values(g_vec[4*i:4*i+4]))
                for j in range(4):
                    animations.append(self.spins[0][i*4+j].animate.set_opacity(g_vec[i*4+j]))
                    animations.append(self.spins[1][i*4+j].animate.set_opacity(g_vec[i*4+j]))
            return animations

        focal1 = 2*RIGHT + 1*UP + 2*LEFT + 2*DOWN
        focal2 = 3*RIGHT + 1*UP + 2*LEFT + 2*DOWN
        focalvec = focal1 - focal2
        center = focal2 + 0.5 * focalvec
        c = np.linalg.norm(self.axes.c2p(*(0.5*focalvec)))
        a = 2
        b = a**2 - c**2

        ellips = DashedVMobject(Ellipse(width=a, height=b).move_to(self.axes.c2p(*center)).set_color(YELLOW))
        self.selection = ellips

        self.update_opacity = update_opacity


    def construct(self):
        self.begin_ambient_camera_rotation(rate=0.1, about='theta')
        self.proj.set_opacity(1)
        self.play(
                Write(self.proj)
        )
        self.wait()
        self.play(
                self.proj.animate.to_edge(UP)
        )
        self.wait()
        self.play(
                Create(self.subscene)
        )
        self.wait()
        self.play(
                Create(self.selection)
        )
        #self.play(
        #        self.subscene.animate.scale(0.5).to_edge(LEFT),
        #)
        #self.wait()
        #self.barcharts.set_opacity(1)
        #self.play(
        #        Create(self.barcharts)
        #)
        #self.wait()

        #new_g = [
        #        0.0, 0.0, 0.0, 0.0,
        #        1.0, 0.0, 0.0, 1.0,
        #        1.0, 0.0, 0.0, 1.0,
        #        0.0, 0.0, 0.0, 0.0,
        #]
        #animations = self.update_opacity(new_g)
        #self.play(*animations)
        #self.wait()
        #self.play(
        #        FadeOut(self.proj),
        #        FadeOut(self.subscene),
        #        FadeOut(self.barcharts)
        #)
