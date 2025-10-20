from manim import *
from manim_slides import *
from scipy.linalg import sqrtm

NPOINTS = 10

class VarMethod(Slide):
    def setup(self):
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{physics}")
        self.energy = MathTex(r"""
            {{ E[\psi] }} {{ = }} \frac{\expval{\psi(\alpha_i)|H|\psi(\alpha_i)}}
                              {\braket{\psi(\alpha_i)}{\psi(\alpha_i)}}
        """, tex_template=myTemplate)
        self.principe_var = MathTex(r"""
            {{ E[\psi] }} {{ = }} \sum_{i=0}^N E_i \big|f(\{\alpha_j\})\big|^2 \geq \Omega
        """, tex_template=myTemplate)
        self.norm = MathTex(r"""
            \ket{ \bar{\psi} (\alpha_i) } = \frac{\ket{\psi(\alpha_i)}}
                    {\sqrt{\braket{\psi}{\psi}}}
        """, tex_template=myTemplate)
        self.schro = MathTex(r"""
            {{ i\hbar\dv{t} }} {{ \ket{\psi} }} {{ = }} {{ H }} {{ \ket{\psi} }}
        """, tex_template=myTemplate)
        self.schro_norm = MathTex(r"""
            {{ i\hbar\dv{t} }} {{ \ket{\bar{\psi}} }} {{ = }} {{ ( }} {{ H }}
                        {{ - }} {{ \expval{H} }} {{ ) }} {{ \ket{\bar{\psi}} }}
        """, tex_template=myTemplate)

    def construct(self):
        self.play(
                Write(self.energy)
        )
        self.next_slide()
        self.play(
                TransformMatchingTex(self.energy, self.principe_var)
        )
        self.next_slide()
        self.norm.next_to(self.energy, DOWN)
        self.play(
                Write(self.norm)
        )
        self.next_slide()
        self.schro.next_to(self.norm, DOWN)
        self.schro_norm.next_to(self.norm, DOWN)
        self.play(
                Write(self.schro)
        )
        self.next_slide()
        self.play(
                TransformMatchingTex(self.schro, self.schro_norm)
        )
        self.next_slide()

class EnergyMinimisation(Scene):
    def setup(self):
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{physics}")
        self._base_eq = MathTex(
                r"i\hbar\pdv{}{t}",r"\ket{\psi(t)}=", r"H\ket{\psi(t)}",
                tex_template=myTemplate
            )
        self._time_ev = MathTex(
                r"\ket{\psi(t)}=", r"e^{-iHt/\hbar}", r"\ket{\psi(0)}",
                tex_template=myTemplate
            )
        self._gs_sum = MathTex(
                r"\ket{\psi(t)}=",
                r"e^{-iHt/\hbar}",
                r"\big(", r"c_1", r"\ket{\phi_1}", r"+c_2", r"\ket{\phi_2}", r"+c_3", r"\ket{\phi_3}", r"+\cdots", r"+c_N", r"\ket{\phi_N}", r"\big)",
                tex_template=myTemplate,
            )
        self._gs_sum2 = MathTex(
                r"\ket{\psi(t)}=",
                r"\big(", r"c_1", r"e^{-i {{ E_1 }} t/\hbar}", r"\ket{\phi_1}", r"+c_2", r"e^{-iE_2t/\hbar}", r"\ket{\phi_2}", r"+c_3", r"e^{-iE_3t/\hbar}", r"\ket{\phi_3}", r"+\cdots", r"+c_N", r"e^{-iE_Nt/\hbar}", r"\ket{\phi_N}", r"\big)",
                tex_template=myTemplate,
                font_size= 35
            )
        self._phase_sym = MathTex(
                r"\ket{\psi(t)}=",
                r"e^{-i {{ E_1 }} t/\hbar}", r"\big(", r"c_1", r"\ket{\phi_1}", r"+c_2", r"e^{-i(E_2- {{ E_1 }} )t/\hbar}", r"\ket{\phi_2}", r"+c_3", r"e^{-i(E_3- {{ E_1 }} )t/\hbar}", r"\ket{\phi_3}", r"+\cdots", r"+c_N", r"e^{-i(E_N- {{ E_1 }} )t/\hbar}", r"\ket{\phi_N}", r"\big)",
                tex_template=myTemplate,
                font_size= 30
            )
        self._phase_sym2 = MathTex(
                r"\ket{\psi(t)}=",
                r"c_1", r"\ket{\phi_1}", r"+c_2", r"e^{-i(E_2- {{ E_1 }} )t/\hbar}", r"\ket{\phi_2}", r"+c_3", r"e^{-i(E_3- {{ E_1 }} )t/\hbar}", r"\ket{\phi_3}", r"+\cdots", r"+c_N", r"e^{-i(E_N- {{ E_1 }} )t/\hbar}", r"\ket{\phi_N}",
                tex_template=myTemplate,
                font_size = 30
            )
        self._phase_sym3 = MathTex(
                r"\ket{\psi(t)}=",
                r"c_1", r"\ket{\phi_1}", r"+c_2", r"e^{-(E_2- {{ E_1 }} ) {{ it/\hbar }} }", r"\ket{\phi_2}", r"+c_3", r"e^{-(E_3- {{ E_1 }} ) {{ it/\hbar }} }", r"\ket{\phi_3}", r"+\cdots", r"+c_N", r"e^{-(E_N- {{ E_1 }} ) {{ it/\hbar }} }", r"\ket{\phi_N}",
                tex_template=myTemplate,
                font_size = 30
            )
        self._t_trans = MathTex(r"{{ \tau }} = {{ it/\hbar }}", tex_template=myTemplate).move_to(DOWN)
        self._imag_time = MathTex(
                r"\ket{\psi( {{ \tau }} )}=",
                r"c_1\ket{\phi_1}+c_2e^{-(E_2-E_1)\tau}\ket{\phi_2}+c_3e^{-(E_3-E_1)\tau}\ket{\phi_3}+\cdots+c_Ne^{-(E_N-E_1)\tau}\ket{\phi_N}",
                tex_template=myTemplate,
                font_size = 30
            )
        self._gs = MathTex(
                r"\lim_{\tau\to\infty}\ket{\psi(\tau)}=", r"C\ket{\phi_1}\propto\ket{\Omega}",
                tex_template=myTemplate
            )
        self._phase_sym.set_color_by_tex(r"E_1", RED)
        self._phase_sym2.set_color_by_tex(r"E_1", RED)
        self._phase_sym3.set_color_by_tex(r"E_1", RED)

    def construct(self):
        self.play(Write(self._base_eq))
        self.wait()
        self.play(TransformMatchingTex(self._base_eq, self._time_ev))
        self.wait()
        self.play(TransformMatchingTex(self._time_ev, self._gs_sum))
        self.wait()
        self.play(TransformMatchingTex(self._gs_sum, self._gs_sum2))
        self.wait()
        self.play(
                self._gs_sum2[4].animate.set_color(RED)
                )
        self.wait()
        self.play(TransformMatchingTex(self._gs_sum2, self._phase_sym))
        self.wait()
        self.play(
                self._phase_sym[1:4].animate.set_color(config.background_color),
                self._phase_sym[4].animate.set_color(config.background_color),
                self._phase_sym[-1].animate.set_color(config.background_color)
        )
        self.wait()
        self.play(TransformMatchingTex(self._phase_sym, self._phase_sym2))
        self.wait()
        self.play(
                Write(self._t_trans),
                TransformMatchingTex(self._phase_sym2, self._phase_sym3)
            )
        self.wait()


        self.play(
                TransformMatchingTex(self._phase_sym3, self._imag_time)
            )
        self.wait()

        width = 1.0
        y = -0.5
        e1 = (-2.0, y, 0.0)
        e2 = (0.8, y, 0.0)
        e3 = (4.3, y, 0.0)
        bbox = [
                Line(0.0, width).move_to(e1),
                Line(0.0, width).move_to(e2),
                Line(0.0, width).move_to(e3),
            ]
        braces =  [
                Brace(bbox[0], UP, fill_color=YELLOW),
                Brace(bbox[1], UP, fill_color=YELLOW),
                Brace(bbox[2], UP, fill_color=YELLOW)
            ]
        t1 = (-2.0, y + 1.3, 0.0)
        t2 = (0.8,  y + 1.3, 0.0)
        t3 = (4.3,  y + 1.3, 0.0)
        texts = [
                MathTex(r">0", color=YELLOW).move_to(t1),
                MathTex(r">0", color=YELLOW).move_to(t2),
                MathTex(r">0", color=YELLOW).move_to(t3),
            ]
        self.play(FadeIn(*braces, *texts))
        self.wait()
        self.play(FadeOut(*braces, *texts))
        self.wait()

        d1 = Dot(2*UP + 2*LEFT, color=YELLOW)
        d3 = Dot(2*UP + 2*LEFT, color=YELLOW)
        t1 = MathTex(r"0", color=YELLOW, font_size=30).move_to(1.8*UP+2*LEFT)
        t2 = MathTex(r"\infty", color=YELLOW, font_size=30).move_to(1.8*UP+3*RIGHT)
        start = 2*UP + 2*LEFT
        end = 2*UP + 3*RIGHT
        d2 = Dot(start, color=YELLOW)
        slider_bounds = VGroup(d1, d2)
        slider = Line(d1.get_center(), d2.get_center(), color=YELLOW)
        x = ValueTracker(0)
        d2.add_updater(lambda z: z.set_x((1.0 - x.get_value()) * start[0] + x.get_value() * end[0]))
        slider.add_updater(lambda z: z.become(Line(d1.get_center(), d2.get_center())))
        self.add(d1, d2, slider, t1, t2)
        self.play(
                self._imag_time.animate.to_edge(UP),
                self._t_trans.animate.move_to(UP + LEFT + UP + LEFT + LEFT + LEFT),
                x.animate.set_value(1.0)
            )
        self.wait()

        val = [1.0, 1.0, 1.0, 0.0, 1.0]
        names = [
                MathTex(r"\Big|\frac{c_1}{c_1}\Big|^2", font_size=30),
                MathTex(r"\Big|\frac{c_2e^{-(E_2-E_1)\tau}}{c_1}\Big|^2", font_size=30),
                MathTex(r"\Big|\frac{c_3e^{-(E_3-E_1)\tau}}{c_1}\Big|^2", font_size=30),
                MathTex(r"\cdots"),
                MathTex(r"\Big|\frac{c_Ne^{-(E_N-E_1)\tau}}{c_1}\Big|^2", font_size=30),
            ]
        barchart = BarChart(val, x_length=12*RIGHT[0], y_length=3*UP[1]).move_to(2*DOWN)
        t = ValueTracker(0)
        d3.add_updater(lambda z: z.set_x(np.exp(- t.get_value()) * start[0] + (1.0 - np.exp( - t.get_value())) * end[0]))
        ts = np.logspace(0.1, 1000, NPOINTS)
        for eq, bar in zip(names, barchart.bars):
            eq.next_to(bar, UP)


        self.add(*names, d3)
        self.play(Create(barchart))

        c = [1.0, 1.0, 1.0, 1.0]
        e = [-0.9, -0.1, 1.0, 1.0]
        for tt in ts:
            w = [1.0, 1.0, 1.0, 0.0, 1.0]
            for i in range(4):
                w[i+1] = np.abs(c[i] * np.exp(- (e[i] + 1.0) * tt)) ** 2
            self.play(
                    barchart.animate.change_bar_values(w),
                    *[name.animate.next_to(bar, UP) for (name, bar) in zip(names, barchart.bars)],
                    t.animate.set_value(tt),
                    rate_func=linear,
                    run_time=0.2,
                    )

        self.play(
                FadeOut(barchart, *names, self._t_trans, slider, d1, d2, d3, t1, t2)
                )
        self.wait()
        self.play(Transform(self._imag_time, self._gs))
        self.wait()

class ConjGradient(ThreeDScene):
    def setup(self):
        point_init = [-0.5, 1.0, 0.0]
        # Setup axis
        self.axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2])

        x_label = self.axes.get_x_axis_label(MathTex(r"x"))
        y_label = self.axes.get_y_axis_label(MathTex(r"x"))
        z_label = self.axes.get_z_axis_label(MathTex(r"x"))

        # Plot quadratic form
        A = np.array([
            [2.0, -0.5],
            [-0.5, 1.0]
            ])
        a3d = np.array([
            [2.0, -0.5, 0.0],
            [-0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            ])
        a3d_inv = np.linalg.inv(a3d)
        s = sqrtm(A)
        dets = np.linalg.det(s)
        M = np.array([
            [s[0,0], s[0,1], 0],
            [s[1,0], s[1,1], 0],
            [0, 0, 1],
            ])
        b = -np.array([1.0, 2.0])
        b3d = -np.array([1.0, 2.0, 0.0])
        def quad(u, v):
            return np.array([u, v, 0.5*u*u*A[0,0] + 0.5*v*v*A[1,1] + 0.5*u*v*A[1,0] + 0.5*u*v*A[0,1] - u*b[0] - v*b[1]])

        self.surface = Surface(
                lambda u, v: self.axes.c2p(*quad(u, v)),
                u_range=[-2, 2],
                v_range=[-2, 2],
                resolution=10
                )
        self.surface.scale(1, about_point=ORIGIN)
        self.surface.set_style(fill_opacity=0.5, stroke_color=GREEN)

        # Setup dots
        coords_x0 = self.axes.c2p(*quad(point_init[0],point_init[1]))
        print(coords_x0)
        self.x0 = Dot3D(coords_x0, radius=0.1)
        grad_x0 = a3d @ (quad(point_init[0], point_init[1])) - b3d
        grad_coords = self.axes.c2p(*grad_x0)
        conj_grad = M @ grad_x0
        #self.grad = Vector(-grad_x0).move_to(coords_x0-0.5*grad_x0)
        self.grad = Arrow3D(start=coords_x0, end=(coords_x0 - grad_coords))
        #self.conjgrad = Arrow3D(start=coords_x0, end=(coords_x0 - grad_x0))
        #self.diffgrad = Arrow3D(start=M@(coords_x0-grad_x0), end=(M@(coords_x0-grad_x0) - conj_grad))

        xmin_points = a3d_inv @ b3d
        xmin_points = quad(xmin_points[0], xmin_points[1])
        print(xmin_points)
        coords_xmin = self.axes.c2p(*xmin_points)
        self.xmin = Dot3D(coords_xmin, radius=0.1, color=YELLOW)

        new_x0 = M @ coords_x0
        coords_p = new_x0 - coords_x0
        #self.p = Vector(coords_p).move_to(coords_x0+0.5*coords_p)
        self.p = Arrow3D(start=coords_x0, end=(coords_x0 + coords_p))
        self.M = M

        #coords_x1 = coords_x0 - grad_x0
        #self.x1 = Dot3D(coords_x1)
        #grad_x1 = a3d @ quad(*(self.axes.p2c(coords_x1)[0:2])) + b3d
        ##self.grad_x1 = Vector(-grad_x1).move_to(coords_x1-0.5*grad_x1)
        #self.grad_x1 = Arrow3D(start=coords_x1, end=(coords_x1 - grad_x1))
        #self.x0_point = Dot3D(coords_x1-0.5*grad_x0)
        #self.x1_point = Dot3D(coords_x1-0.5*grad_x1)


        def cg(x0, n):
            xmin = xmin_points[2]
            r = b - A @ x0
            p = r
            x = x0
            coords_array = []
            fheight = 0.0
            coords_array.append(self.axes.c2p(*quad(x[0], x[1])))
            fval = quad(x0[0], x0[1])[2]
            cg_array = []
            cg_array.append(self.axes.c2p(*r, fheight))
            conj_array = []
            grad_array = []
            grad_array.append(self.axes.c2p(*r, fheight))
            red_dir = []
            for k in range(n):
                alpha = np.dot(r, r)/ np.dot(p, A @ p)
                x = x + alpha * p
                conj_array.append(self.axes.c2p(*(alpha*p), fheight))
                fheight = 0.0
                coords_array.append(self.axes.c2p(*quad(x[0], x[1])))
                grad_array.append(self.axes.c2p(*(-A @ x + b), fheight))
                norm = np.dot(r, r)
                r = r - alpha * (A @ p)
                red_dir.append(self.axes.c2p(*(-alpha * (A@p))))
                cg_array.append(self.axes.c2p(*r, fheight))
                beta = np.dot(r, r) / norm
                p = r + beta * p
            return (coords_array, cg_array, conj_array, grad_array, red_dir)

        dim = 2
        conj_grad_res = cg(point_init[0:2], dim)
        self.points = []
        self.coords = []
        self.cgs = []
        self.cg_dir = []
        self.gradients = []
        self.red_dir = []
        for i in range(dim):
            coords1 = conj_grad_res[0][i]
            conj = conj_grad_res[2][i]
            self.cg_dir.append(Arrow3D(start=coords1, end=(coords1+conj)))
        for i in range(dim+1):
            coords = conj_grad_res[0][i]
            cg = conj_grad_res[1][i]
            grad = conj_grad_res[3][i]
            self.points.append(Dot3D(coords))
            self.coords.append(coords)
            self.cgs.append(Arrow3D(start=(coords), end=(coords+cg), color=BLUE))
            self.gradients.append(Arrow3D(start=coords, end=(coords+grad), color=YELLOW))
            self.red_dir.append(Arrow3D(start=coords, end=(coords+conj_grad_res[4][i-1]), color=RED))


        # Create MObjects for gradients and company
        #print(xfin)
        #r0 = -grad_x0
        #p0 = r0
        #alpha_0 = np.dot(r0, r0) / np.dot(p0, a3d @ p0)
        #self.cx1 = coords_x0 + alpha_0 * self.axes.c2p(*(p0))
        #self.x1 = Dot3D(self.cx1)
        #grad_x1 = self.axes.c2p(*(a3d @ quad(*(self.axes.p2c(self.cx1)[0:2])) - b3d))
        #self.grad_x1 = Arrow3D(start=self.cx1, end=(self.cx1 - self.axes.c2p(*grad_x1)))
        #self.x0_point = Dot3D(coords_x0-0.5*grad_x0)
        #self.x1_point = Dot3D(self.cx1-0.5*grad_x1)
        ##self.x1_point = Dot3D(coords_x1-0.5*grad_x1)
        #r1 = r0 - alpha_0 * (a3d @ p0)
        #print(np.linalg.norm(r1))
        #beta_0 = np.dot(r1, r1) / np.dot(r0, r0)
        #p1 = r1 + beta_0*p0
        #self.conjgrad_x1 = Arrow3D(start=self.cx1, end=(self.cx1 + self.axes.c2p(*p1)))
        #alpha_1 = np.dot(r1, r1) / np.dot(p1, a3d @ p1)
        #self.cx2 = self.cx1 + alpha_1 * self.axes.c2p(*( p1))
        #r2 = r1 - alpha_1 * (a3d @ p1)
        #print(np.linalg.norm(r2))
        #print(xmin_points)
        #print(self.cx2)


        #invM = np.linalg.inv(self.M)
        #beta_0 = np.dot(grad_x1, grad_x1) / np.dot(grad_x0, grad_x0)
        #p0 = - grad_x1 - beta_0 * grad_x0
        #self.conjgrad_x1 = Arrow3D(start=coords_x1, end=(coords_x1 + p0))
        #alpha_0 = np.dot(grad_x0, grad_x0) / np.dot(grad_x0, a3d @ grad_x0)
        #alpha_1 = np.dot(grad_x1, grad_x1) / np.dot(grad_x1, a3d @ grad_x1)
        #r_1 = - grad_x0 + alpha_0 * a3d @ p0
        #self.x2 = coords_x1 + r_1
        #proj_alpha = -grad_x1 + alpha_0 * grad_x0
        #scalar_product = np.dot(- grad_x1, - a3d @ grad_x0)
        #normsq = np.dot(grad_x0, a3d @ grad_x0)
        #projection = - scalar_product * grad_x0 / normsq
        #vec_start = coords_x1 + proj_alpha
        #vec_start_minv = vec_start
        #orthonormal = -grad_x1 + proj_alpha
        #orthonormal_vec_minv_space = orthonormal

        #test = coords_x1
        #test_end = test - (grad_x1 - proj_alpha)

        #cx1_minv = invM @ coords_x1
        #gx1_minv = invM @ grad_x1
        #self.orthonormal_vec = Arrow3D(start=(test), end=(test_end)).set_color(RED)

        # Setup Contour
        def contour1(z, t, t_range):
            if t < t_range[1]:
                x = t
                one = - x * (A[1,0] + A[0,1]) + 2*b[1]
                disc = one * one - 4 * A[1,1] * (A[0,0] * x*x - 2*b[0]*x - 2*z)
                return [x, (one + np.sqrt(disc)) / (2*A[1,1]), z]
            else:
                e = t - (t_range[1] - t_range[0])
                e = (e - t_range[0]) / (t_range[1] - t_range[0])
                x = (1 - e) * t_range[1] + e * t_range[0]
                one = - x * (A[1,0] + A[0,1]) + 2*b[1]
                disc = one * one - 4 * A[1,1] * (A[0,0] * x*x - 2*b[0]*x - 2*z)
                return [x, (one - np.sqrt(disc)) / (2*A[1,1]), z]


        def trange(z):
            epsilon = 1e-12
            a = (A[1,0]+A[0,1])**2 - 4*A[0,0]*A[1,1]
            bprime = -4 * b[1] * (A[1,0]+A[0,1]) + 8*A[1,1]*b[0]
            c = 8*A[1,1]*z+4*b[1]**2
            t1 = (-bprime + np.sqrt(bprime**2 - 4*a*c))/(2*a)
            t2 = (-bprime - np.sqrt(bprime**2 - 4*a*c))/(2*a)
            if t1 > t2:
                return [t2+epsilon, t1-epsilon]
            else:
                return [t1+epsilon, t2-epsilon]

        ncontours = 10
        self.contours = []
        self.contour_ani = []
        eps = 0
        for z in np.linspace(xmin_points[2]+eps, 4, ncontours):
            print(z)
            t_range = trange(z)
            r = t_range[1] - t_range[0]
            new_trange = [t_range[0], t_range[1] + r]
            cont = ParametricFunction(lambda t: self.axes.c2p(*contour1(z, t, t_range)), t_range=new_trange)
            self.contours.append(cont)
            self.contour_ani.append(ApplyMatrix(M, cont))

        self.trans_text = MathTex(r"\vec{x'}=A^{1/2}\vec{x}").to_corner(UR)
        self.diff_text = MathTex(r"\vec{p}=A^{1/2}\vec{x_0}-\vec{x_0}").next_to(self.trans_text, DOWN)

        self.grad_text = MathTex(r"\vec{r_k}=\vec{\nabla f(x_k)}=A\vec{x_k}-b").to_edge(UP).set_opacity(0.0)
        self.gram_schmidt = MathTex(r"\vec{p_k}=r_k-\sum_{i<k}\frac{\vec{r_k}^TA\vec{p_i}}{\vec{p_i}^TA\vec{p_i}}\vec{p_i}").next_to(self.grad_text, DOWN).set_opacity(0.0)
        self.alpha = MathTex(r"\alpha_k=\frac{\vec{p_k}^T\vec{r_k}}{\vec{p_k}^TA\vec{p_k}}").next_to(self.gram_schmidt, DOWN).set_opacity(0.0)
        self.xk = MathTex(r"\vec{x_{k+1}}=\vec{x_k}+\alpha_k\vec{p_k}").next_to(self.alpha, DOWN).set_opacity(0.0)

    def construct(self):
        self.set_camera_orientation(theta=70 * DEGREES, phi=75 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.5)

        self.play(
                Create(self.axes),
                Create(self.surface),
                Create(self.points[0]),
                #Create(self.gradients[0]),
                #Create(self.conjgrad),
                Create(self.xmin)
        )
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.move_camera(theta=0, phi=0)
        self.wait()
        self.play(
                *(Create(contour) for contour in self.contours),
                FadeOut(self.surface),
        )

        self.add_fixed_in_frame_mobjects(self.trans_text)
        self.play(
                Write(self.trans_text),
        )
        self.wait()
        dot_animation = ApplyMatrix(self.M, self.points[0])
        min_animation = ApplyMatrix(self.M, self.xmin)
        #grad_animation = ApplyMatrix(self.M, self.conjgrad)
        self.play(*self.contour_ani, dot_animation, min_animation)
        self.wait()
        invM = np.linalg.inv(self.M)
        contour_ani = (ApplyMatrix(invM, contour) for contour in self.contours)
        dot_ani = ApplyMatrix(invM, self.points[0])
        min_ani = ApplyMatrix(invM, self.xmin)
        self.play(
                *contour_ani, dot_ani, min_ani
        )
        self.wait()
        #self.play(
        #        Create(self.diffgrad)
        #        )
        #self.wait()

        self.add_fixed_in_frame_mobjects(self.grad_text, self.gram_schmidt, self.xk)
        self.grad_text.set_opacity(1.0)
        self.gram_schmidt.set_opacity(1.0)
        self.play(
                FadeOut(self.trans_text, *self.contours, self.axes, self.points[0], self.xmin),
                Write(self.grad_text),
                Write(self.gram_schmidt),
        )
        self.wait()
        # Cheat Gradient
        new_grad = Arrow3D(start=self.coords[0], end=self.coords[1])
        self.xk.set_opacity(1.0).to_corner(UR)
        self.play(
                FadeIn(*self.contours, self.axes, self.points[0], self.xmin),
                FadeOut(self.grad_text, self.gram_schmidt),
                Write(self.xk)
                )
        self.wait()
        self.play(
                Create(self.gradients[0]),
                Create(self.red_dir[1]),
                Create(new_grad),
        )
        self.wait()
        self.play(self.points[0].animate.move_to(self.coords[1]))
        self.wait()
        #self.play(
        #        self.grad.animate.move_to(self.x0_point),
        #        self.grad_x1.animate.move_to(self.x1_point),
        #        #Create(self.orthonormal_vec)
        #        )
        #self.wait()
        final_grad = Arrow3D(start=self.coords[1], end=self.coords[2])
        self.play(
                Create(self.red_dir[2]),
                Create(final_grad),
                Create(self.cgs[1]),
        )
        self.wait()
        #ani_grad = ApplyMatrix(self.M, self.grad)
        #ani_grad_x1 = ApplyMatrix(self.M, self.grad_x1)
        #ani_x0 = ApplyMatrix(self.M, self.x0)
        #ani_xmin = ApplyMatrix(self.M, self.xmin)
        #ortho_ani = ApplyMatrix(self.M, self.orthonormal_vec)
        #conj_ani = ApplyMatrix(self.M, self.conjgrad_x1)
        #self.play(
        #        *self.contour_ani,
        #        ani_grad,
        #        #ani_grad_x1,
        #        ani_x0,
        #        ani_xmin,
        #        #ortho_ani,
        #        #conj_ani
        #)
        #self.wait()


        self.play(
                self.points[0].animate.move_to(self.coords[2])
        )
        self.wait()
        contour_ani = (ApplyMatrix(self.M, contour) for contour in self.contours)
        dot_ani = ApplyMatrix(self.M, self.points[0])
        min_ani = ApplyMatrix(self.M, self.xmin)
        red_dir_ani = ApplyMatrix(self.M, self.red_dir[1])
        self.play(
                *contour_ani,
                dot_ani,
                min_ani,
                red_dir_ani,
        )
        self.wait()
        #self.set_camera_orientation(theta=70 * DEGREES, phi=75 * DEGREES)
        #self.begin_ambient_camera_rotation(rate=0.5)
        #self.wait(2)

        #self.play(
        #        Create(self.orthonormal_vec)
        #    )
        #self.wait()

