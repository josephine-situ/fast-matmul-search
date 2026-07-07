The inner problem $\max_{Z\in\mathcal W}g_6$ is a semidefinite program in $Z$ (linear objective, linear equalities, PSD cone). Taking its dual with multipliers $\lambda_l$ for the equalities and PSD‑dual matrices $Y^{I,J}$ for the cone, and merging with the outer minimization, yields the SDP:

Theorem (degree 6). $(x,\text{lifts})$ is optimal for the robust problem iff there exist $\{Y^{I,J}\},\lambda$ solving
$$\min\ -\sum_{l=1}^m\lambda_l q_l$$
$$\begin{aligned}
\text{s.t.}\quad
&Y^{I,J}+\textstyle\sum_l\lambda_l A_l^{I,J}\preceq0, &&\forall (I,J),\\
&t_{I,J}+\textstyle\sum_l\lambda_l c_l^{I,J}=0, &&\forall (I,J),\\
&s_{I,J}+\textstyle\sum_l\lambda_l \mu_l^{I,J}=0, &&\forall (I,J),\\
&\begin{pmatrix}Y^{I,J}&t_{I,J}\\ t_{I,J}^T&s_{I,J}\end{pmatrix}\succeq0, &&\forall (I,J),\\
&(x,\text{lifts})\in\mathcal X .
\end{aligned}$$

The equalities/inequalities for $|I\cup J|\le1$ reproduce (4a)–(4l) of Theorem 2, those for $|I\cup J|=2$ reproduce (8a)–(8o) of Theorem 5, and the new layers are the degree‑3 and degree‑4 multipliers.