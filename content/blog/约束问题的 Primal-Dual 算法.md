+++

title = "约束问题的 Primal-Dual 算法"

date = "2025-09-19"

[taxonomies]

tags = ["Optimization"]

+++

‍

---

## 约束优化问题与拉格朗日函数

**约束优化问题**：我们考虑如下的约束优化问题，$x$ 是我们的<u>优化变量（Primal Variable）</u>，$f(x)$ 是<u>目标函数（Objective function）</u>，$g_i$ 和 $h_j$ 是<u>约束条件（Constraints）</u>

$$
\begin{aligned}
  \min_{x\in\mathbb{R}^{n}} \quad &f(x) \\
  \mathrm{s.t.} \quad &g_i(x)\leq0,\quad i=1,\ldots,m \\
  & h_j(x)=0,\quad j=1,\ldots,p
\end{aligned}
$$

我们将上述问题称为原始问题（Primal Problem）。

**拉格朗日函数**：我们引入<u>拉格朗日乘子（Lagrange Multiplier）</u>$\lambda_i \geq 0$ 和 $\nu_j$，也称为<u>对偶变量（Dual Variables）</u>，并定义拉格朗日函数

$$
\mathcal{L}(x,\lambda,\nu)=f(x)+\sum_{i=1}^m\lambda_ig_i(x)+\sum_{j=1}^p\nu_jh_j(x)
$$

其中 $\lambda=[\lambda_1,\ldots,\lambda_m]^T$ 且 $\lambda_i\geq0$，$\nu_j \in \mathbb{R}$ 。

**拉格朗日函数的性质分析**：若我们固定 $x$，尝试最大化 $\mathcal{L}(x, \lambda, \nu)$

$$
\max_{\lambda\geq0,\nu}\mathcal{L}(x,\lambda,\nu)
$$

- 如果 $x$ 违反了约束：比如 $g_k(x) > 0$，那么我们可以让 $\lambda_k \to \infty$，从而 $\mathcal{L} \to \infty$。比如 $h_j(x) \neq 0$，我们可以让 $\nu_j \to \pm \infty$，这也导致 $\mathcal{L} \to \infty$。
- 如果 $x$ 满足所有条件：$g_i(x) \leq 0$ 且 $h_j(x) = 0$，为了最大化 $\mathcal{L}$，我们会让所有 $\lambda_i = 0$，即 $\max_{\lambda\geq0,\nu}\sum\lambda_ig_i(x)+\sum\nu_jh_j(x)=0$。

$$
\max_{\lambda\geq0,\nu}\mathcal{L}(x,\lambda,\nu)=
\begin{cases}
f(x) & \mathrm{if~}x\text{ is feasible }(\text{满足所有约束}) \\
\infty & \mathrm{if~}x\text{ is infeasible }(\text{违反任何约束}) & 
\end{cases}
$$

**原优化问题的拉格朗日函数表达（对偶问题）** ：我们最初的约束优化问题 $\min f(x)$ 等价于下面的无约束问题：

$$
\min_x\left(\max_{\lambda\geq0,\nu}\mathcal{L}(x,\lambda,\nu)\right)
$$

或者我们可以将最大、最小的顺序对调一下，获得<u>对偶问题（Dual Problem）</u>：

$$
\max_{\lambda\geq0,\nu}\left(\min_x\mathcal{L}(x,\lambda,\nu)\right).
$$

> 大多数情况下，原始问题和对偶问题的解是相同的。最优解 $(x^*,\lambda^*,\nu^*)$ 位于拉格朗日函数的一个鞍点（Saddle Point）上。

---

## Primal-Dual 算法

**Primal-Dual 算法核心思想**：我们不去直接解 minimax 或 maximin 问题，而是通过迭代的方式同时寻找原始变量 $x$ 和对偶变量 $(\lambda, \nu)$，直到它们收敛到鞍点。

**Primal-Dual 算法**：Primal-Dual 算法分为 Primal 和 Dual 两步，以梯度下降和梯度上升为例

- Primal Variable：梯度下降

$$
x^{(k+1)}\leftarrow x^{(k)}-\eta_x\nabla_x\mathcal{L}(x^{(k)},\lambda^{(k)},\nu^{(k)}).
$$

- Dual Variable：梯度上升。同时由于 $\lambda \geq 0$，因此需要将其投影到非负数

$$
\lambda^{(k+1)} \leftarrow \max \left(0,  \lambda^{(k)} + \eta_\lambda \nabla_\lambda \mathcal{L}(x^{(k)}, \lambda^{(k)}, \nu^{(k)}) \right)
$$

$$
\nu^{(k+1)} \leftarrow \nu^{(k)} + \eta_\nu \nabla_\nu \mathcal{L}(x^{(k)}, \lambda^{(k)}, \nu^{(k)})
$$

‍
