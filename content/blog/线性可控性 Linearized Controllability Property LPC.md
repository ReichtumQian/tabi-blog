+++

title = "线性可控性 Linearized Controllability Property LPC"

date = "2025-09-24"

[taxonomies]

tags = ["Control Theory"]

+++

**LPC（Linearized Controllability Property，线性化可控性）**  是一个用来判断我们能否**在局部**有效控制一个**复杂非线性系统**的工具。

它问的是这样一个问题：如果系统沿着一条预定轨迹（路径）运动时，受到一点点微小的干扰偏离了轨迹，我们有没有能力通过微调控制输入，让它回到我们想要的任何一个附近的微小位置上？如果答案是“有”，那么系统就具备LPC。

---

## 数学语言描述

**系统与轨迹**：考虑一个非线性系统

$$
\dot{x}(t)=f(x(t),u(t)) \tag{1}
$$

其中 $t \in [0, T]$，$x(t) \in \mathbb{R}^n$，$u(t) \in \mathbb{R}^m$，以及 $f \in C^1(\mathbb{R}^m, \mathbb{R}^n)$。给定初始状态 $x_0$ 和控制输入函数 $u: [0, T] \to \mathbb{R}^m$，记真实轨迹 $x^\ast(t) = \varphi_t(u, x_0)$ 满足：

$$
\dot{x}^*(t)=f(x^*(t),u(t)),\quad x^*(0)=x_0.
$$

**沿轨迹的线性化**：考虑状态 $x$ 和控制 $u$ 的微小扰动

$$
x(t)=x^*(t)+\delta x(t),
\quad u_{new}(t)=u(t)+\delta u(t)
$$

代入原系统方程 (1) 并对 $f$ 泰勒展开，我们可以得到 $\delta x(t)$ 的变分方程

$$
\dot{\delta}x(t)=A(t)\delta x(t)+B(t)\delta u(t).
$$

$$
A(t):=\left.\frac{\partial f}{\partial x}\right|_{(x=x^*(t),u=u(t))}, \quad B(t):=\frac{\partial f}{\partial u}\bigg|_{(x=x^*(t),u=u(t))}
$$

此为一个线性时变（Linear Time-Varying, LTV）系统。

**线性化可控性 LPC**：给定非线性系统 $\dot{x}(t)=f(x(t),u(t))$，初始状态 $x_0$，控制输入 $u(t)$，以及真实轨迹 $x^\ast(t)$。该系统在 $t \in [0, T]$ 上是*线性化可控*的若对应的线性时变系统

$$
\dot{z}(t)=A(t)z(t)+B(t)v(t) \tag{2}
$$

$$
A(t):=\left.\frac{\partial f}{\partial x}\right|_{(x=x^*(t),u=u(t))}, \quad B(t):=\frac{\partial f}{\partial u}\bigg|_{(x=x^*(t),u=u(t))}
$$

在 $[0, T]$ 上是可控的。

> (2) 的可控性保证了 $\delta x$ 是可控的，也就是说可以施加微小控制 $\delta u(t) = v(t)$，将偏差 $\delta x(t) = z(t)$ 拉到期望值。

**等价条件**：LTV 系统 (2) 是可控的等价于其在 $[0, T]$ 上的<u>可控性格拉姆矩阵（Controllability Gramian）</u>$W_C(0, T)$ 是非奇异的：

$$
W_C(0,T)=\int_0^T\Phi(T,\tau)B(\tau)B^T(\tau)\Phi^T(T,\tau)d\tau
$$

其中 $\Phi(t, \tau)$ 满足

$$
\dot{\Phi}(t,\tau)=A(t)\Phi(t,\tau), \quad \Phi(\tau,\tau)=I.
$$

‍
