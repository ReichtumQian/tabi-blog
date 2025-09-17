+++

title = "线性时不变系统的能控性 Controllable"

date = "2025-09-17"

[taxonomies]

tags = ["Control Theory"]

+++

‍

---

## 问题陈述

**线性时不变系统 LTI**：我们将系统视作数学算子 $H$，其接收一个输入信号 $x(t)$，产生一个输出信号 $y(t)$，即

$$
y(t) = H[x(t)].
$$

若其满足以下两个条件，则其是*线性时不变（LTI）系统*：

- 线性性：给定任意两个信号 $x_1(t)$ 和 $x_2(t)$，和常数 $a_1$ 和 $a_2$，满足

$$
H[a_1x_1(t)+a_2x_2(t)]=a_1H[x_1(t)]+a_2H[x_2(t)]
$$

- 时不变性(Time-Invariance)：给定信号 $x(t)$ 和任意时间延迟 $\tau$，满足

$$
y(t-\tau)=H[x(t-\tau)]
$$

**系统模型**：我们考虑一个 LTI 系统，其动态由以下状态空间方程描述：

$$
\dot{\mathbf{x}}(t)=A\mathbf{x}(t)+B\mathbf{u}(t),
$$

其中 $\mathbf{x}(t) \in \mathbb{R}^n$ 是状态向量，$\mathbf{u}(t) \in \mathbb{R}^m$ 是输入/控制向量，$A \in \mathbb{R}^{n \times n}$ 是系统矩阵，$B \in \mathbb{R}^{n \times m}$ 是输入矩阵。

> 在能控性理论中我们不需要考虑输出 $y(t)$，因此此处不写出 $y(t)$ 的方程。

**系统模型的解**：该系统方程的解为

$$
\mathbf{x}(t)=e^{At}\mathbf{x}(0)+\int_0^te^{A(t-\tau)}B\mathbf{u}(\tau)d\tau.
$$

其中 $\mathbf{x}(0)$ 是初始状态，$e^{At}$ 是矩阵指数，定义为 $e^{At}=\sum_{k=0}^\infty\frac{(At)^k}{k!}$。

**能控性问题**：能控性的问题是能否通过选择一个合适的控制输入 $\mathbf{u}(t)$，在有限的时间 $t_f > 0$ 内，将系统从任意初始状态 $\mathbf{x}(0)$ 驱动到任意期望的最终状态 $\mathbf{x}(t_f)$。

---

## 能控性的数学定义

**状态能控性**：上述系统对矩阵 $(A, B)$ 是*状态完全可控的（completely state controllable）* ，如果对任意初始状态 $\mathbf{x}(0) \in \mathbb{R}^n$ 和任意最终状态 $\mathbf{x}_f \in \mathbb{R}^n$，存在有限时间 $t_f > 0$ 和分段连续 $\mathbf{u}(t)$，使得

$$
\mathbf{x}(t_f) = \mathbf{x}_{f}.
$$

**等价问题**：代入系统模型的解，可知状态可控性等价于：对于任意 $\mathbf{x}(0)$ 和 $\mathbf{x}_f$，以下方程关于 $\mathbf{u}(\tau)$ 是否有解？

$$
\mathbf{x}_f-e^{At_f}\mathbf{x}(0)=\int_0^{t_f}e^{A(t_f-\tau)}B\mathbf{u}(\tau)d\tau.
$$

**可达集 Reachable Set**：令左侧项 $\tilde{\mathbf{x}}=\mathbf{x}_{f}-e^{At_{f}}\mathbf{x}(0)$，则问题转化为 $\tilde{\mathbf{x}}$ 是否总能被积分项表示。也就是是否在可达集中：

$$
\mathcal{R}_t = \left\{ \mathbf{x} \in \mathbb{R}^n | \mathbf{x} = \int_0^t e^{A(t-\tau)}B\mathbf{u}(\tau) \mathrm{d} \tau, \quad \mathbf{u}(t) \in \mathcal{U} \right\},
$$

其中 $\mathcal{U}$ 表示分段连续函数 $u: [0,t] \to \mathbb{R}^m$ 组成的集合。

---

## 能控性判据

我们有两个主要的等价判据来判断系统的能控性：能控性格拉姆矩阵 (Controllability Grammian)、卡尔曼能控性判据 (Kalman's Rank Condition)

**能控性格拉姆矩阵 (Controllability Grammian)** ：上述系统是完全可控的，当且仅当对于任意 $t > 0$，下面的*能控性格拉姆矩阵* $W_c(t)$ 是非奇异的：

$$
W_c(t)=\int_0^te^{A\tau}BB^\top e^{A^\top \tau}d\tau,
$$

即 $\operatorname{det}(W_c(t)) \neq 0$。

**卡尔曼能控性判据（Kalman's Rank Condition）** ：上述系统是完全可控的，若下面的能控性矩阵 $\mathcal{C} \in \mathbb{R}^{n \times nm}$ 是满秩的：

$$
\mathcal{C}=
\begin{bmatrix}
B & AB & A^2B & \cdots & A^{n-1}B
\end{bmatrix}
$$

即 $\operatorname{rank}(\mathcal{C}) = n$。

‍
