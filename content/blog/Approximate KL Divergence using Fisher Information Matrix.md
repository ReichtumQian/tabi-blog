+++

title = "Approximate KL Divergence using Fisher Information Matrix"

date = "2025-09-18"

[taxonomies]

tags = ["Statistics", "Machine Learning"]

+++

> The proof comes from [[1801.10112] Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence](https://arxiv.org/abs/1801.10112)

---

## 基本概念

**KL 散度**：给定两个分布 $P(x)$ 和 $Q(x)$，它们之间的 KL 散度为

$$
D_{KL}(P || Q) = \mathbb{E}_{x \sim P}[\log P(x) - \log Q(x)].
$$

> 为什么 KL 散度不是两个分布的对数直接减一下？因为信息论中真实分布是 $P$，而我们用 $Q$ 去编码数据，因此需要从 $P$ 中去取数据。

**Fisher 矩阵**：给定概率密度函数 $p(x | \theta)$，Fisher 信息矩阵定义为

$$
F(\theta):=\mathbb{E}_{x\sim p(\cdot\mid\theta)}\Bigg[\left(\frac{\partial}{\partial\theta}\log p(x|\theta)\right)\left(\frac{\partial}{\partial\theta}\log p(x|\theta)\right)^\top\Bigg]
$$

其中 $u_i (x;\theta) :=\frac{\partial}{\partial\theta_i}\log p(x|\theta)$ 为 $\theta_i$ 的 score function。

**Fisher 矩阵的对角元**：一般我们假设 score function 的期望为 $0$，即

$$
\mathbb{E}_{x\sim p(\cdot|\theta)}[u_i(x;\theta)]=0.
$$

此时 Fisher 矩阵的对角元素

$$
F_{ii} = \mathbb{E}\left[u_i^2\right] = \mathbb{E} \left[(u_i - \mathbb{E}[u_i])^2 \right] = \operatorname{Var}(u_i).
$$

即表示对于真实数据产生的样本，参数 $\theta_i$ 对对数似然的一阶梯度的方差。<u>如果方差很大，则说明不同样本会给出很不一样的导数信号</u>，也就是 $\theta_i$ 很重要。

---

## KL 散度与 Fisher 矩阵的关系

**使用 Fisher 矩阵逼近 KL 散度**：设 $\Delta \theta \to 0$，则

$$
D_{KL}(p_{\theta}\|p_{\theta+\Delta\theta})\approx\frac{1}{2}\Delta\theta^{\top}F_{\theta}\Delta\theta,
$$

其中 $F_\theta$ 为 $\theta$ 处的 Fisher 矩阵。

**证明**：这里我们记 $p_\theta(\mathbf{z})=p_\theta(\mathbf{y}|\mathbf{x})$ 和 $\mathbb{E}_{\mathbf{z}}[\cdot]=\mathbb{E}_{\mathbf{x}\sim\mathcal{D},\mathbf{y}\sim p_{\theta}(\mathbf{y}|\mathbf{x})}[\cdot]$。根据 KL 散度的定义

$$
D_{KL}(p_{\theta}(\mathbf{z})\|p_{\theta+\Delta\theta}(\mathbf{z}))=\mathbb{E}_{\mathbf{z}}\left[\log p_{\theta}(\mathbf{z})-\log p_{\theta+\Delta\theta}(\mathbf{z})\right].
$$

将 $\log p_{\theta + \Delta \theta}(\mathbf{z})$ 在 $\theta$ 处展开

$$
\log p_{\theta+\Delta\theta}\approx\log p_{\theta}+\Delta\theta^{\top}\frac{\partial\log p_{\theta}}{\partial\theta}+\frac{1}{2}\Delta\theta^{\top}\frac{\partial^{2}\log p_{\theta}}{\partial\theta^{2}}\Delta\theta .
$$

将 $\log p_{\theta+\Delta\theta}$ 的展开式代入到第一个式子，并且消去 $\mathbb{E}_{\mathbf{z}}[\log p_{\theta}(\mathbf{z})]$ 得到

$$
\begin{aligned}
D_{KL}(p_{\theta}\|p_{\theta+\Delta\theta})\approx-\:\Delta\theta^\top\:\mathbb{E}_\mathbf{z}\left[\frac{\partial\log p_\theta}{\partial\theta}\right]-\frac{1}{2}\Delta\theta^\top\:\mathbb{E}_\mathbf{z}\left[\frac{\partial^2\log p_\theta}{\partial\theta^2}\right]\Delta\theta
\end{aligned} \tag{1}
$$

对于 (1) 式右侧第一项，由于 $\mathbf{x} \sim \mathcal{D}$ 以及 $\mathbf{y} \sim p_\theta(\mathbf{y}|\mathbf{x})$，通过计算可以将其消去：

$$
\begin{aligned}\mathbb{E}_{\mathbf{z}}\left[\frac{\partial\log p_{\theta}(\mathbf{z})}{\partial\theta}\right]&=\mathbb{E}_{\mathbf{x}\sim\mathcal{D}}\left[\sum_{\mathbf{y}}p_{\theta}(\mathbf{y}|\mathbf{x})\frac{\partial\log p_{\theta}(\mathbf{y}|\mathbf{x})}{\partial\theta}\right]\:,\\&=\mathbb{E}_{\mathbf{x}\sim\mathcal{D}}\left[\sum_{\mathbf{y}}p_{\theta}(\mathbf{y}|\mathbf{x})\frac{1}{p_{\theta}(\mathbf{y}|\mathbf{x})}\frac{\partial p_{\theta}(\mathbf{y}|\mathbf{x})}{\partial\theta}\right]\:,\\&=\mathbb{E}_{\mathbf{x}\sim\mathcal{D}}\left[\frac{\partial}{\partial\theta}\sum_{\mathbf{y}}p_{\theta}(\mathbf{y}|\mathbf{x})\right]\:,\\&=\mathbb{E}_{\mathbf{x}\sim\mathcal{D}}[0]=0\:.\end{aligned} \tag{2}
$$

对于 (1) 式右侧第二项

$$
\frac{\partial^2\log p}{\partial\theta^2}=\frac{\partial}{\partial\theta}\left(\frac{1}{p}\frac{\partial p}{\partial\theta}\right) 
\Rightarrow 
\frac{\partial^2\log p}{\partial\theta^2}=-\frac{1}{p^2}\frac{\partial p}{\partial\theta}\frac{\partial p}{\partial\theta}^\top+\frac{1}{p}\frac{\partial^2p}{\partial\theta^2}
$$

其中

$$
\frac{1}{p^2}\frac{\partial p}{\partial\theta}\frac{\partial p}{\partial\theta}^\top=\left(\frac{\partial\log p}{\partial\theta}\right)\left(\frac{\partial\log p}{\partial\theta}\right)^\top
$$

因此得到 (1) 式右侧第二项为

$$
\begin{aligned}
 & \mathbb{E}_{\mathbf{z}}\left[-\frac{\partial^{2}\operatorname{log}p_{\theta}(\mathbf{z})}{\partial\theta^{2}}\right]=-\mathbb{E}_{\mathbf{z}}\left[\frac{1}{p_{\theta}(\mathbf{z})}\frac{\partial^{2}p_{\theta}(\mathbf{z})}{\partial\theta^{2}}\right] \\
 & +\mathbb{E}_{\mathbf{z}}\left[\left(\frac{\partial\log p_\theta(\mathbf{z})}{\partial\theta}\right)\left(\frac{\partial\log p_\theta(\mathbf{z})}{\partial\theta}\right)^\top\right], \\
 & =-\mathbb{E}_{\mathbf{z}}\left[\frac{1}{p_{\theta}(\mathbf{z})}\frac{\partial^{2}p_{\theta}(\mathbf{z})}{\partial\theta^{2}}\right]+\tilde{F}_{\theta}.
\end{aligned} \tag{3}
$$

这里 $\tilde{F}_\theta$ 为 True Fisher，其与 Empirical Fisher 的区别在于其期望是取自模型分布 $\mathbf{x} \sim \mathcal{D}$ 以及 $\mathbf{y} \sim p_\theta(\mathbf{y}|\mathbf{x})$ 而非真实分布 $(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}$：

$$
{F}_\theta=\mathbb{E}_{\mathbf{z}\sim p_\theta}
\begin{bmatrix}
g(\mathbf{z};\theta)g(\mathbf{z};\theta)^\top
\end{bmatrix}, \quad 
{F}_\theta=\mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\mathcal{D}}\left[g(\mathbf{x},\mathbf{y};\theta)g(\mathbf{x},\mathbf{y};\theta)^\top\right]
$$

仿照 (2) 的推导过程，我们可以得到 (3) 右侧第一项也为 $0$。并且在 optimum 最优点处，模型的分布 $\mathbf{x} \sim \mathcal{D}, \mathbf{y} \sim p_\theta(\mathbf{y}|\mathbf{x})$ 逼近真实分布 $(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}$，此时 $F_\theta = \tilde{F}_\theta$，因此

$$
D_{KL}(p_{\theta}\|p_{\theta+\Delta\theta})\approx \tilde{F}_{\theta} \approx F_\theta.
$$



‍

‍
