# Combustion

案例 1 是基于简化模型的一维 FPP 火焰，假设材料属性恒定、Lewis数为单位数、理想气体且无粘性流动。采用一步不可逆反应假设：燃料+氧化剂 → 产物。假设燃料为纯物质，其反应速率 $\omega$ 可通过下式计算：
$$
\begin{equation}
    \omega = Ae^{-\frac{E_a}{RT}}\left(\rho Y_F\right)^\nu
\end{equation}
$$
其中 $A$ 是预指数因子，$E_a$ 是活化能，$R$ 是气体常数，$T$ 是温度，$\rho$ 是气体密度，$Y_F$ 是燃料的质量分数，$\nu$ 是反应级数。

控制常微分方程为：
$$
\begin{gather}
\frac{\mathrm{d}(\rho u)}{\mathrm{d}x} = 0\\
\frac{\mathrm{d} \rho uu}{\mathrm{d}x} = -\frac{\mathrm{d}p}{\mathrm{d}x}\\
\rho uc_p\frac{\mathrm{d}T}{\mathrm{d}x}-\lambda \frac{\mathrm{d}^2T}{\mathrm{d}x^2} = \omega q_F\\
\rho u\frac{\mathrm{d}Y_F}{\mathrm{d}x} - \rho D\frac{\mathrm{d}^2Y_F}{\mathrm{d}x^2} = -\omega  
\end{gather}
$$
其中 $u$ 是速度，$p$ 是压力，$c_p$ 是定压比热容，$\lambda$ 是热导率，$D$ 是扩散系数，$q_F$ 是燃料的热释放率。连续性方程可以简化为：
$$
\begin{equation}
    \rho u = \rho_{\text{in}} u_{\text{in}} = \rho_{\text{in}} s_L
\end{equation}
$$
其中 $s_L$ 是层流火焰速度，是与常微分方程共同求解的特征值。基于理想气体方程
$$
\begin{equation}
    pW = \rho RT
\end{equation}
$$
以及单位 Lewis数的假设
$$
\begin{equation}
    \lambda = \rho c_p D
\end{equation}
$$
可以将控制方程简化为单一的常微分方程：
$$
\begin{equation}
    \rho_{\text{in}} s_L c_p \frac{\mathrm{d}T}{\mathrm{d}x} - \lambda \frac{\mathrm{d}^2T}{\mathrm{d}x^2} = \omega q_F
\end{equation}
$$
其中补充量：
$$
\begin{gather}
    u = \frac{c-\sqrt{c^2-4RT/W}}{2}, \text{s.t. } c=s_L + \frac{RT_\text{in}}{Ws_L} \\
    \rho = \frac{\rho_\text{in}s_L}{u} \\
    Y_F = Y_{F,\text{in}} + \frac{c_p(T_\text{in} - T)}{q_F}
\end{gather}
$$
其中 $W$ 为气体分子量，所有变量 （$\rho, u, p, Y_F, \omega$）均为 函数 $T$ 的函数。


其中物理参数取值：

|参数 | 符号 | 值 (SI) |
|---|---|---|
| Universal gas constant | $R$ | $8.315$|
| Pre-exponential factor | $A$ | $1.4\times10^8$ |
| Reaction order | $\nu$ | $1.6$ |
| Activation energy | $E_a$ | $121417.2$ |
| Molecular weight | $W$ | $0.02899$ |
| Themal conductivity | $\lambda$ | $0.026$ |
| Heat capacity | $c_p$ | $1005$ |
| Fuel calorific value | $q_F$ | $5.0\times10^7$ |

边界、初始条件：
$$
\begin{gather}
    T_\text{in} = 298 K\\
    \frac{\mathrm{d}T}{\mathrm{d}x} = 1.0\times 10^{5} K/m \text{ at } x=0\\
    L = 1.5mm\\
    p_\text{in} = 101325 Pa\\
    \phi = 0.4\\
    Y_{F,\text{in}} = \frac{\phi}{4+\phi}
\end{gather}
$$

参考解通过采用一阶欧拉格式和 10000 个网格的有限差分法求解得到。且 $s_L$ 使用二分法计算。更新 $s_L$ 的标准为：
- 如果 $T>T_{\text{adia}}$，则表明此时火焰熄灭且 $s_L$ 过大；
- 若 $\mathrm{d}T/\mathrm{d}x < 0$，表明回火且 $s_L$ 不足。判定收敛标准为二分区间小于 $10^{-16}$。绝热火焰温度通过设定质量分数 $Y_F=0$ 计算得到。