# GeoNLA：当地理圈层遇上神经网络层——一个关于结构化归纳偏置的理论与实验

**马睿**

---

## 摘要

地理系统天然具有分层、层次化的结构。地球的物理圈层与人文圈层——岩石圈、大气圈、水圈、土壤圈、生物圈和人类圈——通过级联因果变换相互作用，共同决定了我们在地球表面观察到的一切：从土地利用格局到环境质量，从区域发展轨迹到城市扩张边界。这一结构与人工神经网络中的分层传播存在深层相似性，但此前从未有人将其在地理信息科学中形式化地表述出来。

我们提出了**地理-神经层叠架构（GeoNLA）**，严格形式化了神经网络层架构与地理多圈层系统之间的结构同构性。在GeoNLA中，每个地理圈层对应一个专用网络层，层间耦合权重是时变的，以捕捉地理过程的非平稳演化。本文首先给出GeoNLA的理论框架与数学形式化，然后通过两层简化模型的消融实验——在合成数据与真实栅格数据（临夏盆地）两条路线上——对核心假设进行受控验证。实验结果表明，具有乘性调制的分层架构在建模地理人地关系方面优于参数匹配的平坦MLP基线，在合成数据上RMSE降低6.6%、MAE降低13.3%，在真实数据上实现了方向一致的提升（RMSE −2.7%）。

**关键词：** GeoAI，地理信息系统，神经网络架构，分层系统，土地利用模拟，环境监测，地理空间基础模型

---

**Abstract:** Geographic systems exhibit a characteristic layered structure in which Earth's physical and human spheres—lithosphere, atmosphere, hydrosphere, pedosphere, biosphere, and anthroposphere—interact through sequential causal transformations. This architecture bears a deep structural resemblance to the layered computation of artificial neural networks, yet this analogy has never been formally articulated in Geographic Information Science. We propose the **Geo-neural Layered Architecture (GeoNLA)**, which rigorously formalizes the structural isomorphism between neural network layer architectures and geographic multi-sphere systems. In GeoNLA, each geographic sphere corresponds to a dedicated network layer, and inter-layer coupling weights are time-varying to capture the non-stationary evolution of geographic processes. This paper first presents the theoretical framework and mathematical formalization of GeoNLA, then conducts controlled ablation experiments using a two-layer simplified model on two data routes: synthetic data whose label generation mirrors the GeoNLA cascade, and real raster data from the Linxia Basin. Results show that the layered architecture with multiplicative modulation outperforms a parameter-matched flat MLP baseline in modeling geographic human-land relationships, achieving a 6.6% RMSE reduction and 13.3% MAE reduction on synthetic data, with consistent improvements on real data (RMSE −2.7%).

**Keywords:** GeoAI, geographic information systems, neural network architecture, layered systems, land use simulation, environmental monitoring, geospatial foundation models

---

## 1 引言

地理系统是科学所知的最复杂系统之一。地球表面由多个自然圈层与人文圈层的持续交互所塑造，每个圈层贡献独特但相互依赖的影响层次。这些地理圈层充当着一系列信息处理层，其中一个圈层的输出成为下一个圈层的输入，逐步将原始物理输入——太阳辐射、构造力、降水——转化为在地球表面观察到的多维地理现象。

我们在研究过程中注意到一个直觉上颇为强烈的类比：在深度学习中，信息通过连续的层——输入层、隐藏层和输出层——向前传播，每一层施加学习到的变换以提取越来越抽象的表征。在地理系统中，一个类似的过程在展开。基岩地质决定了地形和海拔；地形通过地形效应调节气候；气候支配着降水和温度模式；这些因素又塑造了水文网络；水分供给制约着土壤形成；土壤质量决定了植被类型；而植被与所有前述因素相结合，最终影响着人类土地利用决策。每个地理"层"将其前驱层的输出变换为新的、更复杂的地理变量。

然而，这一类比虽然直觉上很有吸引力，却尚未在GIScience中得到系统化的形式表述。现有方法倾向于将空间因素作为统计模型中的平面特征向量，或者作为深度学习架构的输入，而未显式地编码地理过程的分层因果结构。这代表了一个重大的机遇丧失——如果地理系统的深层架构能够被显式化，它将为设计不仅应用于地理数据、而且在结构上与地理过程本身同构的神经网络架构提供原则性框架。

本文提出了**地理-神经层叠架构（GeoNLA）**，该框架：（1）形式化了神经网络分层架构与地理多圈层系统之间的结构类比；（2）将时间动态整合为控制跨地理层状态转移的核心演化参数；（3）设计了捕捉多空间分辨率地理过程的多尺度架构；（4）通过临夏盆地的实验验证对核心假设进行了消融测试。

---

## 2 理论框架

### 2.1 核心概念类比：圈层即网络层

GeoNLA的概念基础建立在一个结构同构性之上：神经网络的前向传播与地理圈层间的级联因果交互在形式上是同构的。下表给出了这一映射的完整形式化定义。

**表1：地理圈层与神经网络层的映射**

| 地理圈层 | 神经网络层 | 地理功能 | 神经网络类比 |
|---------|----------|---------|-----------|
| 岩石圈 | 输入层 (L1) | 地形、地质 | 原始特征输入 |
| 大气圈 | 隐藏层 (L2) | 气候变量 | 特征变换 |
| 水圈 | 隐藏层 (L3) | 水分分布 | 衍生特征 |
| 土壤圈 | 隐藏层 (L4) | 土壤属性 | 高阶特征 |
| 生物圈 | 隐藏层 (L5) | 生态系统格局 | 抽象特征 |
| 人类圈 | 输出层 (L6) | 土地利用、发展 | 预测目标 |

地理信息从更深层（更基本、变化更慢）的圈层流向表层（更动态、变化更快）的圈层，每个圈层对来自下层圈层的输入进行变换。这与深度神经网络中的前向传播在结构上是同构的。这个类比并非隐喻性的修辞，我们给出了严格的数学形式化。

### 2.2 数学表述

设位置 $\mathbf{x}$ 和时间 $t$ 处的地理状态表示为圈层特定变量的向量：

$$\mathbf{S}_{\text{geo}}(\mathbf{x}, t) = \bigl[\mathbf{L}(\mathbf{x}, t),\; \mathbf{A}(\mathbf{x}, t),\; \mathbf{H}(\mathbf{x}, t),\; \mathbf{P}(\mathbf{x}, t),\; \mathbf{V}(\mathbf{x}, t),\; \mathbf{U}(\mathbf{x}, t)\bigr]$$

其中 $\mathbf{L}$、$\mathbf{A}$、$\mathbf{H}$、$\mathbf{P}$、$\mathbf{V}$ 和 $\mathbf{U}$ 分别表示岩石圈、大气圈、水文、土壤圈、生物圈和人类圈变量。

在GeoNLA中，从圈层 $k$ 到圈层 $k{+}1$ 的变换建模为可学习函数：

$$\mathbf{h}_{k+1}(\mathbf{x}, t) = \sigma\!\left(\mathbf{W}_k \,\mathbf{h}_k(\mathbf{x}, t) + \mathbf{b}_k\right)$$

其中 $\mathbf{h}_k$ 表示地理层 $k$ 处的隐藏状态，$\mathbf{W}_k$ 和 $\mathbf{b}_k$ 为可学习参数（类比于突触权重），$\sigma$ 为非线性激活函数。

**时间动态**是整个框架的关键创新点。我们将时间动态通过时变权重引入：

$$\mathbf{W}_k(t) = \mathbf{W}_k^0 + \Delta\mathbf{W}_k(t)$$

这允许圈层间关系随时间演化，捕捉地理过程中的非平稳性。时间不是被当作一个静态特征丢进模型，而是真正作为驱动状态转移的演化参数。

### 2.3 关键对应关系

这一形式化建立了若干结构性对应：

- **前向传播** ↔ 地理圈层间的级联因果交互
- **激活函数** ↔ 地理阈值与变换过程（例如，从气候到植被的生态阈值，类比于ReLU）
- **跳跃连接** ↔ 绕过中间圈层的直接地理影响（例如，地质对土地利用的直接控制绕过土壤形成过程）
- **反向传播** ↔ 敏感性分析（输出变化如何向后传播以识别关键输入因素）

### 2.4 模型架构组件

在理论层面，GeoNLA集成了四类组件：

**卷积层**负责圈层内空间特征提取。每个地理层 $k$ 包含2–4个卷积块，卷积核大小为 $3{\times}3$ 和 $5{\times}5$，层内跳跃连接允许细粒度空间细节与抽象特征并行传播。每层的通道数反映对应圈层变量的维度。

**图神经网络层**建模地理单元之间的空间交互。图注意力网络（GAT）学习位置特定的空间权重，消息传递同时在层内（圈层内的空间交互）和层间（跨圈层影响）进行，通过学习的注意力权重捕捉托布勒地理学第一定律。

**时序建模**通过三种互补机制整合：带膨胀的因果卷积（TCN）捕捉多尺度时间依赖；时序Transformer编码器捕捉长程依赖并识别关键转变点；循环状态更新确保地理圈层交互随内源动态和外源冲击而演化：

$$\mathbf{W}_k(t{+}1) = f\!\left(\mathbf{W}_k(t),\; \mathbf{h}_k(t),\; \mathbf{c}(t)\right)$$

其中 $\mathbf{c}(t)$ 表示外部时间驱动因素（如政策变化、气候趋势、经济指标）。

**基础模型整合**方面，我们借鉴了Prithvi地理空间基础模型的设计思路 [1]，利用预训练的地理空间模型为特定圈层提取特征，同时保持地理层结构的完整性。

---

## 3 实验设计

为了验证GeoNLA的核心理论——地理圈层叠加过程与神经网络层级计算之间存在结构同构性，我们设计了两层简化模型进行消融实验。完整理论假设了六个地理圈层，但实验验证先从最简形式开始，这是一个务实的选择：如果两层简化都捕捉不到信号，那六层就更难说了。

### 3.1 两层简化模型

GeoNLA的两层简化版通过三个计算阶段实现地理圈层叠加类比：

**阶段1——物理层**（地理圈层变换）：

$$h_1 = \text{ReLU}(\text{Linear}(X) + b_1), \quad h_1 \in \mathbb{R}^{32}$$
$$h_1 = \text{Dropout}(h_1, p=0.2)$$

该层对应于从物理地理圈层（地形 → 气候 → 土壤 → 植被）到涌现潜在表示的变换。32维隐藏空间捕捉了共同约束人类活动的综合地理条件。

**阶段2——乘性时间调制：**

$$h_1(t) = h_1 \odot (1 + \theta_{\text{time}} \odot f_{\text{time}})$$

其中 $\theta_{\text{time}} \in \mathbb{R}^{32}$ 为可学习调制参数向量，$f_{\text{time}}$ 为时间因子向量。这实现了理论框架中的 $W_k(t)$ 公式，捕捉季节和时序动态如何调制物理-人类耦合。

**阶段3——人类层**（人地关系涌现）：

$$y = \sigma(\text{Linear}(h_1(t)) + b_2), \quad y \in \mathbb{R}^{6}$$

**阶段4——社会经济调制：**

$$y(t) = y \odot (1 + \theta_{\text{socio}} \odot f_{\text{socio}})$$

其中 $\theta_{\text{socio}} \in \mathbb{R}^{6}$ 为可学习社会经济调制参数。**总可训练参数量：652。**

![两层GeoNLA模型架构](../../figures/fig1_model_architecture.png)
*图1：两层GeoNLA模型架构。物理层将12维地理输入变换为32维涌现表示，随后进行乘性时间调制，人类层将其映射到6维输出并附加社会经济调制。*

### 3.2 基线模型

作为对照的FlatMLP基线移除了所有GeoNLA假设特有的结构归纳偏置：

- **输入拼接：** 地理特征（12维）、时间因子和社会经济因子拼接为单一14维输入向量
- **两层MLP：** $\text{Linear}(14 \to 32) + \text{ReLU} + \text{Dropout}(0.2) \to \text{Linear}(32 \to 6) + \text{Sigmoid}$
- **无圈层涌现**，无乘性时间或社会经济调制

**总可训练参数量：678。**

参数量紧密匹配（652 vs. 678），这是实验设计的关键——任何性能差异反映的是架构归纳偏置而非模型容量。

### 3.3 合成数据

为了创建一个受控的测试环境，我们生成了标签生成过程精确匹配GeoNLA两阶段级联的合成数据，模拟类似临夏盆地的地理环境。

**表2：按地理圈层组织的合成输入特征（12维）**

| 圈层 | 变量 | 维度 |
|-----|------|------|
| 地形 | 海拔、坡度、起伏度 | 3 |
| 气候 | 温度、降水、干旱指数 | 3 |
| 土壤 | 有机质、黏粒含量、持水能力 | 3 |
| 植被 | NDVI、NDVI振幅、草地比例 | 3 |

**表3：合成输出标签（6维）**

| 变量 | 描述 |
|------|------|
| 耕地比例 | 农田面积占比 |
| 灌溉比例 | 灌溉面积占比 |
| 建设用地 | 建成区比例 |
| 人口密度 | 单位面积人口数 |
| 夜间灯光 | 辐射强度 |
| 道路密度 | 单位面积道路长度 |

标签生成过程精确对应GeoNLA的同构两阶段公式：

1. **物理涌现：** $h_1 = \text{ReLU}(X W_1 + b_1)$，其中 $X \in \mathbb{R}^{N \times 12}$，$W_1 \in \mathbb{R}^{12 \times 32}$。
2. **乘性时间调制：** $h_1(t) = h_1 \odot (1 + \Delta W \odot t)$，其中 $\Delta W \in \mathbb{R}^{32}$ 为可学习调制向量。
3. **人地变换：** $y = \sigma(h_1(t) W_2 + b_2) \odot (1 + \text{社会经济调制})$。

这一数据生成过程确保GeoNLA的归纳偏置——具有乘性调制的分层级联——构成合成任务的"正确答案"。数据集共400个样本，按80/20划分（320训练，80验证）。

### 3.4 真实栅格数据（临夏盆地）

**研究区域。** 临夏盆地（35.3°N–35.8°N, 102.9°E–103.5°E）是甘肃省西北部的一个山间盆地，地形复杂，海拔从约1,700 m到超过3,500 m不等，属于半干旱大陆性季风气候，人类土地利用呈现从密集的农业河谷到稀疏的牧业高地的梯度变化。选择这个区域是因为它的地理梯度足够丰富——在短短几十公里内，你能看到从河谷农耕到高原牧业的完整过渡。

**表4：真实栅格实验的数据源与分辨率**

| 数据产品 | 变量 | 分辨率 |
|---------|------|-------|
| SRTM DEM | 海拔、派生坡度、起伏度 | 30 m → ~1 km |
| WorldClim v2 | 温度、降水、干旱指数 | ~1 km |
| SoilGrids v2.0 | 有机碳、黏粒含量、持水能力 | 250 m → ~1 km |
| Sentinel-2 | NDVI均值、NDVI振幅、草地比例 | 10 m → ~1 km |
| ESA WorldCover | 土地覆被分类 | 10 m → ~1 km |
| WorldPop | 人口密度 | ~1 km |
| VIIRS夜间灯光 | 辐射强度 | ~500 m → ~1 km |

所有栅格图层均重采样至统一的~1 km网格。6个输出标签由栅格产品派生：建设用地比例来自WorldCover，人口密度来自WorldPop，夜间灯光来自VIIRS，道路密度来自OpenStreetMap派生图层，耕地比例和灌溉比例由WorldCover中的农田亚类结合灌溉数据集派生。

坦率地说，真实栅格数据的实验条件远不如合成数据那么"干净"——标签由遥感产品派生而非直接观测，引入了分类误差、空间配准偏差和时间不匹配。但这正是我们想要的：如果GeoNLA的优势在噪声环境中仍然存在，那它的说服力就大得多。

### 3.5 实验设置

**表5：超参数配置**

| 超参数 | 值 |
|-------|---|
| 优化器 | Adam |
| 学习率 | 1 × 10⁻³ |
| 权重衰减 | 1 × 10⁻⁴ |
| 学习率调度器 | ReduceLROnPlateau (patience=15, factor=0.5) |
| 损失函数 | MSE |
| 训练轮数 | 200 |
| 批次大小 | 32 |
| 随机种子 | 42 |

所有随机种子（Python、NumPy、PyTorch CPU/CUDA）均固定为42，确保实验的可复现性。

---

## 4 实验结果

### 4.1 合成数据实验

**表6：合成数据上的整体验证指标（400样本，80/20划分）**

| 模型 | 参数量 | RMSE ↓ | MAE ↓ | R²（均值）↑ |
|------|-------|--------|-------|-----------|
| GeoNLA | 652 | 0.1140 | 0.0790 | 0.7918 |
| FlatMLP | 678 | 0.1221 | 0.0912 | 0.7621 |
| **改进** | — | **−6.6%** | **−13.3%** | **+0.030** |

GeoNLA在三个指标上均实现了提升。MAE的降幅（−13.3%）比RMSE（−6.6%）更明显，这个现象值得留意：MAE对大误差更敏感，说明GeoNLA的改进主要体现在减少了那些"离谱"的预测偏差上。换句话说，分层架构不是均匀地提升所有预测，而是更有效地避免了严重错误。

**表7：合成数据上的逐维度 R²**

| 输出维度 | GeoNLA | FlatMLP | Δ |
|---------|--------|---------|---|
| 耕地比例 | 0.710 | 0.651 | +0.059 |
| 灌溉比例 | 0.822 | 0.746 | +0.076 |
| 建设用地 | 0.711 | 0.700 | +0.011 |
| 人口密度 | 0.891 | 0.880 | +0.011 |
| 夜间灯光 | 0.829 | 0.838 | −0.009 |
| 道路密度 | 0.787 | 0.757 | +0.030 |

逐维度分析的结果很有启发性。改进最大的两个维度——灌溉比例（+0.076）和耕地比例（+0.059）——恰好都是受到物理地理级联（地形 → 气候 → 土壤 → 水资源可利用性）最强烈控制的变量。这与GeoNLA的假设完全吻合：这些输出从分层地理变换中获益最多。而夜间灯光出现了微小的下降（−0.009），这并不令人意外——夜间灯光与物理地理的关系相对直接，较少通过圈层叠加级联中介，所以分层架构在这里没有太多用武之地。

![训练和验证损失曲线](../../figures/fig0_loss_curves.png)
*图2：GeoNLA与FlatMLP在合成数据上的训练和验证损失曲线。GeoNLA的验证损失（右）整体低于FlatMLP，且收敛更稳定。*

![验证损失对比](../../figures/fig6_ablation_val_loss.png)
*图3：验证损失直接对比。GeoNLA在大部分训练过程中保持更低的验证损失。*

![RMSE、MAE、R² 柱状图对比](../../figures/fig7_ablation_metrics.png)
*图4：GeoNLA与FlatMLP在合成数据上的RMSE、MAE和R²比较。三个指标均显示GeoNLA的优势。*

![逐维度R² 对比](../../figures/fig8_ablation_r2_per_dim.png)
*图5：合成数据上逐维度R²对比。GeoNLA在耕地比例和灌溉比例上的优势最为显著。*

### 4.2 真实栅格数据实验

**表8：真实栅格数据上的整体验证指标（临夏盆地，~1 km网格）**

| 模型 | 参数量 | RMSE ↓ | MAE ↓ | R²（均值）↑ |
|------|-------|--------|-------|-----------|
| GeoNLA | 652 | 0.0342 | 0.0114 | 0.9120 |
| FlatMLP | 678 | 0.0351 | 0.0117 | 0.9107 |
| **改进** | — | **−2.7%** | **−3.1%** | **+0.001** |

改进方向一致，但幅度比合成实验小。这个结果并不令人意外。两个模型都达到了较高的R²值（≥ 0.91），说明在~1 km分辨率下，地理变量之间的关系本身就比较平滑，留给架构差异发挥的空间不大。即便如此，GeoNLA仍然在所有负向指标上保持领先。

**表9：真实栅格数据上的逐维度 R²**

| 输出维度 | GeoNLA | FlatMLP | Δ |
|---------|--------|---------|---|
| 建设用地 | 0.850 | 0.837 | +0.013 |
| 人口密度 | 0.955 | 0.958 | −0.003 |
| 夜间灯光 | 0.971 | 0.976 | −0.005 |
| 道路密度 | 0.872 | 0.872 | 0.000 |

GeoNLA在建设用地上的优势最大（+0.013），这再次印证了合成实验中的发现：当输出整合了多种物理地理约束时，分层架构的优势就体现出来了。人口密度和夜间灯光——与输入特征有更直接统计关系的变量——两个模型表现相当。道路密度完全相同，表明架构差异对这一输出的影响可以忽略不计。

![预测值 vs. 真实值散点图](../../figures/fig3_prediction_comparison.png)
*图6：真实栅格数据上所有输出维度的预测值vs.真实值散点图。两个模型的预测都较为接近对角线，GeoNLA在建设用地维度上的离散度略小。*

### 4.3 关键可视化分析

**涌现层的结构。** 一个让我们感到振奋的发现来自对32维涌现表示的t-SNE分析。GeoNLA的涌现层保留了清晰的地理相似性结构：来自相似地形-气候-土壤簇的网格单元在嵌入空间中位置接近，而来自不同地理环境的单元则良好分离。这个性质并非我们显式强制的——它是架构设计的自然产物。

![涌现层t-SNE可视化](../../figures/fig2_emergence_tsne.png)
*图7：32维涌现层的t-SNE可视化。不同颜色代表不同的地理环境类型，GeoNLA在嵌入空间中保留了清晰的地理相似性结构。*

**权重矩阵的可解释性。** 学习到的权重矩阵提供了观察模型内部运作的窗口。$W_1$（物理耦合，12 × 32）展示了与四个地理圈层对应的连贯块状结构，每个圈层对涌现表示贡献了独特的特征模式。$W_2$（人地耦合，32 × 6）则揭示了哪些涌现特征驱动哪些人类活动输出。

![权重热力图](../../figures/fig4_weight_heatmaps.png)
*图8：学习得到的权重热力图。上：$W_1$（物理耦合），展示与四个地理圈层对应的连贯块状结构。下：$W_2$（人地耦合），揭示涌现特征与人类活动输出的映射关系。*

**时间调制效应。** 时间调制可视化可能是最直观的验证。在固定地理条件下观察12个月的预测人类活动值，乘性调制产生了逼真的季节变化——农业输出在生长季达到峰值，建筑和人口相关输出呈现不同的时间模式。这验证了GeoNLA框架中乘性时间耦合机制的物理意义。

![时间调制效应](../../figures/fig5_temporal_modulation.png)
*图9：12个月的时间调制效应。固定地理条件下，乘性时间耦合驱动了具有物理合理性的季节变化模式。*

---

## 5 讨论

### 5.1 架构归纳偏置的力量

实验结果一致支持核心GeoNLA假设：具有乘性调制的分层架构在建模地理人地关系方面优于参数匹配的平坦MLP。

参数量匹配的设计（652 vs. 678）在这里至关重要。性能增益不能归因于模型容量更大，它反映的是GeoNLA架构所编码的**归纳偏置**——地理信息通过结构化级联流动、并伴随乘性圈层间耦合这一假设。换句话说，我们并没有给模型更多"智力"，而是给了它一个更合理的"思维方式"。

合成数据上的优势是显著的，真实数据上的优势虽然减弱但方向一致。这个落差是合理的——真实世界的标签由遥感产品派生而非由严格的两层乘性过程生成，噪声稀释了信号。

### 5.2 哪些变量受益于分层架构

逐维度R²分析揭示了一个清晰的模式：GeoNLA的优势集中在那些最直接受物理地理级联塑造的输出上。在合成数据中是耕地比例和灌溉比例，在真实数据中是建设用地。这些输出都依赖于地形、气候、土壤和植被的综合效应——恰恰是GeoNLA架构设计来捕捉的信息路径。

反过来，与输入特征有更直接或更简单关系的输出——夜间灯光、道路密度——在两个模型之间表现相当。这说明GeoNLA的优势不是普适的，而是集中在圈层叠加因果链起主导作用的输出上。这个发现本身就是有意义的：它告诉我们，分层架构的价值不在于"什么都更好"，而在于它正确编码了地理学中特定类型的因果关系。

### 5.3 涌现表示的地理意义

32维隐藏空间保留了有意义的地理相似性，这一点值得多说两句。在FlatMLP中，中间层只是一个"信息搅拌器"——14个维度的特征被线性混合后经过非线性激活，得到的32维向量没有明确的地理含义。而在GeoNLA中，中间层对应着从物理圈层到人类圈层的"涌现"过程，t-SNE可视化证实它确实保留了地理相似性。这不是一个被显式优化的目标，而是架构设计的自然副产品。

权重矩阵 $W_1$ 和 $W_2$ 的结构同样提供了可解释性。$W_1$ 显示了四个地理圈层各自的特征如何耦合到涌现表示中，$W_2$ 则展示了哪些涌现特征对哪些人类活动输出贡献最大。这使得GeoNLA不仅是一个预测工具，更是一个可以分析学习到的地理耦合路径的工具。

### 5.4 局限性与诚实评估

我们需要坦诚地面对几个局限：

**两层简化的天花板。** 完整的GeoNLA理论假设六个地理圈层，但当前实验只验证了物理层→人类层的两步。这就像用两个齿轮验证了一台钟表的原理——核心逻辑是对的，但真正的复杂性在于六个齿轮之间的协同。扩展到完整的六层架构可能揭示额外优势，也会引入优化和数据需求方面的挑战。

**标签噪声。** 真实栅格实验的输出标签由遥感分类和人口栅格产品派生，分类误差、空间配准偏差和时间不匹配都可能掩盖潜在的地理关系。

**空间自相关被忽略了。** 当前实现将每个网格单元独立处理，但地理现象的空间依赖结构是其本质特征。我们计划在后续版本中整合卷积层和GNN层。

**样本量的约束。** 合成数据400个样本，真实数据来自单一盆地的~1 km网格采样——以深度学习的标准来看，这些都是小样本实验。跨区域更大规模的研究将更有说服力。

**相关性不等于因果性。** 虽然GeoNLA架构编码了地理圈层的因果排序，但实验验证展示的仍然是相关性层面的改进。建立真正的因果有效性需要干预或准实验研究设计。

### 5.5 与现有工作的关系

GeoNLA填补了当前GeoAI研究中的若干缺口。NASA/IBM的Prithvi模型 [1] 证明了地理空间数据可以通过适当设计的Transformer架构产生跨域泛化的表征，但其架构由数据驱动的学习目标驱动，而非显式的地理原则。ESRI将深度学习整合进ArcGIS平台 [2]，将这些模型作为黑箱函数运行。GeoNLA通过提供理论基础来补充这些工作，阐明基础模型在应用于地理问题时应如何构建其内部结构。

在土地利用模拟方面，PLUS模型 [3] 通过逻辑回归整合生态和气候约束，但将各因素视为独立预测变量而非因果排序系统的各层。CNN与Google Earth Engine的结合已被用于努山塔拉城市扩张预测 [4]，但未显式编码地理分层。GeoNLA的原则——地理圈层自然地对应于不同的抽象层次——与这些经验发现是一致的。

在空间分析方面，GNNWR模型 [5] 和GTNNWR模型 [6] 证明了神经网络比参数化空间模型更有效地捕捉空间异质性，且空间关系的时间变化是地理过程的基本特征而非噪声。GeoNLA通过使层间耦合权重时变化来采纳这一原则。

---

## 6 结论与展望

本文提出了GeoNLA——一个将地理圈层叠加过程与神经网络层级计算之间的结构同构性形式化的框架。核心洞察很直接：深度神经网络中信息的逐层变换映射了地球地理圈层间交互的级联过程，而将这一类比显式化可以为设计结构上与地理过程对齐的神经网络架构提供原则性基础。

通过两层简化模型的实验验证，我们证明了GeoNLA在合成数据（RMSE −6.6%，MAE −13.3%）和临夏盆地真实栅格数据（RMSE −2.7%，MAE −3.1%）上均优于参数匹配的平坦MLP基线。两条数据路径上的改进方向一致，改进幅度反映了数据生成过程与GeoNLA分层乘性假设的对齐程度。除预测性能外，GeoNLA架构还提供了可解释的权重结构、保留了地理相似性的涌现表示、以及逼真的时间调制模式。

未来工作将聚焦三个方向：（1）将两层模型扩展为对应所有六个地理圈层的完整GeoNLA架构；（2）整合卷积层和图神经网络层以捕捉空间依赖结构；（3）在更大的跨区域数据集上检验地理-神经同构性的泛化能力。最终，我们期待这样一个未来：面向地理应用的神经网络架构不再是借用计算机视觉方法后临时适配，而是认真地将地理学理论作为架构灵感的来源。分层的地球，如同分层的网络，通过不断增加复杂性的连续阶段变换信息——通过使这一类比显式化，我们可以构建既从数据中学习、又尊重其试图表征的世界结构的模型。

完整的模型实现、数据处理脚本及实验复现代码已开源，托管于 GitHub（https://github.com/RuiMaandAmir/Geo-NLA）。

---

## 致谢

本研究使用了AI辅助工具（Coze）进行文献整理、语言润色和排版辅助。核心科学思路、理论框架、数学推导和实验设计均为作者原创。

实验数据来源于以下公开数据源：SRTM DEM [7]、WorldClim v2 [8]、SoilGrids v2.0 [9]、ESA WorldCover [10]、WorldPop、VIIRS夜间灯光 [11]、Sentinel-2 及 OpenStreetMap。

---

## 参考文献

[1] Jakubik, J. et al. (2024). Foundations of large-scale environmental analysis with multimodal AI. *Nature*, 639, 109–117.

[2] Esri (2026). ArcGIS platform integration with AI foundation models.

[3] Liang, X. et al. (2021). Simulating multiple future land use scenarios by incorporating ecological and climate constraints: the PLUS model. *Environmental Modelling & Software*, 146, 105230.

[4] Miru et al. (2025). Spatiotemporal CNN prediction of urban expansion in Nusantara.

[5] Du, Z. et al. (2020). Geographically neural network weighted regression for the analysis of spatial nonstationarity. *International Journal of Geographical Information Science*, 34(11), 1688–1714.

[6] Wu, S. et al. (2021). Geographically and temporally neural network weighted regression. *International Journal of Geographical Information Science*, 35(12), 2559–2584.

[7] NASA JPL (2013). Shuttle Radar Topography Mission (SRTM) Global. NASA EOSDIS Land Processes DAAC. https://doi.org/10.5067/MEPEWDXG2HSK

[8] Fick, S. E. & Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. *International Journal of Climatology*, 37(12), 4376–4390.

[9] Poggio, L. et al. (2021). SoilGrids 2.0: producing soil information for soil quantities with quantified spatial uncertainty. *SOIL*, 7(2), 999–1020.

[10] ESA WorldCover Consortium (2021). ESA WorldCover 10 m 2020 v1.0. Zenodo. https://doi.org/10.5281/zenodo.5571936

[11] Elvidge, C. D. et al. (2021). VIIRS DNB annual composites. *Earth Observation Group*, Colorado School of Mines.

[12] Ma, R. (2026). *Geo-NLA: Neural Network Layered Architecture as an Analogy for Geographic Systems*. Zenodo. https://doi.org/10.5281/zenodo.21350728
