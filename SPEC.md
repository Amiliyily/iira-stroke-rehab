# IIRA 项目技术规格书(SPEC v1.0)

> 论文题目:**Injury-inspired adaptive residual learning for robot-assisted upper-limb
> rehabilitation after stroke: a 6-DOF simulation benchmark study**
> (损伤启发的脑卒中上肢康复与六自由度机械臂仿真)
>
> 目标期刊格式:Nature Communications(Article, algorithm-development 风格)。
> 本文件是唯一权威技术依据:所有代码、实验、图、表、论文公式必须与此文件一致。

---

## 0. 运行环境

使用 Python 3.11 或更新版本；依赖见 requirements.txt。所有命令在仓库根目录执行。

## 1. 目录结构(在项目根 `./` 下创建)

```
stroke_rehab_iira/
  SPEC.md                 (本文件)
  README.md               (使用说明,最后写)
  src/
    arm6dof.py            (DH 参数、正运动学、雅可比、RNE 逆动力学、M(q)/C/g)
    injury.py             (损伤模型与虚拟受试者采样、FMA 合成)
    tasks.py              (最小 jerk 参考轨迹生成)
    simulate.py           (闭环仿真引擎:控制器 × 受试者 × 试次)
    dataset.py            (数据集生成与落盘)
    controllers/
      base.py             (Controller 基类:reset/step/save/load)
      classical.py        (基线 1–11)
      neural.py           (基线 12–15 + IIRA)
    run_experiment.py     (5 折 CV 主程序,落盘权重/日志/指标)
    make_figures.py       (图 1–6)
    make_tables.py        (表 1–4,booktabs 三线表 tex + csv)
  data/                   (dataset.npz, subjects.csv)
  results/
    weights/{model}_fold{k}.npz|.pt
    logs/{model}_fold{k}.csv        (每个 epoch 的训练损失等)
    trajectories/{model}_fold{k}.npz(测试试次轨迹,供画图)
    metrics_per_fold.csv
    metrics_summary.csv
  figures/   (fig1..fig6, PNG 300dpi + PDF)
  tables/    (table1..4 .tex + .csv)
  paper/
    main.tex  refs.bib  (pdflatex 可编译出 main.pdf)
```

## 2. 六自由度机械臂仿真(UR5 量级)

### 2.1 模型
标准 DH 参数(单位 m / rad,UR5 量级,做"仿真研究"声明,非真实 UR5 复刻):

| i | a_{i-1} | α_{i-1} | d_i | θ_i 偏置 |
|---|---------|---------|-----|----------|
| 1 | 0       | +π/2    | 0.0892 | 0 |
| 2 | -0.425  | 0       | 0      | 0 |
| 3 | -0.392  | 0       | 0      | 0 |
| 4 | 0       | +π/2    | 0.1093 | 0 |
| 5 | 0       | -π/2    | 0.0948 | 0 |
| 6 | 0       | 0       | 0.0823 | 0 |

连杆质量(kg):m = [3.70, 8.39, 2.28, 1.22, 1.22, 0.22];各杆质心取杆件几何中点
(沿杆长方向 1/2 处),转动惯量取细杆近似 I = (1/12) m L² 对角阵。
关节限位:θ ∈ ±2π(仿真不触发),速度限 ±3.0 rad/s,力矩限 τ ∈ ±[150,150,150,28,28,28] N·m。

### 2.2 动力学(论文 Methods 需推导)
刚体动力学标准形:
```
M(q)q̈ + C(q,q̇)q̇ + g(q) = τ_motor + τ_patient + τ_assist − F_v q̇ − F_c sign(q̇)
```
粘性摩擦 F_v = diag([0.5]*3 + [0.05]*3),库仑摩擦 F_c = diag([0.8]*3 + [0.1]*3)。
实现:递归 Newton–Euler 逆动力学 `rne(q,qd,qdd)`;
M(q) 用单位加速度法逐列求(M[:,i] = rne(q,0,e_i) − rne(q,0,0));
C(q,q̇)q̇ + g(q) = rne(q,qd,0)。积分:半隐式欧拉,dt = 0.01 s(100 Hz)。
`rne` 与理论值(动能法数值微分)做单元自检,误差 <1e-8,自检结果打印并存 `results/selfcheck.txt`。

### 2.3 任务
末端三维点到点够取(reaching):起点 q0 = [0, -π/2, π/2, -π/2, -π/2, 0] 对应末端位姿,
8 个方向(水平面内 45° 间隔,方位角 φ ∈ {0,45,…,315}°),5 档速度(距离 d ∈ {0.10, 0.14,
0.18, 0.22, 0.26} m),时长 T = 1.5–2.5 s(按 d/速度定,固定 2.0 s 基准),
最小 jerk 参考:
```
s(t) = 10(t/T)³ − 15(t/T)⁴ + 6(t/T)⁵
p_ref(t) = p0 + d·u·s(t)   (u 为方向单位向量)
```
关节参考由阻尼最小二乘逆运动学逐点生成(λ=1e-3),参考关节速度由雅可比伪逆差分得到。
目标位置均在工作空间内(先正解校验,不可达的方向缩短 d 直到可达并记录)。

## 3. 损伤模型(“injury-inspired” 核心,论文 Methods 推导)

患者残余随意力矩:
```
τ_patient = (1 − λ_w)·U·S·c(t)  −  λ_s·B_sp⊙tanh(q̇/0.5)⊙|q̇|  +  λ_n·ξ(t)
```
- λ_w ∈ [0, 0.85]:无力(weakness),整体力矩衰减;
- S ∈ R^{6×4}:健康协同矩阵(固定、行归一化的非负矩阵,论文给出具体数值);
  损伤通过协同混合噪声实现:c̃(t) = (1−λ_r)c(t) + λ_r·Πc(t),Π 为随机置换,
  λ_r ∈ [0, 0.6] 为协同结构瓦解度(synergy dissolution);
- λ_s ∈ [0, 3]:痉挛(spasticity),速度依赖性阻力矩,B_sp = [1,1.2,1.2,0.3,0.3,0.3];
- λ_n ∈ [0, 0.5]:共同收缩/低频发抖噪声,ξ(t) 为截止 3 Hz 的滤波白噪声(N·m)。
- c(t):患者"意图激活"。仿真中令 c(t) 由一阶肌电样滤波器驱动:
  c(t) 由期望轨迹的前馈需求经 τ=120ms 一阶滞后产生,幅值上限 1(饱和)。

损伤参数向量 θ_inj = (λ_w, λ_r, λ_s, λ_n)。
虚拟受试者 60 名:轻/中/重三层各 20 名:
- 轻度:λ_w~U(0.05,0.25), λ_r~U(0,0.15), λ_s~U(0,0.8), λ_n~U(0,0.10)
- 中度:λ_w~U(0.25,0.50), λ_r~U(0.15,0.35), λ_s~U(0.8,1.8), λ_n~U(0.10,0.25)
- 重度:λ_w~U(0.50,0.85), λ_r~U(0.35,0.60), λ_s~U(1.8,3.0), λ_n~U(0.25,0.50)
合成 Fugl–Meyer 上肢评分(0–66,论文要强调这是合成代理指标):
```
FMA = 66·(1 − 0.50λ_w − 0.20λ_r − 0.067λ_s − 0.20λ_n) + ε,  ε~N(0,1.5), 截断到 [4,66]
```
(λ_s 系数 0.067 使 λ_s=3 时贡献 −0.2;确保三层 FMA 大致落在 50–66 / 30–50 / 10–30。)
固定全局种子 seed=20261011;受试者参数存 `data/subjects.csv`。

## 4. 提出的算法:IIRA(Injury-Inspired Residual Assistance)

### 4.1 结构(图 1 原理图必须画这些模块)
输入滑动窗 W=50 步(0.5 s)特征 x_t = [q, q̇, e, ė, τ_a, τ_patient不可用→用 τ_total] ∈ R^{50×30}
(q,q̇,e,ė 各 6 维,τ_a 6 维,共 30 维; τ_patient 不可测,控制输入只用可测量)。
- (a) 时序编码器:1D 卷积(k=5, ch 32)+ MLP → 隐向量 h ∈ R^{64};
- (b) 协同瓶颈:z = W_z h ∈ R^4,激活 tanh —— 与损伤模型的 4 条协同对应(生物学约束);
- (c) 双头:θ̂_inj = MLP_θ(z) ∈ R^4(sigmoid×上界);ρ̂ = σ(w_ρᵀz) ∈ [0,1],
  ρ 为患者自主贡献率(patient contribution ratio);
- (d) 残余网络:f_ψ([q,q̇,e,ė,θ̂]) ∈ R^6,两层 128 隐层,tanh 输出 × 力矩限;
- (e) 按需辅助(AAN)增益调制:
```
K_p(t) = K_p^max·(1 − ρ̂_t),  K_d(t) = K_d^max·(1 − ρ̂_t)
K_p^max = [60,60,60,12,12,12] N·m/rad, K_d^max = [6,6,6,1.5,1.5,1.5]
τ_a = sat( K_p(t)e + K_d(t)ė + f_ψ(·) + ĝ(q) − τ̂_injury-bias,  τ_limits )
```
- (f) 安全层:速度阻尼(|q̇|>2.5 时附加 −2·(|q̇|−2.5)sign(q̇))、力矩饱和、工作空间球形约束
  (|p|>0.85 m 时向内的雅可比映射阻尼力)。

### 4.2 训练目标(论文给出完整推导)
监督信号来自仿真真值(θ_inj, ρ 真值定义为 ρ = ‖τ_patient 有用分量‖/(‖τ_patient‖+‖τ_a‖+ε)):
```
L = w1·E‖e‖² + w2·E‖jerk‖²_norm + w3·E‖τ_a‖² + w4·E[hinge(安全)] 
    + μ1·MSE(θ̂,θ) + μ2·BCE(ρ̂,ρ)
```
顶配权重(IIRA-full,论文声明为验证集网格搜索结果,网格也要真的跑并存 logs):
w = (1.0, 0.02, 0.05, 0.5), μ = (1.0, 0.5)。
消融变体(表 3):IIRA w/o 协同瓶颈;w/o AAN(固定增益);w/o 残余网络;w/o 安全层;
w/o 损伤估计头。每个变体都要真的训练、真的评测。

### 4.3 训练协议
训练数据:训练折 48 名受试者 × 每人 40 试次(8 方向 × 5 档)的闭环 rollout
(用增益调度 PD 做行为克隆式数据采集策略,注入探索噪声)。
Adam, lr=1e-3, batch 256, 60 epochs, early stop(patience 8, 验证=训练受试者中留出 10 名)。
**所有 fold × 模型 × epoch 的损失逐行写 logs CSV;每 fold 保存最终权重文件。**

## 5. 十五个基线(全部"默认/文献常用配置",论文要逐一引用出处)

| # | 基线 | 默认配置 |
|---|------|----------|
| 1 | PID | K_p=[40,40,40,8,8,8], K_i=0.5·K_p·0.1, K_d=[4,4,4,1,1,1] |
| 2 | PD + 重力补偿 | 同 PID 去积分 + ĝ(q) |
| 3 | 计算力矩控制 (Computed Torque) | 标称模型,K_p,K_d 同上 |
| 4 | 阻抗控制 (Impedance, Hogan 1985) | M_d=I, K_d 阻尼=2√K_p, 刚度=K_p |
| 5 | 导纳控制 (Admittance) | 外环导纳 M_a=2I, B_a=10, K_a=50 + 内环 PD |
| 6 | 滑模控制 (SMC) | 切换增益 η=15, 边界层 0.05 |
| 7 | LQR | 对 q0 处线性化,Q=diag(10·I₆, I₆), R=0.1·I₆ |
| 8 | 线性 MPC | 预测域 10,Q/R 同 LQR,无约束解析解 |
| 9 | 扰动观测器 + PD (DOB) | Q 滤波器带宽 20 rad/s |
| 10 | 迭代学习控制 (ILC) | 重复试次间 P 型更新,学习率 0.3,测试试次从第 1 次冷启动记录逐次曲线 |
| 11 | RBF 网络自适应控制 (Slotine) | 25 个高斯基/关节,宽度 1.0,自适应率 5 |
| 12 | MLP 残余 | [64,64],tanh,sklearn 默认 Adam 参数 |
| 13 | LSTM 残余 | hidden 64,1 层 |
| 14 | GRU 残余 | hidden 64,1 层 |
| 15 | Transformer-lite 残余 | d=64, 单头注意力, 2 层, 窗 50 |

12–15 与 IIRA 用相同训练数据与训练协议(仅结构与默认超参不同),保证公平。

## 6. 交叉验证与评测协议

- 受试者级 5 折 CV:60 人按层分层抽样切 5 折(每折 12 人:轻 4/中 4/重 4),
  fold 划分固定存 `data/folds.csv`。
- 每折:神经类模型(12–15、IIRA 及消融)在 48 名训练受试者上训练;
  经典控制器(1–11)无需训练但需按相同协议评测;ILC 在每名测试受试者每方向上
  连续做 5 次重复试次,记录全部 5 次(论文报告第 5 次成绩并画学习曲线)。
- 测试协议(所有控制器完全一致):每名测试受试者 8 方向 × 中速档(d=0.18 m, T=2.0 s)
  共 8 试次。
- 指标(7 项,逐试次计算后按受试者→折→模型汇总):
  1. 末端位置 RMSE (mm)
  2. 关节角 RMSE (deg)
  3. 归一化 jerk(无量纲,jerk² 积分 × T⁵/d²)
  4. 平均辅助力矩 RMS (N·m)
  5. 安全违规率 %(超速、超力矩、出工作空间的事件占比)
  6. 稳定误差(最后 0.3 s 平均 |e|)(mm)
  7. 单步计算耗时 (ms,同一进程计时中位数)
  另:IIRA 及消融报告损伤参数估计 MAE 与 ρ̂ 的 Pearson r(表 3 附加列)。
- 统计:配对比较 IIRA vs 每个基线,Wilcoxon 符号秩(scipy.stats.wilcoxon,
  以"受试者×折"为单元),Holm 校正;效应量 Cliff's δ。显著性标 *。
- **落盘要求(用户硬性要求"保留所有运算过程和权重")**:
  - `results/weights/` 每个模型×fold 的完整参数(npz 或 pt;经典控制器存其增益/滤波器参数);
  - `results/logs/` 神经模型逐 epoch 训练/验证损失 CSV;网格搜索过程 CSV;
  - `results/trajectories/` 每模型每 fold 全部测试试次轨迹 npz(q, qd, p, p_ref, τ_a, t);
  - `results/metrics_per_fold.csv`(长表:model, fold, subject, trial, 7 指标);
  - `results/metrics_summary.csv`(model × 指标 mean±std);
  - 全局随机种子、git 式配置快照写 `results/config_snapshot.json`。

## 7. 图(全部 matplotlib,PNG 300 dpi + PDF 矢量;字号 ≥7 pt;色盲友好调色板)

- fig1_principle:IIRA 系统原理框图(患者-损伤模型→时序编码器→协同瓶颈→双头→
  AAN 增益→残余网络→机械臂→安全层 闭环箭头),matplotlib patches 绘制。
- fig2_engineering:工程图三联:(a) 机械臂侧视工程简图(杆长尺寸标注、关节编号 J1–J6、
  DH 坐标系);(b) 三维可达工作空间点云投影;(c) DH 参数表图注。
- fig3_data:(a) 轻/中/重三例受试者同一方向的末端轨迹 vs 参考;(b) 对应 τ_patient 与 τ_a
  曲线;(c) 60 名受试者 FMA 分布直方图(按层着色)。
- fig4_main:(a) 16 控制器 7 指标分组条形图(5 折 mean±std,显著性星号);
  (b) 雷达图(归一化,6 轴);(c) RMSE–辅助力矩 Pareto 散点(越左上越好)。
- fig5_training:(a) 神经模型训练/验证损失曲线;(b) IIRA 损伤参数估计散点
  (θ̂ vs θ, 4 子图, 标 R²);(c) 消融条形图。
- fig6_stratified:轻/中/重三层 × 4 模型(PID、阻抗、LSTM、IIRA)的 RMSE/辅助力矩热图或分组条。

## 8. 三线表(booktabs,只画 \toprule \midrule \bottomrule)

- table1_benchmark:行业现状对标(研究/期刊年份/机器人/DOF/控制策略/样本/主要结局)。
  真实文献,从 `your-local-literature-directory/` 下的 PDF(如 04_PREP2_2017.pdf、
  05_RATULS_2019.pdf、07_StrokeRehab_NeurIPS_2022.pdf、08_ATLAS_v2_2022.pdf、
  10_COBRA_2024.pdf、11_fNIRS_Robot_Prediction_2024.pdf 等)提取准确题录,再补
  Krebs/MIT-Manus、Lo et al. JAMA 2019、Hogan 1985、Fugl-Meyer 1975 等经典文献。
- table2_main:16 控制器 × 7 指标(mean±std),最优加粗,显著性上标。
- table3_ablation:IIRA-full vs 5 消融 × 指标 + θ̂ MAE、ρ̂ r。
- table4_config:仿真/训练超参数总表。

## 9. 论文(Nature Communications 格式)

- `paper/main.tex`:NC Article 风格(pdflatex 可编译;若无 nature.cls 则用 article
  定制:无页眉双栏?NC 正文单栏;标题页含 Title/Authors/Affiliations/Corresponding;
  Abstract ≤150 词;正文 Introduction(无标题?NC 的 Intro 无 section 标题,用粗体首句)——
  按 NC 惯例:Abstract → Introduction(无编号)→ Results(带子标题)→ Discussion →
  Methods(后置)→ Data availability → Code availability → References(编号,
  superscript 引用)→ Acknowledgements / Author contributions / Competing interests。
- Introduction 要求(用户硬性要求"对标顶刊、言之有物"):卒中流行病学(WHO/GBD 数据)、
  上肢功能障碍比例、机器人辅助康复证据链(引用 RATULS Lancet Neurol 2019 阴性结果、
  Lo et al. JAMA 2019、Krebs 系列、PREP2 预测、COBRA 2024 等 benchmark),
  **每条现状痛点 → 明确列出 → 对应本工作的具体改进**(用一段 "Here we ..." 总结三个创新点:
  ①损伤启发的可解释协同瓶颈估计;②按需辅助与安全层的统一可微框架;③16 控制器
  5 折受试者级公开基准)。
- Methods 必须含完整数学推导:RNE 动力学、最小 jerk、损伤模型四参数、IIRA 架构与损失、
  AAN 稳定性讨论(简要 Lyapunov 论证:τ_a 有界+PD 增益正定→误差一致最终有界)、
  统计方法。
- Results 数字必须来自 `results/metrics_summary.csv` 真实数据,禁止编造;
  引用 fig1–6、table1–4。
- 生物学/临床意义(用户硬性要求):Discussion 中论述协同瓶颈与临床协同分析
  (Cheung et al.)的对应、AAN 与主动参与/神经可塑性(运动再学习、Hebbian 可塑性)的关系、
  FMA 代理指标的局限、仿真到真实(transfer gap)的局限与伦理声明(纯仿真研究,无人体数据)。
- `paper/refs.bib`:≥35 条真实文献,优先从 `前沿和benchmark/` 的 PDF 首页提取题录,
  其余用教科书级准确信息(作者、题名、期刊、年、卷页、DOI 若可知)。

## 10. 验收标准(DoD)

1. `run_experiment.py` 一键跑通(允许先跑 reduced 快速版自检,再全量);
2. results/ 下权重、日志、轨迹、指标齐全;
3. figures/ 六张图(PNG+PDF)、tables/ 四张三线表(tex+csv);
4. `paper/main.pdf` 编译成功,正文 4500–6000 词量级,数字与 results 一致;
5. README 写明一键复现命令。

## 附录 A:v1.1 changelog(被动性保持修复,2026-10-11)

v1.0 全量实验发现 IIRA-full 闭环抖振(归一化 jerk ~1e10,违规率 ~12%),机制:AAN 按
(1−ρ̂) 同时削减 K_d 阻尼 + 残余网络学到的痉挛/库仑补偿构成正速度反馈 + 羽量级腕关节
(~1e-4 kg·m²)→ 阻尼裕度失守。v1.1 对 §4.1 控制律做以下 4 处修订(v1.0 条文保留备查;
归档结果见 results_archive_v1_naive/):

1. **AAN 只调制刚度**(修订 §4.1e):K_p(t) = K_p^min + (K_p^max − K_p^min)(1 − ρ̂),
   K_p^min = 0.2·K_p^max;K_d 恒为 K_d^max 不再下调。动机:足额阻尼是闭环被动性的
   充分条件(外力做功被 K_d 耗散),刚度下调保留"按需辅助"语义;属 AAN 文献标准做法。
2. **残余网络输出低通**(修订 §4.1d→力矩叠加):τ_f ← τ_f + dt/(τ_c+dt)·(f_ψ(·) − τ_f),
   τ_c = 0.05 s,力矩叠加用滤波后 τ_f。动机:限制残余通道带宽,消除窗口延迟反馈的
   高频极限环。
3. **θ̂ 痉挛补偿改为耗散安全形式**(修订 §4.1e 的 τ̂_injury-bias 项):
   τ_comp = −κ·λ̂_s·B_sp ⊙ tanh(q̇_f/0.5) ⊙ |q̇_f|,κ=0.8,q̇_f 为 20 Hz 一阶低通后的
   关节速度;该项符号恒为耗散,即使 θ̂ 过估也不可能形成正反馈;不使用未滤波 q̇。
   其雅可比按引擎既有政策并入隐式阻尼矩阵。
4. **安全层速度阻尼改用滤波速度 q̇_f 计算**(其余不变:阈值 2.5 rad/s、增益 2、
   工作空间球约束不变)。

消融语义随之更新:noaan = 固定 K_p^max(不随 ρ̂ 下调);notheta = 无损伤估计头、
因而不施加 τ_comp;noresidual = 无残余网络(也无残余低通);nobottleneck / nosafety
含义不变。动力学、损伤模型、15 个基线、5 折协议、训练协议、落盘格式全部不变。

## 附录 B:v2.0 闭环患者模型(2026-10-12，复核修订)

动机:v1 患者模型为纯前馈(τ_vol = (1−λ_w)·U·S·c(t),c 由 τ_ff 前馈驱动),不随机器人
辅助改变策略——无主动纠偏、无偷懒(slacking)、无疲劳、无跨试次学习。transfer_v2 的
sim-to-real 诊断表明该系统性错位是迁移失败主因之一。v2.0 将患者升级为闭环自适应
控制器(实现于 src/patient_closed_loop.py;v1 模型保持不变,主实验结果不受影响):

    τ_patient = (1 − λ_w·F(t))·U·S·c_eff(t) − λ_s·B_sp⊙tanh(q̇/0.5)⊙|q̇| + λ_n·ξ(t)
    c_eff(t) = m(t)·[(1−λ_r)c(t) + λ_r·Πc(t)]
    c(t) = sat(LPF_120ms(US⁺·[τ_ff(t) + K_h⊙e(t−d_h) + K_hv⊙ė(t−d_h)]))

1. **误差反馈驱动**:患者以自身反馈增益 K_h(初值 [8,8,8,2,2,2] N·m/rad,约为机器人
   K_p^max 的 0.13 倍——弱于机器人)，显式反馈延迟 d_h=120 ms（100 Hz 下 12 步），
   再经 120 ms EMG 惯性低通对跟踪误差纠偏；延迟前的历史误差置零。
   K_hv = 0.1·K_h。反馈与经同一协同伪逆 US⁺ 映射到 4 维激活空间,保持损伤模型的
   协同结构。
2. **偷懒/节约效应**(Marchal-Crespo & Reinkensmeyer 2009):努力乘子 m(t)∈[0.3,1],
   ṁ = −β·(ŝ − s*),ŝ 为动态辅助占比 ‖τ_a,prev−g_prev‖/(‖τ_a,prev−g_prev‖+‖τ_vol‖)
   的 2 s EMA,s*=0.5，零总力矩时取 s*；重力支撑不计代劳。
   β=0.15 s⁻¹;辅助越多患者越省力。session 间 m 向 1 恢复 50%。
3. **疲劳**:Ḟ = k_f·m·‖c‖/√4 (k_f=0.03)，随实际激活努力累积，试次内单调上升；
   试次间恢复 50%，session 间恢复 90%；F ∈ [1,1.15]，使最重无力 λ_w≤0.85
   仍有 1−λ_wF≥0.0225，不出现随意力矩符号反转。
4. **跨试次运动学习**(Hebbian 式):每试次结束 K_h ← K_h·(1 + α·P_trial),
   P_trial=[RMS(τ_vol)/(RMS(τ_vol)+RMS(τ_a−g)+ε)]·mean(m‖c‖/√4)，
   同时反映患者动态发力占比与实际激活量，避免把被动痉挛、噪声、静态支撑当主动学习。
   α=0.02，K_h 上限 1.5×初值；每个 trial 后立即更新并记录更新后的 FMA。
   FMA 代理随之变化:FMA_eff = clip(FMA_base + 20·(K̄_h/K̄_h0 − 1), 4, 66)。

慢状态(m,F,K_h,辅助占比 EMA)跨试次/session 携带；快状态(c,ξ,延迟队列)每试次重置。
通过引擎 set_feedback 注入误差及上一时刻动态辅助（显式耦合，避免代数环）；v1 Subject
接口不变。在线记录 pre-integration g 与去噪 τ_patient，直接计算与原零噪重放等价的
ρ_realized/ρ_dynamic，避免旧实现 post-integration 速度/重力导致的时序错位。

参数为行为假设的合理量级，未经临床拟合；不预设方向或线性倍增关系。β、α、k_f
分别改变节约、学习、疲劳的速度，反馈延迟及增益影响闭环稳定性。另做三层代表患者
的 0.5×/2×单参数敏感性（β、α、k_f，控制器 SMC/IIRA/IIRA-tuned），只做探索性描述，
不充当 60 人主检验。所有实际参数、来源、权重/代码 SHA256 与运行时间随 run 落盘。

实验协议(与主实验受试者级 5 折一致):控制器子集 {pid, impedance, smc, mlp, lstm,
iira(v1.1 律), iira_noaan, iira_tuned(调参最优律)};神经模型直接沿用各 fold 已保存
权重(患者模型变更造成分布偏移,为零训练迁移研究点,不重训);每名测试受试者
3 session × 8 试次(8 方向,d=0.18 m)。核心假设:高增益控制器(SMC)下患者显著
偷懒(ρ_dynamic 逐试次下降);IIRA/AAN 下参与度维持或上升、FMA 代理改善更快。

采用同一 subject/session/direction 噪声种子跨控制器配对；60 人×8 控制器×24 trial=
11,520 行。慢状态在每个 subject/controller 间独立初始化。noaan 沿用主实验保存的
iira_noaan fold 权重；iira-tuned 固定既有部署律最优参数，不在闭环测试患者上重新择优。

统计单位为受试者（n=60），不是 trial：slacking 主指标为同方向配对的 session 1→3
随意力矩相对下降幅度（正值=偷懒）；辅以方向中心化的 trial 斜率、m 下降和 rho 斜率。
方向中心化使八个方向任务难度不会伪造 trial 趋势；另给 session 内趋势作为探索性结果。
FMA 主学习量为 session-3 完成后的代理减去 baseline（不是 3×拟合斜率）。各控制器
slacking/rho/学习分别双侧 Wilcoxon 对零检验，各族 Holm；预先对 IIRA/IIRA-tuned
vs SMC、IIRA vs noaan、tuned vs IIRA 的 RMSE/rho/slack/ΔFMA/viol/jerk 做配对检验并 Holm。
同时给配对秩二列效应量、受试者 bootstrap 95% CI；所有符号和不显著结果照实记录。
FMA 改善由模型规定的学习机制产生，只是合成代理，不构成临床疗效证据。

结果使用 results/coadapt/run_20261012_v2_1/（旧 results/coadapt 根目录实验保留）。
每控制器仅保存一条同一中度患者/session-3/direction-0代表轨迹；不保存其他全轨迹。
图 fig8 五面板、表 table6 三线表，CSV→汇总→TeX 按显示精度 allclose 校验。
结构/数值与文献依据：Marchal-Crespo & Reinkensmeyer (2009)，
https://doi.org/10.1186/1743-0003-6-20；Reinkensmeyer et al. (2007)，
https://pubmed.ncbi.nlm.nih.gov/18002881/ 。仅采用节约机制动机，文献不标定本模型参数。

## 附录 C：外部大规模运动参考泛化（2026-10-12，临床评分放宽）

用户允许采用更大的运动数据集而放宽临床评分。使用 CMU 官方 bulk ASF/AMC ZIP，
仅从归档分段取成员，不抓取各动作页面。官方当前目录称 2,605 条动作，实际 2007 年
bulk ZIP 的中央目录为 2,514 AMC、112 capture blocks；同一人可能对应多个编号，
因此编号不等于独立患者，不能将它们当作卒中受试者。

磁盘选择规则先于效果评测固定：全部压缩大小≤100,000 字节的 AMC，以及每个 capture
block 最小的一条，共 480 个候选（约 43.6 MB）；逐成员 CRC32 校验。
缺失右臂关节、非角度单位、非有限值或少于 80 帧为数据质量排除，完整记录原因。
AMC rhumerus 的三个旋转、rradius/rwrist/rhand 的第一个旋转映射到六维输入；
取中央至多 241 帧，归一化到 2 秒/201 帧（名义源帧率120Hz，短记录时间拉伸），
角度先连续化（unwrap）并经 3 Hz 二阶 Butterworth 双向平滑；以起始角为零点，固定缩放
[0.2,0.2,0.2,0.1,0.1,0.1]，截断到 ±[0.6,0.6,0.6,0.3,0.3,0.3] rad，叠加 q_home。
这是预声明的形态兼容重定向，不是人体关节到 UR5 的物理等价重建。
由同一 arm6dof 算 p_ref、q̇_ref、q̈_ref、逆动力学前馈。
非直线动作 jerk 使用距起点的最大末端位移（最小 0.01 m）归一化，与原点到点任务
的幅值定义存在差异，因此不跨数据集比较其绝对 jerk。

保留原 60 虚拟患者、原 5 折、8 控制器；神经权重与部署律冻结。将全部有效外部动作
以固定种子洗牌映射到 60×8 个参考槽位，若少于480则循环补齐；只做1 session×8动作，
额外评测 3,840 试次（不会替换附录 B 的 3 session/11,520 试次实验）。
同一 virtual subject/slot 对控制器使用相同噪声，患者 slow state 在8试次内保留。
受试者级统计单位仍为60名虚拟患者，参考编号和capture block完整记录，实际有效数照实报告。
新结果写入 results/external_motion_cmu/run_20261012_1/，不保存任何仿真全轨迹；
仅保留原coadapt每控制器1条代表轨迹。

工程通过条件预声明为平均末端 RMSE≤50 mm 且平均 viol≤1%，同时报告 rho_dynamic、
随意发力与 jerk；这只是运动参考下的仿真可运行性条件，没有临床验证含义。
IIRA-tuned vs IIRA 的六指标配对 Wilcoxon 用单族 Holm，报告受试者均值与 SD。
严禁将本实验称为真人控制验证、卒中临床标签泛化、未知 FMA 预测或康复疗效成功。
官方来源：https://mocap.cs.cmu.edu/ 与 https://mocap.cs.cmu.edu/faqs.php 。
