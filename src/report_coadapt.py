"""Validate completed CV, compare frozen-weight shift, and write the report."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import dataset as ds
from coadapt_analysis import test_row

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/coadapt/run_20261012_v2_1'
MODELS=['pid','impedance','smc','mlp','lstm','iira','iira_noaan','iira_tuned']

def main():
    d=pd.read_csv(RUN/'trials_long.csv')
    cfg=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
    assert cfg['complete'] and len(d)==11520
    keys=['model','fold','subject','session','direction']
    assert not d.duplicated(keys).any()
    assert d.groupby(['model','subject']).size().eq(24).all()
    assert d.groupby(['model','fold']).size().eq(288).all()
    assert set(d.model)==set(MODELS)
    _,fold_of=ds.load_subjects(str(ROOT))
    assert all(f==fold_of[s]+1 for f,s in zip(d.fold,d.subject))
    assert d.groupby(['fold','subject','session','direction']).seed.nunique().eq(1).all()
    nums=d.select_dtypes('number')
    assert np.isfinite(nums.to_numpy()).all()
    assert d.m_end.between(.3,1).all() and d.F_end.between(1,1.15).all()
    assert d.rho_dynamic.between(0,1).all()
    assert (d.F_end>=d.F_start-1e-12).all()
    for _,g in d.groupby(['model','subject']):
        g=g.sort_values('trial_global')
        assert g.iloc[0].m_start==1 and g.iloc[0].F_start==1
        assert g.iloc[0].kh_mean_start==5
        assert np.allclose(g.kh_mean_start.to_numpy()[1:],g.kh_mean_after_learning.to_numpy()[:-1])
        nextF=1+(g.F_end.to_numpy()[:-1]-1)*.5
        boundaries=g.trial_in_session.to_numpy()[1:]==1
        nextF[boundaries]=1+(nextF[boundaries]-1)*.1
        assert np.allclose(nextF,g.F_start.to_numpy()[1:],rtol=0,atol=1e-12)
        nextm=g.m_end.to_numpy()[:-1].copy()
        nextm[boundaries]=1-(1-nextm[boundaries])*.5
        assert np.allclose(nextm,g.m_start.to_numpy()[1:],rtol=0,atol=1e-12)
        assert (np.diff(g.fma_after_learning)>=-1e-12).all()
    assert len(list(RUN.glob('traj_*.npz')))==8
    core=['src/patient_closed_loop.py','src/simulate.py','src/run_coadapt.py',
          'src/run_coadapt_v2.py','src/coadapt_analysis.py',
          'src/controllers/neural.py','src/controllers/classical.py','src/arm6dof.py']
    for p in core:
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==cfg['sha256'][str(Path(p))]
    for path,expected in cfg['sha256'].items():
        if path.startswith('results\\weights'):
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected
    validation=dict(rows=len(d),subjects=60,folds=5,controllers=8,
                    complete_unique_24_trials_per_subject_controller=True,
                    fold_membership_matches_saved_data=True,paired_noise_seeds=True,
                    all_numeric_finite=True,slow_state_continuity_and_bounds=True,
                    frozen_weight_hashes_unchanged=True,solver_source_hashes_unchanged=True,
                    representative_trajectory_count=8)
    (RUN/'validation_experiment.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    # Matching session 1 to old feedforward patient: same subject/ref/noise seed.
    old=pd.read_csv(ROOT/'results/metrics_per_fold_v2.csv')
    tuned=pd.read_csv(ROOT/'results/tuning/iira_tuned_metrics_per_fold.csv')
    old=pd.concat([old,tuned],ignore_index=True)
    old=old[(old.model.isin(MODELS))&(old.rep==0)].rename(columns={'trial':'direction'})
    merged=d[d.session==1].merge(old,on=['model','fold','subject','direction'],suffixes=('_closed','_ff'),validate='one_to_one')
    shift=[]
    for (m,f,s),g in merged.groupby(['model','fold','subject']):
        shift.append(dict(model=m,fold=f,subject=s,
                          rmse_feedforward=g.rmse_p_mm_ff.mean(),rmse_closed=g.rmse_p_mm_closed.mean(),
                          rmse_difference=(g.rmse_p_mm_closed-g.rmse_p_mm_ff).mean(),
                          viol_feedforward=g.viol_pct_ff.mean(),viol_closed=g.viol_pct_closed.mean()))
    shift=pd.DataFrame(shift);shift.to_csv(RUN/'shift_per_subject.csv',index=False)
    shift.groupby('model').mean(numeric_only=True).to_csv(RUN/'shift_summary.csv')
    per=pd.read_csv(RUN/'per_subject.csv');S=pd.read_csv(RUN/'summary.csv').set_index('model')
    stats=pd.read_csv(RUN/'stats.csv')
    def result(fam,m):return stats[(stats.family==fam)&(stats.model_a==m)].iloc[0]
    lines=['# IIRA v2.0 闭环患者实验报告','',
           '完成 60 名虚拟受试者、原受试者级 5 折、8 控制器、3 session × 8 方向，共 11,520 试次。',
           '权重全部冻结；同一 subject/session/direction 跨控制器使用相同噪声。统计单位为 60 名受试者。',
           '', '结论：SMC 发力下降 7.30%，rho_dynamic 方向校正斜率 −0.000747/试次（p_Holm=0.000778），支持节约假设。',
           'IIRA 同样发力下降 3.41%，rho 显著下降（p_Holm=0.00499），参与度维持或上升假设不成立。',
           'IIRA 相对 SMC 发力下降较少（p=0.0170），ΔFMA 快 0.143 点（p=0.0305）；',
           'IIRA-tuned 相对 SMC 学习快 0.090 点但不显著（p=0.184）。所有 p 均为对应族 Holm 校正。',
           'tuned 将 IIRA RMSE 186.58→123.31 mm（−33.9%）、viol 3.90→0.254%，',
           '但 rho 0.478→0.470，ΔFMA 2.715→2.662，jerk 上升；不是所有目标上的改进。',
           'IIRA vs noaan 的 rho、slacking、学习差异均未通过多重校正；noaan RMSE 更低。',
           '', '## 主假设与逐控制器结果','',
           'slacking 指数为同方向 session 1→3 随意力矩下降百分比，正值表示发力减少。',
           'rho 斜率先去除方向固定效应；学习量为三次 session 完成后 FMA 合成代理减 baseline。',
           '各对零检验族独立 Holm 校正；所有检验均双侧。','',
           '|控制器|RMSE (mm)|rho_dynamic|发力下降 % (p_Holm)|rho斜率/试次 (p_Holm)|ΔFMA代理 (p_Holm)|viol %|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:
        r=S.loc[m];sl=result('slacking_vs0',m);rh=result('rho_vs0',m);le=result('learning_vs0',m)
        lines.append(f'|{m}|{r.rmse_p_mm_mean:.2f}|{r.rho_dynamic_mean:.4f}|'
                     f'{r.slacking_pct_mean:+.3f} ({sl.p_holm:.3g})|'
                     f'{r.rho_slope_mean:+.3g} ({rh.p_holm:.3g})|'
                     f'{r.delta_fma_mean:+.3f} ({le.p_holm:.3g})|{r.viol_pct_mean:.3f}|')
    lines+=['','## 配对检验','',
            '差值为 A−B；RMSE/slacking/viol/jerk 越小越好，rho/ΔFMA 越大越好。',
            '24 个配对检验统一 Holm 校正，95% CI 为受试者配对差值的 5,000 次 bootstrap。','',
            '|A vs B|指标|差值 [95% CI]|p_Holm|秩二列效应量|','|---|---|---:|---:|---:|']
    for _,r in stats[stats.family=='paired'].iterrows():
        lines.append(f'|{r.model_a} vs {r.model_b}|{r.metric}|{r.mean_difference:.4g} '
                     f'[{r.ci_low:.4g}, {r.ci_high:.4g}]|{r.p_holm:.3g}|{r.rank_biserial:.3f}|')
    lines+=['','## 参数敏感性与研究边界','',
            '敏感性只用 fold-1 三层各一名代表患者，SMC/IIRA/IIRA-tuned，每项 24 试次；不做总体显著性推断。',
            'SMC rho 斜率在 β=0.5× 为 +0.000113，而 β=2× 为 −0.000292，方向随节约速率改变。',
            'α=0.5×/2× 使代表患者 ΔFMA 约 1.34–1.37 / 5.45–5.81；量级依赖模型标定。',
            '三名代表患者中 IIRA slacking 接近零且扫描中可变号；60 人主实验却显著下降，提示异质性。',
            '',
            '- FMA 改善由正学习率的更新式产生，不能将显著增长解读为独立证实康复疗效。',
            '- rho_dynamic 沿用总去噪患者力矩，含痉挛阻力；学习仅用主动随意力矩与激活量。',
            '- 原测试距离 0.18 m、固定方向顺序；session 内未去方向混杂的斜率仅探索性。',
            '- IIRA-tuned 沿用先前在 fold-1 测试患者上选出的部署律，保留既有选择偏差，不称严格独立泛化。',
            '- noaan 使用原主实验独立训练的消融权重，比较同时包含训练消融和部署律差异。',
            '- 无临床参数拟合、无真实患者验证；可选快速微调未执行。',
            '- 引擎 cosh 溢出警告来自摩擦 Jacobian 在大速度时的极限计算；全部落盘数值有限，违规照实保留。',
            '', '## 与 SPEC 的差异及修订','',
            '实现遵循本轮先修订的附录 B。相对旧草稿：显式 120 ms 延迟；F_MAX=1.15 防力矩反向；',
            '动态辅助扣除重力；学习同时计主动发力占比与激活努力；保留辅助 EMA 慢记忆；',
            '使用 pre-integration 在线去噪力矩；FMA 在每次学习后记录；表中的 ΔFMA 为实际差值。',
            'slacking 改为同方向跨 session 发力下降，并同时保存方向校正斜率，减少固定任务顺序混杂。',
            '约定之外仅代表患者敏感性、未做可选微调。原根目录结果保留作为未修正的历史草稿。',
            '图稿通过 view_image 目检和 Poppler PDF 渲染检查；当前环境未提供 ReadMediaFile，工具名称有偏差。',
            '', '## 产物与耗时','',
            f'主实验运行 {cfg["runtime_min"]:.2f} min（4 进程、每进程 torch 1 线程）；敏感性 3.68 min（与主实验重叠）。',
            '逐试次、逐受试者、逐折、逐 session、统计、参数、权重/代码 SHA256 均已落盘。',
            'results/coadapt/run_20261012_v2_1/ 是本轮权威结果，根目录旧结果未覆盖。',
            'figures/fig8_coadapt.png/.pdf 与 figures/fig8a_patient_flow.png/.pdf；tables/table6_coadapt.csv/.tex。',
            '每控制器 1 条同一中度患者代表轨迹，其余轨迹仅在内存中用于计算指标。',
            '', '## 机制依据','',
            '[Marchal-Crespo & Reinkensmeyer 2009](https://doi.org/10.1186/1743-0003-6-20)；',
            '[Reinkensmeyer, Wolbrecht & Bobrow 2007](https://pubmed.ncbi.nlm.nih.gov/18002881/)。',
            '文献支持节约与负荷共享的建模动机，不提供本实验的数值标定。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
