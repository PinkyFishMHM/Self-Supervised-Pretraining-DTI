import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


# 1. 绘制Figure 3 - Few-shot学习性能曲线
def plot_fewshot_performance():
    """绘制Few-shot学习性能图（对应论文Figure 3）"""

    # 数据来自论文Table V
    data_percentages = [10, 25, 50, 100]

    # BindingDB数据
    bindingdb_baseline = [0.736, 0.781, 0.834, 0.896]
    bindingdb_enhanced = [0.834, 0.867, 0.902, 0.934]

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制曲线
    ax.plot(data_percentages, bindingdb_baseline, 'o-', color='#1f77b4',
            linewidth=2.5, markersize=8, label='Original DLM-DTI')
    ax.plot(data_percentages, bindingdb_enhanced, 's-', color='#ff7f0e',
            linewidth=2.5, markersize=8, label='Enhanced DLM-DTI')

    # 添加数据标签
    for i, (x, y1, y2) in enumerate(zip(data_percentages, bindingdb_baseline, bindingdb_enhanced)):
        # 基线模型标签
        retention1 = y1 / bindingdb_baseline[-1] * 100
        ax.annotate(f'{y1:.3f}\n({retention1:.1f}%)',
                    xy=(x, y1), xytext=(0, -20),
                    textcoords='offset points', ha='center', fontsize=9)

        # 增强模型标签
        retention2 = y2 / bindingdb_enhanced[-1] * 100
        ax.annotate(f'{y2:.3f}\n({retention2:.1f}%)',
                    xy=(x, y2), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)

    # 设置图形属性
    ax.set_xlabel('Training Data Percentage', fontsize=14)
    ax.set_ylabel('AUROC Score', fontsize=14)
    ax.set_title('Few-shot Learning Performance on BindingDB Dataset', fontsize=16, fontweight='bold')
    ax.set_ylim(0.70, 0.95)
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 添加性能提升文本框
    textstr = 'Performance Improvements:\n10%: +13.3%\n25%: +11.1%\n50%: +8.2%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('figure3_fewshot_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


# 2. 绘制Figure 4 - 冷启动场景性能对比
def plot_coldstart_performance():
    """绘制冷启动场景性能图（对应论文Figure 4）"""

    scenarios = ['Cold Drug', 'Cold Target', 'Cold Drug-Target']
    datasets = ['BindingDB', 'DAVIS', 'BIOSNAP']

    # 数据来自论文
    baseline_data = {
        'Cold Drug': [0.783, 0.800, 0.796],
        'Cold Target': [0.736, 0.760, 0.758],
        'Cold Drug-Target': [0.698, 0.726, 0.718]
    }

    enhanced_data = {
        'Cold Drug': [0.847, 0.863, 0.851],
        'Cold Target': [0.798, 0.821, 0.813],
        'Cold Drug-Target': [0.762, 0.789, 0.771]
    }

    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    x = np.arange(len(datasets))
    width = 0.35

    for idx, (ax, scenario) in enumerate(zip(axes, scenarios)):
        baseline_values = baseline_data[scenario]
        enhanced_values = enhanced_data[scenario]

        # 绘制柱状图
        bars1 = ax.bar(x - width / 2, baseline_values, width,
                       label='Baseline DLM-DTI', color='#4c72b0', alpha=0.8)
        bars2 = ax.bar(x + width / 2, enhanced_values, width,
                       label='Enhanced DLM-DTI', color='#dd8452', alpha=0.8)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 添加改进百分比
        for i, (baseline, enhanced) in enumerate(zip(baseline_values, enhanced_values)):
            improvement = (enhanced - baseline) * 100
            ax.text(i, enhanced + 0.01, f'+{improvement:.1f}%',
                    ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')

        # 设置子图属性
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('AUROC Score', fontsize=12)
        ax.set_title(f'{scenario}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0.65, 0.9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Cold-start Prediction Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure4_coldstart_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


# 3. 绘制Table IV - 整体性能对比表格的可视化
def plot_overall_performance():
    """绘制整体性能对比图"""

    # 数据来自论文Table IV
    methods = ['Random\nForest', 'SVM', 'DLM-DTI', 'DTI-LM', 'SGCL-DTI', 'MFCADTI', 'Enhanced\nDLM-DTI']

    # AUROC scores
    bindingdb_auroc = [0.764, 0.772, 0.896, 0.912, 0.919, 0.923, 0.934]
    davis_auroc = [0.781, 0.789, 0.908, 0.925, 0.931, 0.935, 0.942]
    biosnap_auroc = [0.798, 0.805, 0.898, 0.915, 0.921, 0.924, 0.928]

    # VRAM requirements
    vram = [2.1, 1.8, 7.7, 12.4, 15.2, 18.6, 7.9]

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：AUROC性能对比
    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax1.bar(x - width, bindingdb_auroc, width, label='BindingDB', color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x, davis_auroc, width, label='DAVIS', color='#ff7f0e', alpha=0.8)
    bars3 = ax1.bar(x + width, biosnap_auroc, width, label='BIOSNAP', color='#2ca02c', alpha=0.8)

    ax1.set_xlabel('Methods', fontsize=12)
    ax1.set_ylabel('AUROC Score', fontsize=12)
    ax1.set_title('AUROC Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=0)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0.7, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')

    # 突出显示最佳方法
    ax1.axvspan(x[-1] - 0.4, x[-1] + 0.4, alpha=0.2, color='green')

    # 右图：内存效率对比
    colors = ['#d62728' if v > 10 else '#2ca02c' if v < 8 else '#ff7f0e' for v in vram]
    bars = ax2.bar(methods, vram, color=colors, alpha=0.8)

    # 添加数值标签
    for bar, v in zip(bars, vram):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                 f'{v:.1f}GB', ha='center', va='bottom', fontsize=10)

    ax2.set_xlabel('Methods', fontsize=12)
    ax2.set_ylabel('VRAM Requirement (GB)', fontsize=12)
    ax2.set_title('Memory Efficiency Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 20)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加参考线
    ax2.axhline(y=8, color='red', linestyle='--', alpha=0.5, label='8GB VRAM Limit')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('overall_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# 4. 绘制训练过程曲线
def plot_training_curves():
    """绘制训练过程曲线"""

    # 加载训练日志
    df = pd.read_csv('training_logs.csv')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 预训练损失曲线
    ax1 = axes[0, 0]
    pretrain_df = df[df['phase'] == 'pre-training']
    ax1.plot(pretrain_df['epoch'], pretrain_df['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.plot(pretrain_df['epoch'], pretrain_df['hint_loss'], 'g--', alpha=0.7, label='Hint Loss')
    ax1.plot(pretrain_df['epoch'], pretrain_df['contrastive_loss'], 'r--', alpha=0.7, label='Contrastive Loss')
    ax1.plot(pretrain_df['epoch'], pretrain_df['reconstruction_loss'], 'c--', alpha=0.7, label='Reconstruction Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Pre-training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 微调性能曲线
    ax2 = axes[0, 1]
    finetune_df = df[df['phase'] == 'fine-tuning']
    ax2.plot(finetune_df['epoch'], finetune_df['baseline_auroc'], 'b-', linewidth=2, label='Baseline AUROC')
    ax2.plot(finetune_df['epoch'], finetune_df['enhanced_auroc'], 'r-', linewidth=2, label='Enhanced AUROC')
    ax2.fill_between(finetune_df['epoch'],
                     finetune_df['baseline_auroc'],
                     finetune_df['enhanced_auroc'],
                     alpha=0.3, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AUROC Score', fontsize=12)
    ax2.set_title('Fine-tuning Performance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # GPU内存使用
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['gpu_memory_gb'], 'g-', linewidth=2)
    ax3.axhline(y=8.0, color='red', linestyle='--', linewidth=2, label='8GB Limit')
    ax3.fill_between(df['epoch'], 0, df['gpu_memory_gb'], alpha=0.3, color='green')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax3.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 10)

    # 训练时间
    ax4 = axes[1, 1]
    pretrain_time = pretrain_df['time_minutes'].values
    finetune_time = finetune_df['time_minutes'].values

    ax4.bar(range(len(pretrain_time)), pretrain_time, color='blue', alpha=0.7, label='Pre-training')
    ax4.bar(range(len(pretrain_time), len(pretrain_time) + len(finetune_time)),
            finetune_time, color='red', alpha=0.7, label='Fine-tuning')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Time per Epoch (minutes)', fontsize=12)
    ax4.set_title('Training Time Analysis', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Enhanced DLM-DTI Training Process Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_process_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# 5. 绘制消融实验结果
def plot_ablation_study():
    """绘制消融实验结果（对应论文Table VI）"""

    components = ['Full Model', 'w/o Cross-modal\nPre-training', 'w/o Contrastive\nLearning',
                  'w/o Adaptive\nSampling', 'w/o Memory\nOptimization']

    # BindingDB数据
    bindingdb_auroc = [0.934, 0.906, 0.921, 0.922, 0.932]
    bindingdb_auprc = [0.891, 0.863, 0.860, 0.876, 0.888]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(components))
    width = 0.35

    bars1 = ax.bar(x - width / 2, bindingdb_auroc, width, label='AUROC', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width / 2, bindingdb_auprc, width, label='AUPRC', color='#ff7f0e', alpha=0.8)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # 添加性能下降指示
    for i in range(1, len(components)):
        auroc_drop = (bindingdb_auroc[0] - bindingdb_auroc[i]) * 100
        auprc_drop = (bindingdb_auprc[0] - bindingdb_auprc[i]) * 100

        ax.annotate(f'AUROC: -{auroc_drop:.1f}%\nAUPRC: -{auprc_drop:.1f}%',
                    xy=(i, max(bindingdb_auroc[i], bindingdb_auprc[i]) + 0.02),
                    ha='center', fontsize=9, color='red')

    ax.set_xlabel('Model Configuration', fontsize=14)
    ax.set_ylabel('Performance Score', fontsize=14)
    ax.set_title('Ablation Study Results on BindingDB', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=0)
    ax.legend(loc='lower left', fontsize=12)
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # 突出显示完整模型
    ax.axvspan(-0.5, 0.5, alpha=0.2, color='green')

    plt.tight_layout()
    plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# 6. 生成汇总统计报告
def generate_summary_report():
    """生成汇总统计报告"""

    print("\n" + "=" * 60)
    print("Enhanced DLM-DTI 性能汇总报告")
    print("=" * 60)

    # 整体性能提升
    print("\n1. 整体性能提升（相比基线DLM-DTI）:")
    print("   - BindingDB: AUROC +4.2%, AUPRC +6.8%")
    print("   - DAVIS: AUROC +3.8%, AUPRC +5.4%")
    print("   - BIOSNAP: AUROC +3.2%, AUPRC +4.7%")

    # Few-shot学习
    print("\n2. Few-shot学习性能:")
    print("   - 10%数据: 保持89.3%性能（基线82.1%）")
    print("   - 25%数据: 保持92.8%性能（基线87.2%）")
    print("   - 50%数据: 保持96.6%性能（基线93.1%）")

    # 冷启动场景
    print("\n3. 冷启动场景改进:")
    print("   - Cold Drug: 平均提升7.6%")
    print("   - Cold Target: 平均提升7.9%")
    print("   - Cold Drug-Target: 平均提升8.4%")

    # 计算效率
    print("\n4. 计算效率:")
    print("   - VRAM需求: 7.9GB（仅增加0.2GB）")
    print("   - 训练时间: 增加13%")
    print("   - 推理速度: 2.3ms/对")

    # 关键创新贡献
    print("\n5. 关键组件贡献:")
    print("   - 跨模态预训练: AUROC提升2.8%")
    print("   - 对比学习: AUPRC提升3.1%")
    print("   - 自适应采样: 性能提升1.2-1.8%")
    print("   - 内存优化: 保持性能同时减少38%内存")

    print("\n" + "=" * 60)


# 主函数
if __name__ == "__main__":
    print("开始生成Enhanced DLM-DTI可视化图表...\n")

    # 生成所有图表
    print("1. 生成Few-shot学习性能图...")
    plot_fewshot_performance()

    print("\n2. 生成冷启动场景性能图...")
    plot_coldstart_performance()

    print("\n3. 生成整体性能对比图...")
    plot_overall_performance()

    print("\n4. 生成训练过程分析图...")
    plot_training_curves()

    print("\n5. 生成消融实验结果图...")
    plot_ablation_study()

    print("\n6. 生成汇总报告...")
    generate_summary_report()

    print("\n所有可视化图表已生成完成！")