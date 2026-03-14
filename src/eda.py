"""
eda.py - 탐색적 데이터 분석 (EDA) 시각화
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    ALL_FEATURES, COMPOSITION_FEATURES, FONT_FAMILY,
    OUTPUT_DIR, TARGET_PS, TARGET_UTS,
)


def run_eda(df: pd.DataFrame) -> None:
    """EDA 차트 생성 후 outputs/1_EDA.png 저장"""

    plt.rcParams['font.family']       = FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(
        '오스테나이트계 스테인리스강 - 데이터 탐색 분석 (EDA)',
        fontsize=16, fontweight='bold', y=0.98,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    temps = sorted(df['Temperature (K)'].dropna().unique())

    # (1) 온도별 항복강도 박스플롯
    ax1 = fig.add_subplot(gs[0, :2])
    ps_by_temp = [df[df['Temperature (K)'] == t][TARGET_PS].dropna().values for t in temps]
    ax1.boxplot(
        ps_by_temp, positions=range(len(temps)), widths=0.6, patch_artist=True,
        boxprops=dict(facecolor='#4C72B0', alpha=0.7),
        medianprops=dict(color='red', linewidth=2),
    )
    ax1.set_xticks(range(len(temps)))
    ax1.set_xticklabels([str(int(t)) for t in temps], rotation=45, fontsize=8)
    ax1.set_xlabel('Temperature (K)', fontsize=11)
    ax1.set_ylabel('0.2% Proof Stress (MPa)', fontsize=11)
    ax1.set_title('온도별 항복강도 분포', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # (2) 온도별 데이터 수
    ax2 = fig.add_subplot(gs[0, 2])
    counts = df['Temperature (K)'].value_counts().sort_index()
    ax2.bar([str(int(t)) for t in counts.index], counts.values, color='#55A868', alpha=0.8)
    ax2.set_xlabel('Temperature (K)', fontsize=10)
    ax2.set_ylabel('데이터 수', fontsize=10)
    ax2.set_title('온도별 데이터 수', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=90, labelsize=7)
    ax2.grid(axis='y', alpha=0.3)

    # (3) 항복강도 vs UTS 산점도
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(df[TARGET_PS], df[TARGET_UTS],
                     c=df['Temperature (K)'], cmap='coolwarm', alpha=0.4, s=10)
    plt.colorbar(sc, ax=ax3, label='Temp (K)')
    ax3.set_xlabel('0.2% Proof Stress (MPa)', fontsize=10)
    ax3.set_ylabel('UTS (MPa)', fontsize=10)
    ax3.set_title('항복강도 vs UTS', fontsize=12, fontweight='bold')

    # (4) 주요 원소 분포
    ax4 = fig.add_subplot(gs[1, 1])
    for elem, color in zip(['Cr', 'Ni', 'Mo'], ['#4C72B0', '#DD8452', '#55A868']):
        ax4.hist(df[elem].dropna(), bins=30, alpha=0.6, label=elem, color=color, density=True)
    ax4.set_xlabel('wt%', fontsize=10)
    ax4.set_ylabel('밀도', fontsize=10)
    ax4.set_title('주요 원소 함량 분포 (Cr, Ni, Mo)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # (5) 온도별 평균 강도 추이
    ax5 = fig.add_subplot(gs[1, 2])
    mean_ps  = df.groupby('Temperature (K)')[TARGET_PS].mean()
    mean_uts = df.groupby('Temperature (K)')[TARGET_UTS].mean()
    ax5.plot(mean_ps.index,  mean_ps.values,  'o-', color='#4C72B0', label='Proof Stress', linewidth=2)
    ax5.plot(mean_uts.index, mean_uts.values, 's-', color='#DD8452', label='UTS',          linewidth=2)
    ax5.set_xlabel('Temperature (K)', fontsize=10)
    ax5.set_ylabel('강도 (MPa)', fontsize=10)
    ax5.set_title('온도별 평균 강도 추이', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # (6) 상관관계 행렬
    ax6 = fig.add_subplot(gs[2, :])
    main_comp = ['Cr', 'Ni', 'Mo', 'Mn', 'Si', 'N', 'C', 'Cu', 'Temperature (K)', TARGET_PS, TARGET_UTS]
    corr = df[main_comp].corr()
    im = ax6.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax6, fraction=0.02, pad=0.04)
    ax6.set_xticks(range(len(main_comp)))
    ax6.set_yticks(range(len(main_comp)))
    ax6.set_xticklabels(main_comp, rotation=45, ha='right', fontsize=9)
    ax6.set_yticklabels(main_comp, fontsize=9)
    ax6.set_title('주요 변수 상관관계 행렬', fontsize=12, fontweight='bold')
    for i in range(len(main_comp)):
        for j in range(len(main_comp)):
            ax6.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=7,
                     color='white' if abs(corr.values[i, j]) > 0.5 else 'black')

    # (7) 결측치 현황
    ax7 = fig.add_subplot(gs[3, 0])
    missing = df[ALL_FEATURES + [TARGET_PS, TARGET_UTS]].isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    ax7.barh(missing.index, missing.values, color='#C44E52', alpha=0.8)
    ax7.set_xlabel('결측치 수', fontsize=10)
    ax7.set_title('피처별 결측치 현황', fontsize=12, fontweight='bold')
    ax7.grid(axis='x', alpha=0.3)

    # (8) Proof Stress 분포
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.hist(df[TARGET_PS].dropna(), bins=50, color='#4C72B0', alpha=0.8, edgecolor='white')
    ax8.axvline(df[TARGET_PS].mean(),   color='red',    linestyle='--', label=f"평균: {df[TARGET_PS].mean():.1f}")
    ax8.axvline(df[TARGET_PS].median(), color='orange', linestyle='--', label=f"중앙값: {df[TARGET_PS].median():.1f}")
    ax8.set_xlabel('0.2% Proof Stress (MPa)', fontsize=10)
    ax8.set_ylabel('빈도', fontsize=10)
    ax8.set_title('항복강도 분포', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(alpha=0.3)

    # (9) UTS 분포
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.hist(df[TARGET_UTS].dropna(), bins=50, color='#DD8452', alpha=0.8, edgecolor='white')
    ax9.axvline(df[TARGET_UTS].mean(),   color='red',  linestyle='--', label=f"평균: {df[TARGET_UTS].mean():.1f}")
    ax9.axvline(df[TARGET_UTS].median(), color='blue', linestyle='--', label=f"중앙값: {df[TARGET_UTS].median():.1f}")
    ax9.set_xlabel('UTS (MPa)', fontsize=10)
    ax9.set_ylabel('빈도', fontsize=10)
    ax9.set_title('UTS 분포', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(alpha=0.3)

    plt.savefig(f'{OUTPUT_DIR}/1_EDA.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("EDA 저장 완료 → 1_EDA.png")
