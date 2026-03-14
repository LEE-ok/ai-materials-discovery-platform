"""
analyzer.py - 탐색적 데이터 분석
ceramic notebook의 Analyzer 역할
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    ALL_FEATURES, COMPOSITION_FEATURES, FEATURE_DISPLAY_NAMES,
    FONT_FAMILY, OUTPUT_DIR, TARGET_PS, TARGET_UTS, TARGETS,
)


def _setup():
    plt.rcParams['font.family']        = FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False


class Analyzer:
    """
    ceramic notebook의 Analyzer와 동일한 인터페이스
    """

    def __init__(self, data_manager):
        self.dm = data_manager
        self.df = data_manager.df

    # ── 1. 컬럼별 기본 정보 ──────────────

    def plot_column(self) -> None:
        """컬럼별 통계 정보 출력 + 저장"""
        _setup()
        df = self.dm.df_model

        print("\n[컬럼별 기본 정보]")
        print(df[ALL_FEATURES + TARGETS].describe().round(3).to_string())

        # 결측치 현황 시각화
        fig, ax = plt.subplots(figsize=(10, 7))
        missing = self.df[ALL_FEATURES + TARGETS].isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)

        if len(missing) > 0:
            ax.barh(missing.index, missing.values, color='#C44E52', alpha=0.8)
            ax.set_xlabel('결측치 수', fontsize=11)
        else:
            ax.text(0.5, 0.5, '결측치 없음', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)

        ax.set_title('피처별 결측치 현황', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/1a_missing_values.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("결측치 현황 저장 완료 → 1a_missing_values.png")

    # ── 2. 바이올린 플롯 ─────────────────

    def violinplot_normalized_column(self) -> None:
        """정규화된 컬럼별 분포 (바이올린 플롯)"""
        _setup()
        df = self.dm.df_model[ALL_FEATURES + TARGETS].copy()

        # 정규화
        df_norm = (df - df.mean()) / (df.std() + 1e-8)

        # 상위 15개 컬럼만 (가독성)
        cols = ALL_FEATURES[:15] + TARGETS

        fig, ax = plt.subplots(figsize=(18, 6))
        data_list = [df_norm[c].dropna().values for c in cols]
        vp = ax.violinplot(data_list, positions=range(len(cols)),
                           showmedians=True, showextrema=True)

        for pc in vp['bodies']:
            pc.set_facecolor('#4C72B0')
            pc.set_alpha(0.6)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('정규화된 값', fontsize=11)
        ax.set_title('오스테나이트계 철강 - 정규화된 컬럼 분포 (바이올린 플롯)', fontsize=13, fontweight='bold')
        ax.axhline(0, color='red', linestyle='--', alpha=0.4)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/1b_violinplot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("바이올린 플롯 저장 완료 → 1b_violinplot.png")

    # ── 3. 상관관계 행렬 ─────────────────

    def plot_correlation_matrix(self) -> None:
        """변수 간 상관관계 행렬"""
        _setup()
        main_cols = ['Cr','Ni','Mo','Mn','Si','N','C','Cu',
                     'Temperature (K)', TARGET_PS, TARGET_UTS]
        corr = self.df[main_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

        ax.set_xticks(range(len(main_cols)))
        ax.set_yticks(range(len(main_cols)))
        ax.set_xticklabels(main_cols, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(main_cols, fontsize=10)
        ax.set_title('주요 변수 상관관계 행렬', fontsize=13, fontweight='bold')

        for i in range(len(main_cols)):
            for j in range(len(main_cols)):
                ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(corr.values[i,j]) > 0.5 else 'black')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/1c_correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("상관관계 행렬 저장 완료 → 1c_correlation_matrix.png")

    # ── 4. 히스토그램 ────────────────────

    def plot_histogram_column(self) -> None:
        """컬럼별 데이터 분포 히스토그램"""
        _setup()
        df = self.dm.df_model

        # 주요 피처 + 타겟
        plot_cols = ['Cr','Ni','Mo','Mn','Si','N','C','Cu',
                     'Temperature (K)', TARGET_PS, TARGET_UTS]
        n_cols = 4
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()
        fig.suptitle('오스테나이트계 철강 - 컬럼별 데이터 분포', fontsize=14, fontweight='bold')

        colors = ['#4C72B0','#DD8452','#55A868','#C44E52',
                  '#9467BD','#8C564B','#E377C2','#7F7F7F',
                  '#BCBD22','#17BECF','#AEC7E8']

        for i, col in enumerate(plot_cols):
            ax = axes[i]
            vals = df[col].dropna()
            ax.hist(vals, bins=30, color=colors[i % len(colors)], alpha=0.8, edgecolor='white')
            ax.axvline(vals.mean(),   color='red',    linestyle='--', linewidth=1.5,
                       label=f'평균: {vals.mean():.1f}')
            ax.axvline(vals.median(), color='orange', linestyle='--', linewidth=1.5,
                       label=f'중앙값: {vals.median():.1f}')
            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.set_ylabel('빈도', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        # 빈 subplot 숨기기
        for j in range(len(plot_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/1d_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("히스토그램 저장 완료 → 1d_histogram.png")

    # ── 5. 온도별 강도 분포 (도메인 특화) ─

    def plot_temp_distribution(self) -> None:
        """온도별 항복강도/UTS 분포 (오스테나이트 도메인 특화)"""
        _setup()
        df = self.df
        temps = sorted(df['Temperature (K)'].dropna().unique())

        fig = plt.figure(figsize=(20, 10))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
        fig.suptitle('오스테나이트계 철강 - 온도별 강도 분포', fontsize=14, fontweight='bold')

        # (1) 온도별 항복강도 박스플롯
        ax1 = fig.add_subplot(gs[0, :2])
        ps_by_temp = [df[df['Temperature (K)'] == t][TARGET_PS].dropna().values for t in temps]
        ax1.boxplot(ps_by_temp, positions=range(len(temps)), widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
        ax1.set_xticks(range(len(temps)))
        ax1.set_xticklabels([str(int(t)) for t in temps], rotation=45, fontsize=8)
        ax1.set_xlabel('Temperature (K)', fontsize=11)
        ax1.set_ylabel('0.2% Proof Stress (MPa)', fontsize=11)
        ax1.set_title('온도별 항복강도 분포', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # (2) 온도별 평균 강도 추이
        ax2 = fig.add_subplot(gs[0, 2])
        mean_ps  = df.groupby('Temperature (K)')[TARGET_PS].mean()
        mean_uts = df.groupby('Temperature (K)')[TARGET_UTS].mean()
        ax2.plot(mean_ps.index,  mean_ps.values,  'o-', color='#4C72B0', label='Proof Stress', linewidth=2)
        ax2.plot(mean_uts.index, mean_uts.values, 's-', color='#DD8452', label='UTS',          linewidth=2)
        ax2.set_xlabel('Temperature (K)', fontsize=10)
        ax2.set_ylabel('강도 (MPa)', fontsize=10)
        ax2.set_title('온도별 평균 강도 추이', fontsize=12, fontweight='bold')
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.savefig(f'{OUTPUT_DIR}/1e_temp_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("온도별 분포 저장 완료 → 1e_temp_distribution.png")
