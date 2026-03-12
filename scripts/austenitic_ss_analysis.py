"""
오스테나이트계 철강 고온 강도 예측 AI 모델
데이터: STMECH_AUS_SS.xls
목표변수: 0.2% 항복강도 (proof stress), UTS
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib, os

rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────
# 1. 데이터 로드 및 전처리
# ─────────────────────────────────────────

df = pd.read_excel('../data/STMECH_AUS_SS.xls', sheet_name='alldata', header=5)

# 'Na' 문자열 → NaN 변환
df.replace('Na', np.nan, inplace=True)

# 숫자형 변환
numeric_cols = ['Solution_treatment_temperature', 'Solution_treatment_time(s)',
                'Grains mm-2', 'Elongation (%)', 'Area_reduction (%)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 합금 성분 피처 (21개 원소)
composition_features = ['Cr','Ni','Mo','Mn','Si','Nb','Ti','Zr','Ta','V','W',
                        'Cu','N','C','B','P','S','Co','Al','Sn','Pb']

# 공정 피처
process_features = ['Solution_treatment_temperature', 'Temperature (K)',
                    'Product form', 'Type of melting']

# Type of melting: 숫자만 사용 (Na → 0)
df['Type of melting'] = pd.to_numeric(df['Type of melting'], errors='coerce').fillna(0)

all_features = composition_features + process_features

TARGET_PS = '0.2%proof_stress (M Pa)'
TARGET_UTS = 'UTS (M Pa)'

# 결측치 처리: 필수 피처와 타겟만 사용
df_model = df[all_features + [TARGET_PS, TARGET_UTS]].copy()
df_model['Solution_treatment_temperature'] = df_model['Solution_treatment_temperature'].fillna(
    df_model['Solution_treatment_temperature'].median())
df_model = df_model.dropna(subset=[TARGET_PS, TARGET_UTS])

print(f"최종 모델링 데이터: {df_model.shape[0]}행 x {df_model.shape[1]}열")

X_raw = df_model[all_features].values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X_raw)
y_ps = df_model[TARGET_PS].values
y_uts = df_model[TARGET_UTS].values

# ─────────────────────────────────────────
# 2. 데이터 분석 시각화
# ─────────────────────────────────────────

fig = plt.figure(figsize=(20, 24))
fig.suptitle('오스테나이트계 스테인리스강 - 데이터 탐색 분석 (EDA)', 
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# (1) 온도별 항복강도 분포
ax1 = fig.add_subplot(gs[0, :2])
temps = sorted(df['Temperature (K)'].dropna().unique())
ps_by_temp = [df[df['Temperature (K)']==t][TARGET_PS].dropna().values for t in temps]
bp = ax1.boxplot(ps_by_temp, positions=range(len(temps)), widths=0.6,
                 patch_artist=True, 
                 boxprops=dict(facecolor='#4C72B0', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2))
ax1.set_xticks(range(len(temps)))
ax1.set_xticklabels([str(int(t)) for t in temps], rotation=45, fontsize=8)
ax1.set_xlabel('Temperature (K)', fontsize=11)
ax1.set_ylabel('0.2% Proof Stress (MPa)', fontsize=11)
ax1.set_title('온도별 항복강도 분포', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# (2) 데이터 수 히스토그램 (온도별)
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

# (4) 주요 원소 분포 (Cr, Ni, Mo)
ax4 = fig.add_subplot(gs[1, 1])
for elem, color in zip(['Cr','Ni','Mo'], ['#4C72B0','#DD8452','#55A868']):
    vals = df[elem].dropna()
    ax4.hist(vals, bins=30, alpha=0.6, label=elem, color=color, density=True)
ax4.set_xlabel('wt%', fontsize=10)
ax4.set_ylabel('밀도', fontsize=10)
ax4.set_title('주요 원소 함량 분포 (Cr, Ni, Mo)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# (5) 온도별 평균 강도 추이
ax5 = fig.add_subplot(gs[1, 2])
mean_ps = df.groupby('Temperature (K)')[TARGET_PS].mean()
mean_uts = df.groupby('Temperature (K)')[TARGET_UTS].mean()
ax5.plot(mean_ps.index, mean_ps.values, 'o-', color='#4C72B0', label='Proof Stress', linewidth=2)
ax5.plot(mean_uts.index, mean_uts.values, 's-', color='#DD8452', label='UTS', linewidth=2)
ax5.set_xlabel('Temperature (K)', fontsize=10)
ax5.set_ylabel('강도 (MPa)', fontsize=10)
ax5.set_title('온도별 평균 강도 추이', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# (6) 성분 간 상관관계
ax6 = fig.add_subplot(gs[2, :])
main_comp = ['Cr','Ni','Mo','Mn','Si','N','C','Cu','Temperature (K)', TARGET_PS, TARGET_UTS]
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
        ax6.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if abs(corr.values[i,j]) > 0.5 else 'black')

# (7) 결측치 현황
ax7 = fig.add_subplot(gs[3, 0])
missing = df[all_features + [TARGET_PS, TARGET_UTS]].isnull().sum()
missing = missing[missing > 0].sort_values(ascending=True)
ax7.barh(missing.index, missing.values, color='#C44E52', alpha=0.8)
ax7.set_xlabel('결측치 수', fontsize=10)
ax7.set_title('피처별 결측치 현황', fontsize=12, fontweight='bold')
ax7.grid(axis='x', alpha=0.3)

# (8) Proof Stress 분포
ax8 = fig.add_subplot(gs[3, 1])
ax8.hist(df[TARGET_PS].dropna(), bins=50, color='#4C72B0', alpha=0.8, edgecolor='white')
ax8.axvline(df[TARGET_PS].mean(), color='red', linestyle='--', label=f"평균: {df[TARGET_PS].mean():.1f}")
ax8.axvline(df[TARGET_PS].median(), color='orange', linestyle='--', label=f"중앙값: {df[TARGET_PS].median():.1f}")
ax8.set_xlabel('0.2% Proof Stress (MPa)', fontsize=10)
ax8.set_ylabel('빈도', fontsize=10)
ax8.set_title('항복강도 분포', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(alpha=0.3)

# (9) UTS 분포
ax9 = fig.add_subplot(gs[3, 2])
ax9.hist(df[TARGET_UTS].dropna(), bins=50, color='#DD8452', alpha=0.8, edgecolor='white')
ax9.axvline(df[TARGET_UTS].mean(), color='red', linestyle='--', label=f"평균: {df[TARGET_UTS].mean():.1f}")
ax9.axvline(df[TARGET_UTS].median(), color='blue', linestyle='--', label=f"중앙값: {df[TARGET_UTS].median():.1f}")
ax9.set_xlabel('UTS (MPa)', fontsize=10)
ax9.set_ylabel('빈도', fontsize=10)
ax9.set_title('UTS 분포', fontsize=12, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(alpha=0.3)

plt.savefig('../outputs/1_EDA.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA 저장 완료")

# ─────────────────────────────────────────
# 3. 모델 학습 (Random Forest, GBM, MLP)
# ─────────────────────────────────────────

X_train, X_test, y_ps_train, y_ps_test, y_uts_train, y_uts_test = train_test_split(
    X, y_ps, y_uts, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, 
                                            min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gradient Boosting': HistGradientBoostingRegressor(max_iter=200, max_depth=5,
                                                    learning_rate=0.05, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                                   max_iter=500, random_state=42, early_stopping=True,
                                   validation_fraction=0.1)
}

results = {}
trained_models = {}

print("\n=== 모델 학습 및 평가 ===")
for name, model in models.items():
    use_scaled = (name == 'Neural Network')
    Xtr = X_train_s if use_scaled else X_train
    Xte = X_test_s if use_scaled else X_test
    
    # Proof Stress
    m_ps = model.__class__(**model.get_params())
    m_ps.fit(Xtr, y_ps_train)
    pred_ps = m_ps.predict(Xte)
    
    # UTS
    m_uts = model.__class__(**model.get_params())
    m_uts.fit(Xtr, y_uts_train)
    pred_uts = m_uts.predict(Xte)
    
    results[name] = {
        'ps_mae': mean_absolute_error(y_ps_test, pred_ps),
        'ps_rmse': np.sqrt(mean_squared_error(y_ps_test, pred_ps)),
        'ps_r2': r2_score(y_ps_test, pred_ps),
        'uts_mae': mean_absolute_error(y_uts_test, pred_uts),
        'uts_rmse': np.sqrt(mean_squared_error(y_uts_test, pred_uts)),
        'uts_r2': r2_score(y_uts_test, pred_uts),
        'pred_ps': pred_ps,
        'pred_uts': pred_uts,
    }
    trained_models[name] = {'ps': m_ps, 'uts': m_uts}
    
    print(f"\n[{name}]")
    print(f"  Proof Stress → MAE: {results[name]['ps_mae']:.2f} MPa, "
          f"RMSE: {results[name]['ps_rmse']:.2f} MPa, R²: {results[name]['ps_r2']:.4f}")
    print(f"  UTS          → MAE: {results[name]['uts_mae']:.2f} MPa, "
          f"RMSE: {results[name]['uts_rmse']:.2f} MPa, R²: {results[name]['uts_r2']:.4f}")

# ─────────────────────────────────────────
# 4. 모델 성능 비교 시각화
# ─────────────────────────────────────────

fig, axes = plt.subplots(3, 4, figsize=(20, 16))
fig.suptitle('모델별 예측 성능 비교', fontsize=16, fontweight='bold')

colors = {'Random Forest': '#4C72B0', 'Gradient Boosting': '#DD8452', 'Neural Network': '#55A868'}

for row, (name, res) in enumerate(results.items()):
    color = colors[name]
    
    # Proof Stress 예측 vs 실제
    ax = axes[row, 0]
    lim_min = min(y_ps_test.min(), res['pred_ps'].min()) - 5
    lim_max = max(y_ps_test.max(), res['pred_ps'].max()) + 5
    ax.scatter(y_ps_test, res['pred_ps'], alpha=0.3, s=8, color=color)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5)
    ax.set_xlabel('실제 (MPa)', fontsize=9)
    ax.set_ylabel('예측 (MPa)', fontsize=9)
    ax.set_title(f'{name}\nProof Stress (R²={res["ps_r2"]:.3f})', fontsize=10, fontweight='bold')
    ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
    ax.grid(alpha=0.3)
    
    # UTS 예측 vs 실제
    ax = axes[row, 1]
    lim_min = min(y_uts_test.min(), res['pred_uts'].min()) - 5
    lim_max = max(y_uts_test.max(), res['pred_uts'].max()) + 5
    ax.scatter(y_uts_test, res['pred_uts'], alpha=0.3, s=8, color=color)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5)
    ax.set_xlabel('실제 (MPa)', fontsize=9)
    ax.set_ylabel('예측 (MPa)', fontsize=9)
    ax.set_title(f'{name}\nUTS (R²={res["uts_r2"]:.3f})', fontsize=10, fontweight='bold')
    ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
    ax.grid(alpha=0.3)
    
    # Proof Stress 잔차
    ax = axes[row, 2]
    residuals_ps = y_ps_test - res['pred_ps']
    ax.scatter(res['pred_ps'], residuals_ps, alpha=0.3, s=8, color=color)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('예측값 (MPa)', fontsize=9)
    ax.set_ylabel('잔차 (MPa)', fontsize=9)
    ax.set_title(f'{name}\nProof Stress 잔차', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # UTS 잔차
    ax = axes[row, 3]
    residuals_uts = y_uts_test - res['pred_uts']
    ax.scatter(res['pred_uts'], residuals_uts, alpha=0.3, s=8, color=color)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('예측값 (MPa)', fontsize=9)
    ax.set_ylabel('잔차 (MPa)', fontsize=9)
    ax.set_title(f'{name}\nUTS 잔차', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/2_model_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("모델 성능 시각화 저장 완료")

# ─────────────────────────────────────────
# 5. 성능 요약 테이블 + 지표 비교 차트
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('모델 성능 지표 비교', fontsize=14, fontweight='bold')

model_names = list(results.keys())
short_names = ['RF', 'GBM', 'MLP']
bar_colors = [colors[n] for n in model_names]

metrics = [('MAE (MPa)', 'ps_mae', 'uts_mae'), 
           ('RMSE (MPa)', 'ps_rmse', 'uts_rmse'),
           ('R²', 'ps_r2', 'uts_r2')]

for ax, (title, ps_key, uts_key) in zip(axes, metrics):
    ps_vals = [results[n][ps_key] for n in model_names]
    uts_vals = [results[n][uts_key] for n in model_names]
    x = np.arange(len(model_names))
    w = 0.35
    bars1 = ax.bar(x - w/2, ps_vals, w, label='Proof Stress', color=[c+'CC' for c in bar_colors], 
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w/2, uts_vals, w, label='UTS', color=bar_colors, 
                   edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*bar.get_height(),
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*bar.get_height(),
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('../outputs/3_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("지표 비교 차트 저장 완료")

# ─────────────────────────────────────────
# 6. 피처 중요도 (최고 성능 모델 - Random Forest)
# ─────────────────────────────────────────

feature_names_display = composition_features + ['용체화 온도', '시험 온도(K)', '제품 형태', '용해 방법']

best_model_ps = trained_models['Random Forest']['ps']
best_model_uts = trained_models['Random Forest']['uts']

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Random Forest 피처 중요도', fontsize=14, fontweight='bold')

for ax, model, title in zip(axes, [best_model_ps, best_model_uts], 
                             ['항복강도 (Proof Stress)', 'UTS']):
    imp = model.feature_importances_
    sorted_idx = np.argsort(imp)
    colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp)))
    
    ax.barh(np.array(feature_names_display)[sorted_idx], imp[sorted_idx],
            color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('중요도 (Feature Importance)', fontsize=11)
    ax.set_title(f'{title} 예측\n피처 중요도', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 상위 5개 강조
    top5_idx = sorted_idx[-5:]
    for idx in top5_idx:
        bar_val = imp[idx]
        pos = list(sorted_idx).index(idx)
        ax.get_children()[pos].set_edgecolor('red')
        ax.get_children()[pos].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('../outputs/4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("피처 중요도 저장 완료")

# ─────────────────────────────────────────
# 7. 온도별 예측 성능 분석
# ─────────────────────────────────────────

df_test = df_model.iloc[X_test.shape[0]*0:].copy()  # use direct index
test_idx = df_model.index[len(df_model) - len(X_test):]
df_test_sub = df_model.loc[df_model.index[int(len(df_model)*0.8):]]

# Recompute test set predictions with RF
pred_ps_rf = trained_models['Random Forest']['ps'].predict(X_test)
pred_uts_rf = trained_models['Random Forest']['uts'].predict(X_test)

# Get test temperatures
X_test_df = pd.DataFrame(X_test, columns=all_features)
test_temps = X_test_df['Temperature (K)'].values

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Random Forest - 온도별 예측 오차 분석', fontsize=14, fontweight='bold')

for ax, pred, actual, title in zip(axes, 
                                    [pred_ps_rf, pred_uts_rf],
                                    [y_ps_test, y_uts_test],
                                    ['항복강도 (Proof Stress)', 'UTS']):
    mae_by_temp = {}
    for t in sorted(np.unique(test_temps)):
        mask = test_temps == t
        if mask.sum() >= 3:
            mae_by_temp[t] = mean_absolute_error(actual[mask], pred[mask])
    
    temps_sorted = sorted(mae_by_temp.keys())
    maes = [mae_by_temp[t] for t in temps_sorted]
    
    bars = ax.bar(range(len(temps_sorted)), maes, color='#4C72B0', alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(temps_sorted)))
    ax.set_xticklabels([str(int(t)) for t in temps_sorted], rotation=45, fontsize=8)
    ax.set_xlabel('Temperature (K)', fontsize=11)
    ax.set_ylabel('MAE (MPa)', fontsize=11)
    ax.set_title(f'{title}\n온도별 MAE', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 평균 MAE 선
    avg_mae = np.mean(maes)
    ax.axhline(avg_mae, color='red', linestyle='--', linewidth=1.5, label=f'평균 MAE: {avg_mae:.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig('../outputs/5_temp_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("온도별 오차 분석 저장 완료")

# ─────────────────────────────────────────
# 8. 최종 결과 요약 출력
# ─────────────────────────────────────────

print("\n" + "="*60)
print("           최종 성능 요약 (테스트 세트)")
print("="*60)
print(f"{'모델':<22} {'PS_MAE':>8} {'PS_R²':>8} {'UTS_MAE':>9} {'UTS_R²':>8}")
print("-"*60)
for name in model_names:
    r = results[name]
    print(f"{name:<22} {r['ps_mae']:>7.2f}  {r['ps_r2']:>7.4f}  {r['uts_mae']:>8.2f}  {r['uts_r2']:>7.4f}")
print("="*60)

# 최우수 모델 저장
best_name = max(results, key=lambda x: results[x]['ps_r2'])
print(f"\n최우수 모델: {best_name}")
joblib.dump(trained_models[best_name]['ps'], '../outputs/best_model_ps.pkl')
joblib.dump(trained_models[best_name]['uts'], '../outputs/best_model_uts.pkl')
joblib.dump(scaler, '../outputs/scaler.pkl')
joblib.dump(imputer, '../outputs/imputer.pkl')
print("모델 저장 완료: best_model_ps.pkl, best_model_uts.pkl")
print("\n모든 분석 완료!")