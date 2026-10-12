import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="기온 예측기")

st.title("🌡️ 기온 예측기")

st.markdown(
    "서울의 연평균기온 추이를 분석하고, 기후 변화의 경향을 파악한 뒤 "
    "회귀 분석을 활용해 미래 기온을 예측합니다."
)


@st.cache_data
def load_data():
    """서울 기온 CSV를 불러와 연도별 집계 데이터를 반환합니다."""
    url = (
        "https://raw.githubusercontent.com/greatsong/modudata/"
        "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    )
    df = pd.read_csv(url, encoding="utf-8-sig")
    df = df[["날짜", "지점", "평균기온", "최저기온", "최고기온"]].copy()
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.dropna(subset=["평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= 2025]
    grouped = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측수=("평균기온", "count"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["관측수"] >= 300]
    return grouped


data = load_data()
years = data["연도"].values
temps = data["평균기온"].values

# ── 전체 기간 회귀 분석 ──
full_coeffs = np.polyfit(years, temps, 1)
full_slope_per_year = full_coeffs[0]
full_intercept = full_coeffs[1]
full_slope_per_100 = full_slope_per_year * 100
r_full = float(np.corrcoef(years, temps)[0, 1])

n_years = len(years)
first_year = int(years.min())
last_year = int(years.max())

# ── 최근 20년 회귀 분석 ──
recent = data.tail(20)
recent_years = recent["연도"].values
recent_temps = recent["평균기온"].values
recent_coeffs = np.polyfit(recent_years, recent_temps, 1)
recent_slope_per_year = recent_coeffs[0]
recent_intercept = recent_coeffs[1]
recent_slope_per_100 = recent_slope_per_year * 100
r_recent = float(np.corrcoef(recent_years, recent_temps)[0, 1])

ratio = recent_slope_per_100 / full_slope_per_100
recent_first_year = int(recent_years.min())
recent_last_year = int(recent_years.max())

# ══════════════════════════════════════════════════
# 출력 영역
# ══════════════════════════════════════════════════

# 1) 상관계수 + 사용 연도 정보
st.markdown(f"**피어슨 상관계수 (r):** {r_full:.2f}")
st.markdown(
    f"**회귀 직선에 사용한 해:** {n_years}개 · {first_year}년 ~ {last_year}년"
)

# 2) 100년당 상승폭 헤드라인
st.markdown(
    f"## 📈 100년에 약 {full_slope_per_100:.1f} ℃ 상승"
)

# 3) 전체 기간 vs 최근 20년 비교 (2열)
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"전체 기간 ({first_year}~{last_year})",
        value=f"{full_slope_per_100:.1f} ℃/100년",
    )

with col2:
    recent_delta = recent_slope_per_100 - full_slope_per_100
    st.metric(
        label=f"최근 20년 ({recent_first_year}~{recent_last_year})",
        value=f"{recent_slope_per_100:.1f} ℃/100년",
        delta=f"+{recent_delta:.1f} ℃/100년",
        delta_color="normal",
    )

# 비율 문장
st.markdown(
    f"최근 20년의 기온 상승 속도는 전체 기간의 약 **{ratio:.1f}배**에 달합니다."
)

# 4) 슬라이더 + 예측 기온
selected_year = st.slider("예측 연도를 선택하세요", min_value=1900, max_value=2100, value=2025)
predicted_temp = full_slope_per_year * selected_year + full_intercept

st.markdown(f"## 🌡️ {selected_year}년 예측 기온")
st.metric(label="연평균기온 예측", value=f"{predicted_temp:.1f} ℃")

if selected_year <= last_year:
    st.markdown(f"※ {selected_year}년은 관측 데이터 범위 내입니다.")
else:
    st.markdown(f"※ {selected_year}년은 관측 데이터 범위 밖이므로 외삽 예측입니다.")

# 5) Plotly 차트
x_range = np.linspace(1900, 2100, 200)
full_line = full_slope_per_year * x_range + full_intercept
# 최근 20년 직선은 실제 관측 구간에만 그려 축 스케일이 망가지지 않도록 함
recent_x_range = np.linspace(recent_first_year, recent_last_year, 100)
recent_line = recent_slope_per_year * recent_x_range + recent_intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=years,
        y=temps,
        mode="markers",
        name="연평균기온 (관측)",
        marker=dict(color="blue", size=6),
    )
)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=full_line,
        mode="lines",
        name=f"전체 기간 추세 (r={r_full:.2f})",
        line=dict(dash="dash", color="blue"),
    )
)
fig.add_trace(
    go.Scatter(
        x=recent_x_range,
        y=recent_line,
        mode="lines",
        name=f"최근 20년 추세 (r={r_recent:.2f})",
        line=dict(color="orange"),
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    legend=dict(title="구분"),
    margin=dict(l=10, r=10, t=30, b=10),
)

st.plotly_chart(fig, use_container_width=True)


