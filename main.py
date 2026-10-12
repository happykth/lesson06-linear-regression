import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


@st.cache_data
def load_and_process():
    """서울 기온 CSV를 로드하고 연도별 평균기온을 집계합니다."""
    url = (
        "https://raw.githubusercontent.com/greatsong/modudata/"
        "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    )
    df = pd.read_csv(url, encoding="utf-8-sig")

    # 날짜 열을 datetime으로 파싱
    df["날짜"] = pd.to_datetime(df["날짜"])

    # 평균기온이 NA인 행 제거
    df = df.dropna(subset=["평균기온"])

    # 2025년 이후 데이터 제거 (클래스 참조 기간: 2025년까지)
    df = df[df["날짜"].dt.year <= 2025]

    # 연도 파생
    df["연도"] = df["날짜"].dt.year

    # 연도별 집계: 평균기온과 관측일수
    yearly = (
        df.groupby("연도")
        .agg(연평균기온=("평균기온", "mean"), 관측일수=("평균기온", "count"))
        .reset_index()
    )

    # 300일 미만 관측 연도 제거
    yearly = yearly[yearly["관측일수"] >= 300]
    return yearly


def main():
    st.set_page_config(page_title="기온 예측기", page_icon="🌡️")
    st.title("🌡️ 기온 예측기")

    st.markdown(
        "서울의 연평균기온 데이터를 기반으로 선형 회귀분석을 수행하고, "
        "선택한 연도의 예상 기온을 예측합니다."
    )

    yearly = load_and_process()

    연도 = yearly["연도"].values
    연평균기온 = yearly["연평균기온"].values

    # 피어슨 상관계수 계산
    corr = np.corrcoef(연도, 연평균기온)[0, 1]

    # 1차 선형 회귀 (최소자승법)
    slope, intercept = np.polyfit(연도, 연평균기온, 1)

    first_year = int(연도.min())
    last_year = int(연도.max())
    n_years = len(연도)

    # 분석 결과 출력
    st.write(f"**피어슨 상관계수 (r):** {corr:.4f}")
    st.write(
        f"**회귀 직선에 사용한 해:** {n_years}개 · {first_year}년 ~ {last_year}년"
    )

    # 연도 슬라이더
    selected_year = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2025)

    # 예측 온도 계산
    predicted_temp = float(slope * selected_year + intercept)

    # 큰 예측값 표시 (headline + metric)
    st.markdown(f"## 📊 {predicted_temp:.1f} ℃")
    st.metric(label="예상 연평균기온", value=f"{predicted_temp:.1f} ℃")

    # 선택한 연도가 피팅 범위 내인지 밖인지 안내
    if selected_year < first_year or selected_year > last_year:
        st.caption(
            "⚠️ 선택한 연도는 회귀 모델의 피팅 범위를 벗어납니다. "
            "해당 구간은 외삽(추정)입니다."
        )
    else:
        st.caption("✅ 선택한 연도는 회귀 모델의 피팅 범위 내에 있습니다.")

    # Plotly 차트 생성
    fig = go.Figure()

    # 산점도: 관측값
    fig.add_trace(
        go.Scatter(
            x=연도,
            y=연평균기온,
            mode="markers",
            name="연평균기온 (관측값)",
            marker=dict(size=6, color="steelblue"),
        )
    )

    # 회귀 직선 (슬라이더 전체 범위 1900~2100까지 확장)
    x_line = np.arange(1900, 2101)
    y_line = slope * x_line + intercept

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="예측 직선",
            line=dict(color="red", dash="dash", width=2),
        )
    )

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)",
        legend=dict(x=0, y=1),
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
