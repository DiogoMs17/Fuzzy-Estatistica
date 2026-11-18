"""
FuzzyEdu — avaliação didática de desempenho estudantil com lógica fuzzy (Mamdani).

Execução:
    pip install -r requirements.txt
    streamlit run app.py

Observação: os parâmetros das funções de pertinência e as regras são hipóteses
didáticas configuráveis; não constituem um modelo validado de previsão de notas.
"""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =============================================================================
# Configuração
# =============================================================================
st.set_page_config(
    page_title="FuzzyEdu | Desempenho estudantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "navy": "#172B59",
    "blue": "#3157D5",
    "teal": "#147F91",
    "low": "#D96B62",
    "mid": "#E4AC43",
    "high": "#25A184",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#17233B; --muted:#71809A; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(180deg,#F7F9FD 0%,#F1F5FB 100%); }
    .block-container { max-width:1440px; padding-top:1.5rem; padding-bottom:3rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:#17233B !important; letter-spacing:-.03em; }
    .hero { background:linear-gradient(115deg,#172B59 0%,#294AA8 58%,#147F91 100%);
            padding:30px 34px; border-radius:22px; color:white; margin-bottom:22px;
            box-shadow:0 14px 34px rgba(27,55,111,.17); }
    .hero h1 { color:white !important; font-size:2rem; margin:4px 0 8px; }
    .hero p { color:#E3EBFF; margin:0; font-size:1rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:700; color:#BBD0FF; }
    .section-kicker { color:#3157D5; font-weight:800; font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; }
    .result { display:inline-block; padding:8px 13px; border-radius:999px; font-weight:800; }
    .result-low { background:#FFF1F0; color:#B93832; border:1px solid #F7D0CC; }
    .result-mid { background:#FFF7E7; color:#9B6500; border:1px solid #F4E0B5; }
    .result-high { background:#E9F8F3; color:#087B5B; border:1px solid #BDEBDD; }
    div[data-testid="stMetric"] { background:white; border:1px solid #E5EAF3; padding:14px 16px; border-radius:14px; }
    [data-testid="stSidebar"] { background:#F0F4FA; border-right:1px solid #E1E8F2; }
    div.stButton > button { border-radius:10px; font-weight:700; min-height:2.7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Modelo fuzzy
# =============================================================================
@st.cache_resource
def build_fuzzy_system():
    """Cria e mantém em cache o sistema Mamdani."""
    hours = ctrl.Antecedent(np.arange(0, 51, 1), "horas")
    attendance = ctrl.Antecedent(np.arange(0, 101, 1), "frequencia")
    performance = ctrl.Consequent(np.arange(0, 101, 1), "desempenho")

    # Parâmetros didáticos iniciais — ajustar conforme a especificação acadêmica.
    hours["baixas"] = fuzz.trapmf(hours.universe, [0, 0, 8, 18])
    hours["medias"] = fuzz.trimf(hours.universe, [10, 25, 40])
    hours["altas"] = fuzz.trapmf(hours.universe, [30, 40, 50, 50])

    attendance["baixa"] = fuzz.trapmf(attendance.universe, [0, 0, 50, 70])
    attendance["media"] = fuzz.trimf(attendance.universe, [55, 75, 90])
    attendance["alta"] = fuzz.trapmf(attendance.universe, [80, 90, 100, 100])

    performance["baixo"] = fuzz.trapmf(performance.universe, [0, 0, 25, 45])
    performance["medio"] = fuzz.trimf(performance.universe, [30, 50, 70])
    performance["alto"] = fuzz.trapmf(performance.universe, [55, 75, 100, 100])

    rules = [
        ctrl.Rule(hours["baixas"] & (attendance["baixa"] | attendance["media"]), performance["baixo"]),
        ctrl.Rule(hours["baixas"] & attendance["alta"], performance["medio"]),
        ctrl.Rule(hours["medias"] & attendance["baixa"], performance["baixo"]),
        ctrl.Rule(hours["medias"] & attendance["media"], performance["medio"]),
        ctrl.Rule(hours["medias"] & attendance["alta"], performance["alto"]),
        ctrl.Rule(hours["altas"] & attendance["baixa"], performance["medio"]),
        ctrl.Rule(hours["altas"] & (attendance["media"] | attendance["alta"]), performance["alto"]),
    ]
    return hours, attendance, performance, ctrl.ControlSystem(rules)


def classify(value: float) -> tuple[str, str]:
    if value < 40:
        return "Baixo", "result-low"
    if value < 65:
        return "Médio", "result-mid"
    return "Alto", "result-high"


@st.cache_data(show_spinner=False)
def calculate(hours_value: float, attendance_value: float) -> float:
    """Calcula um caso individual; o cache evita repetir entradas idênticas."""
    _, _, _, system = build_fuzzy_system()
    simulation = ctrl.ControlSystemSimulation(system)
    simulation.input["horas"] = float(np.clip(hours_value, 0, 50))
    simulation.input["frequencia"] = float(np.clip(attendance_value, 0, 100))
    simulation.compute()
    result = simulation.output.get("desempenho")
    if result is None or not np.isfinite(result):
        raise ValueError("O modelo não conseguiu produzir uma saída para esses valores.")
    return float(result)


def calculate_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula resultados linha a linha, preservando a ordem da base."""
    scores = []
    for h, a in zip(data["Hours_Studied"], data["Attendance"]):
        scores.append(calculate(float(h), float(a)))
    result = data.copy()
    result["Fuzzy_Score"] = scores
    result["Fuzzy_Class"] = result["Fuzzy_Score"].map(lambda x: classify(x)[0])
    return result


def membership_chart(universe, curves, current, x_title):
    fig = go.Figure()
    for label, membership in curves:
        fig.add_trace(
            go.Scatter(
                x=universe, y=membership, mode="lines",
                name=label, line={"width": 3},
            )
        )
    fig.add_vline(x=current, line_dash="dash", line_width=2, line_color=COLORS["navy"])
    fig.update_layout(
        height=285,
        margin={"l": 8, "r": 8, "t": 25, "b": 8},
        xaxis_title=x_title,
        yaxis_title="Grau de pertinência",
        yaxis={"range": [0, 1.05], "tickvals": [0, .25, .5, .75, 1]},
        legend={"orientation": "h", "y": 1.15, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"family": "DM Sans", "color": "#52627A"},
    )
    return fig


def read_csv_safely(uploaded_file) -> pd.DataFrame:
    """Tenta codificações e separadores comuns em arquivos CSV."""
    raw = uploaded_file.getvalue()
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        for separator in (None, ";", ",", "\t"):
            try:
                frame = pd.read_csv(
                    io.BytesIO(raw), encoding=encoding, sep=separator,
                    engine="python" if separator is None else "c",
                )
                # Remove espaços acidentais dos nomes das colunas.
                frame.columns = [str(column).strip() for column in frame.columns]
                if len(frame.columns) > 1:
                    return frame
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                last_error = exc
    raise ValueError(f"Não foi possível interpretar o CSV. {last_error or ''}")


def normalize_numeric(series: pd.Series) -> pd.Series:
    """Converte números; também aceita decimal com vírgula em colunas textuais."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    normalized = series.astype(str).str.strip().str.replace("%", "", regex=False)
    # Para entradas como 12,5, troca a vírgula decimal por ponto.
    normalized = normalized.str.replace(",", ".", regex=False)
    return pd.to_numeric(normalized, errors="coerce")


# =============================================================================
# Cabeçalho e navegação
# =============================================================================
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Projeto acadêmico • Inteligência computacional</div>
      <h1>FuzzyEdu — Análise de desempenho estudantil</h1>
      <p>Um sistema Mamdani que transforma horas de estudo e frequência em um indicador didático de 0 a 100.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎓 FuzzyEdu")
    st.caption("Painel de avaliação com lógica difusa")
    st.divider()
    page = st.radio("Navegação", ["Simulador", "Base de dados", "Como funciona"])
    st.divider()
    st.markdown("**Modelo utilizado**")
    st.markdown("Mamdani · Centroide")
    st.caption("Os limites numéricos são parâmetros iniciais documentados e ajustáveis.")


# =============================================================================
# Página 1 — Simulador
# =============================================================================
if page == "Simulador":
    st.markdown('<div class="section-kicker">01 / Simulação individual</div>', unsafe_allow_html=True)
    st.header("Teste um cenário de estudante")
    st.write("Altere os controles para recalcular automaticamente o indicador.")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Dados de entrada")
        hours = st.slider(
            "Horas de estudo por semana", min_value=0, max_value=50,
            value=20, step=1,
            help="Quantidade aproximada de horas dedicadas aos estudos por semana.",
        )
        attendance = st.slider(
            "Frequência nas aulas", min_value=0, max_value=100,
            value=80, step=1, format="%d%%",
        )
        st.caption("Escalas: 0–50 horas por semana e 0–100% de frequência.")
        st.info("O resultado é atualizado automaticamente ao alterar os controles.")

    try:
        score = calculate(hours, attendance)
    except Exception as exc:
        st.error(f"Falha ao calcular o indicador fuzzy: {exc}")
        st.stop()

    label, css_class = classify(score)
    with right:
        st.markdown('<div class="section-kicker">Resultado da inferência</div>', unsafe_allow_html=True)
        metric_col, gauge_col = st.columns([1, 1])
        with metric_col:
            st.metric("Indicador fuzzy", f"{score:.1f}/100")
            st.markdown(
                f'<span class="result {css_class}">Desempenho {label}</span>',
                unsafe_allow_html=True,
            )
        with gauge_col:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "/100", "font": {"size": 27, "color": COLORS["navy"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#AAB5C8"},
                    "bar": {"color": COLORS["blue"], "thickness": .25},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "#FCE7E5"},
                        {"range": [40, 65], "color": "#FFF2D7"},
                        {"range": [65, 100], "color": "#DDF4EC"},
                    ],
                },
            ))
            gauge.update_layout(
                height=190, margin={"l": 8, "r": 8, "t": 10, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        st.divider()
        st.markdown("**Interpretação**")
        explanations = {
            "Baixo": "A combinação informada ativa principalmente regras associadas ao desempenho baixo.",
            "Médio": "O resultado ficou na faixa intermediária; uma entrada pode compensar parcialmente a outra.",
            "Alto": "A combinação informada ativa principalmente regras associadas ao desempenho alto.",
        }
        st.write(explanations[label])
        st.caption("Indicador didático do modelo fuzzy. Não é uma nota prevista nem uma avaliação definitiva do estudante.")

    st.markdown("### Como os valores são interpretados")
    hours_var, attendance_var, _, _ = build_fuzzy_system()
    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("**Horas de estudo**")
        st.plotly_chart(
            membership_chart(
                hours_var.universe,
                [("Baixas", hours_var["baixas"].mf), ("Médias", hours_var["medias"].mf), ("Altas", hours_var["altas"].mf)],
                hours, "Horas por semana",
            ),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.caption("A linha tracejada marca o valor selecionado.")
    with chart2:
        st.markdown("**Frequência**")
        st.plotly_chart(
            membership_chart(
                attendance_var.universe,
                [("Baixa", attendance_var["baixa"].mf), ("Média", attendance_var["media"].mf), ("Alta", attendance_var["alta"].mf)],
                attendance, "Frequência (%)",
            ),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.caption("Um valor pode pertencer parcialmente a mais de um conjunto fuzzy.")


# =============================================================================
# Página 2 — Base de dados
# =============================================================================
elif page == "Base de dados":
    st.markdown('<div class="section-kicker">02 / Exploração dos dados</div>', unsafe_allow_html=True)
    st.header("Análise de uma base de estudantes")
    st.write("Envie um CSV para calcular o indicador fuzzy de cada registro.")
    uploaded = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

    with st.expander("Formato esperado do arquivo"):
        st.markdown(
            """
            **Colunas obrigatórias**
            - `Hours_Studied`: horas de estudo por semana, de 0 a 50.
            - `Attendance`: frequência percentual, de 0 a 100.

            **Coluna opcional**
            - `Exam_Score`: nota real da prova, usada apenas em uma comparação exploratória.

            O arquivo deve conter cabeçalho. O aplicativo aceita CSV separado por vírgula ou ponto e vírgula.
            """
        )

    if uploaded is not None:
        try:
            original = read_csv_safely(uploaded)
            required = ["Hours_Studied", "Attendance"]
            missing = [column for column in required if column not in original.columns]

            if missing:
                st.error("Colunas obrigatórias não encontradas: " + ", ".join(missing))
                st.write("Colunas identificadas:", ", ".join(map(str, original.columns)))
            else:
                data = original.copy()
                for column in required:
                    data[column] = normalize_numeric(data[column])

                total_rows = len(data)
                valid_mask = (
                    data[required].notna().all(axis=1)
                    & data["Hours_Studied"].between(0, 50)
                    & data["Attendance"].between(0, 100)
                )
                clean = data.loc[valid_mask].copy()
                invalid_count = total_rows - len(clean)

                if clean.empty:
                    st.warning("Nenhum registro válido. Verifique se horas estão entre 0 e 50 e frequência entre 0 e 100.")
                else:
                    if "Exam_Score" in clean.columns:
                        clean["Exam_Score"] = normalize_numeric(clean["Exam_Score"])

                    with st.spinner(f"Calculando {len(clean):,} registros..."):
                        result = calculate_dataframe(clean)

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Registros válidos", f"{len(result):,}".replace(",", "."))
                    k2.metric("Registros descartados", f"{invalid_count:,}".replace(",", "."))
                    k3.metric("Média do indicador", f"{result['Fuzzy_Score'].mean():.1f}/100")
                    k4.metric("Média de horas", f"{result['Hours_Studied'].mean():.1f} h")

                    if invalid_count:
                        st.warning(f"{invalid_count} registro(s) foram descartados por dados ausentes ou valores fora dos limites.")

                    st.markdown("### Distribuição das classificações")
                    left_chart, right_chart = st.columns(2)
                    order = ["Baixo", "Médio", "Alto"]
                    color_map = {"Baixo": COLORS["low"], "Médio": COLORS["mid"], "Alto": COLORS["high"]}
                    with left_chart:
                        counts = (
                            result["Fuzzy_Class"].value_counts()
                            .reindex(order, fill_value=0)
                            .rename_axis("Classificação")
                            .reset_index(name="Quantidade")
                        )
                        fig = px.bar(
                            counts, x="Classificação", y="Quantidade",
                            text="Quantidade", color="Classificação",
                            color_discrete_map=color_map, category_orders={"Classificação": order},
                        )
                        fig.update_layout(
                            height=330, showlegend=False, plot_bgcolor="white",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with right_chart:
                        fig = px.scatter(
                            result, x="Hours_Studied", y="Fuzzy_Score",
                            color="Fuzzy_Class", color_discrete_map=color_map,
                            category_orders={"Fuzzy_Class": order},
                            hover_data=["Attendance"],
                            labels={
                                "Hours_Studied": "Horas de estudo",
                                "Fuzzy_Score": "Indicador fuzzy",
                                "Fuzzy_Class": "Classificação",
                                "Attendance": "Frequência (%)",
                            },
                        )
                        fig.update_layout(height=330, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)

                    if "Exam_Score" in result.columns:
                        comparison = result.dropna(subset=["Exam_Score"])
                        if len(comparison) >= 2:
                            st.markdown("### Comparação exploratória com a nota real")
                            st.caption("O indicador foi definido por regras manuais; não foi treinado para prever Exam_Score.")
                            correlation = comparison[["Fuzzy_Score", "Exam_Score"]].corr().iloc[0, 1]
                            st.metric(
                                "Correlação de Pearson",
                                f"{correlation:.3f}" if pd.notna(correlation) else "Não disponível",
                            )
                            # Sem trendline OLS: evita dependência adicional do statsmodels.
                            comparison_fig = px.scatter(
                                comparison, x="Fuzzy_Score", y="Exam_Score",
                                labels={"Fuzzy_Score": "Indicador fuzzy", "Exam_Score": "Nota real"},
                            )
                            comparison_fig.update_layout(
                                height=350, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)"
                            )
                            st.plotly_chart(comparison_fig, use_container_width=True)
                        else:
                            st.info("São necessários pelo menos dois registros com Exam_Score válido para calcular a correlação.")

                    st.markdown("### Prévia dos resultados")
                    st.dataframe(result.head(200), use_container_width=True, hide_index=True)
                    csv_result = result.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ Baixar base calculada (CSV)",
                        data=csv_result,
                        file_name="resultado_fuzzy.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as exc:
            st.error(f"Não foi possível processar o arquivo: {exc}")
            st.caption("Confira o cabeçalho, a codificação e o separador do CSV.")


# =============================================================================
# Página 3 — Metodologia
# =============================================================================
else:
    st.markdown('<div class="section-kicker">03 / Metodologia</div>', unsafe_allow_html=True)
    st.header("Entenda a lógica do sistema")
    st.write(
        "O modelo Mamdani transforma as entradas em graus de pertinência, "
        "avalia as regras SE–ENTÃO, agrega as saídas e aplica o centroide "
        "para produzir um indicador entre 0 e 100."
    )

    st.markdown("### Variáveis e conjuntos fuzzy")
    variables = pd.DataFrame(
        [
            ["Horas de estudo", "Entrada", "0 a 50 horas/semana", "Baixas, médias e altas"],
            ["Frequência", "Entrada", "0% a 100%", "Baixa, média e alta"],
            ["Desempenho", "Saída", "0 a 100 pontos", "Baixo, médio e alto"],
        ],
        columns=["Variável", "Papel", "Universo", "Conjuntos"],
    )
    st.dataframe(variables, use_container_width=True, hide_index=True)

    st.markdown("### Regras SE–ENTÃO")
    rules = pd.DataFrame(
        [
            ["Baixas", "Baixa ou média", "Baixo"],
            ["Baixas", "Alta", "Médio"],
            ["Médias", "Baixa", "Baixo"],
            ["Médias", "Média", "Médio"],
            ["Médias", "Alta", "Alto"],
            ["Altas", "Baixa", "Médio"],
            ["Altas", "Média ou alta", "Alto"],
        ],
        columns=["Horas de estudo", "Frequência", "Desempenho"],
    )
    st.dataframe(rules, use_container_width=True, hide_index=True)

    st.markdown("### Etapas do processamento")
    stages = [
        ("1", "Fuzzificação", "Converte os valores numéricos em graus de pertinência."),
        ("2", "Aplicação das regras", "Avalia quais regras SE–ENTÃO são ativadas."),
        ("3", "Agregação", "Combina os resultados produzidos pelas regras."),
        ("4", "Defuzzificação", "Usa o centroide para obter um único valor numérico."),
    ]
    columns = st.columns(4)
    for column, (number, title, description) in zip(columns, stages):
        with column:
            st.markdown(f"**{number}. {title}**")
            st.write(description)

    st.warning(
        "Limitação acadêmica: a atividade não especifica necessariamente os pontos "
        "numéricos das funções de pertinência. Este aplicativo usa parâmetros iniciais "
        "explícitos no código. Eles devem ser confirmados com o professor. O resultado "
        "é um indicador didático, não uma previsão validada de nota."
    )


st.divider()
st.markdown(
    "<div style='text-align:center;color:#8190A7;font-size:.8rem'>"
    "FuzzyEdu • Atividade de Lógica Fuzzy • Python + Streamlit + scikit-fuzzy"
    "</div>",
    unsafe_allow_html=True,
)
