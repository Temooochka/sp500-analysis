# =====================================================================
#  📈 Аналитика S&P 500 по CSV-выгрузке (Google Colab)
#  Файлы: sp500_stocks.csv, sp500_companies.csv, sp500_index.csv
#  Загрузите их в /content или склонируйте репозиторий — папка data/ найдётся сама
#  (zip-архивы с CSV тоже подходят — код распакует их сам)
#  ⚠️ Учебный анализ исторических данных, не инвестиционная рекомендация.
# =====================================================================
import glob
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots
from scipy import stats as sps
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

DATA_CANDIDATES = ["/content", "/content/sample_data", "/content/sp500-analysis/data", "data", "."]
OUT_DIR = "/content" if os.path.isdir("/content") else "."
TRADING_DAYS = 252
RISK_FREE = 0.02  # безрисковая ставка для коэффициента Шарпа, годовая


def stock_files(d):
    """sp500_stocks.csv целиком или по частям (sp500_stocks_part1.csv, part2…)."""
    return sorted(glob.glob(os.path.join(d, "sp500_stocks*.csv")))


def find_data_dir():
    """Ищет папку с CSV; при необходимости распаковывает лежащие рядом zip-архивы."""
    for d in DATA_CANDIDATES:
        if not os.path.isdir(d):
            continue
        if not stock_files(d):
            for z in glob.glob(os.path.join(d, "*.zip")):
                zipfile.ZipFile(z).extractall(d)
        if stock_files(d) and all(os.path.exists(os.path.join(d, f)) for f in ["sp500_companies.csv", "sp500_index.csv"]):
            return d
    raise FileNotFoundError("Не найдены sp500_stocks.csv, sp500_companies.csv, sp500_index.csv в " + ", ".join(DATA_CANDIDATES))


DATA_DIR = find_data_dir()
print(f"📂 Данные: {os.path.abspath(DATA_DIR)}")

# =====================================================================
#  🎨 Оформление: единый стиль, палитра секторов, красивые таблицы
# =====================================================================
try:
    from IPython import get_ipython
    from IPython.display import HTML, display
    IN_NOTEBOOK = get_ipython() is not None
except ImportError:
    IN_NOTEBOOK = False

INK, MUTED, GRID, UP, DOWN, ACCENT = "#0f172a", "#64748b", "#e2e8f0", "#16a34a", "#dc2626", "#2563eb"
SECTOR_RU = {
    "Technology": "Технологии", "Communication Services": "Коммуникации", "Consumer Cyclical": "Потреб. циклич.",
    "Financial Services": "Финансы", "Healthcare": "Здравоохранение", "Industrials": "Промышленность",
    "Consumer Defensive": "Потреб. базовые", "Energy": "Энергетика", "Utilities": "Коммунальные",
    "Real Estate": "Недвижимость", "Basic Materials": "Сырьё",
}
SECTOR_COLORS = dict(zip(SECTOR_RU.values(), [
    "#2563eb", "#7c3aed", "#db2777", "#0891b2", "#16a34a", "#ea580c",
    "#65a30d", "#b45309", "#0d9488", "#9333ea", "#78716c",
]))

pio.templates["sp"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, 'Segoe UI', Roboto, Arial", size=13, color=INK),
    title=dict(font=dict(size=21, color=INK), x=0.01, xanchor="left", y=0.97),
    paper_bgcolor="white", plot_bgcolor="white",
    colorway=list(SECTOR_COLORS.values()),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(size=12, color=INK)),
    xaxis=dict(gridcolor=GRID, linecolor="#cbd5e1", zeroline=False, ticks="outside", tickcolor="#cbd5e1"),
    yaxis=dict(gridcolor=GRID, linecolor="#cbd5e1", zeroline=False),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", font=dict(size=12)),
    margin=dict(l=60, r=30, t=90, b=50),
))
pio.templates.default = "plotly_white+sp"

report = []  # HTML-фрагменты для итогового отчёта, в порядке появления


def show(fig, title, subtitle=None, height=None, to_report=True):
    """Показать график с заголовком/подзаголовком и добавить его в HTML-отчёт."""
    text = f"<b>{title}</b>" + (f"<br><span style='font-size:13px;color:{MUTED}'>{subtitle}</span>" if subtitle else "")
    fig.update_layout(title_text=text)
    if height:
        fig.update_layout(height=height)
    fig.show()
    if to_report:
        report.append(fig.to_html(full_html=False, include_plotlyjs=False))


def header(text, note=None):
    print("\n" + "═" * 72 + f"\n  {text}\n" + "═" * 72)
    report.append(f"<h2>{text}</h2>" + (f"<p class='note'>{note}</p>" if note else ""))
    if IN_NOTEBOOK:
        display(HTML(f"<div style='margin:28px 0 6px;padding:14px 18px;border-radius:12px;"
                     f"background:linear-gradient(90deg,#1e3a8a,#2563eb);color:white;"
                     f"font:600 20px Inter,Arial'>{text}"
                     + (f"<div style='font-weight:400;font-size:13px;opacity:.85'>{note}</div>" if note else "")
                     + "</div>"))


def table(df, pct=(), num=(), money=(), grad=(), bars=(), caption=None, cmap="RdYlGn", index=True):
    """Таблица с форматированием, цветовыми шкалами и мини-барами (в Colab) или текстом (в консоли)."""
    fmt = {**{c: "{:+.1%}" for c in pct}, **{c: "{:,.2f}" for c in num}, **{c: "${:,.0f}" for c in money}}
    sty = df.style.format(fmt, na_rep="—")
    for c in grad:
        sty = sty.background_gradient(subset=[c], cmap=cmap)
    for c in bars:
        sty = sty.bar(subset=[c], align="mid", color=["#fecaca", "#bbf7d0"])
    sty = sty.set_table_styles([
        {"selector": "th", "props": f"background:#f1f5f9;color:{INK};font-weight:600;padding:6px 10px;text-align:left"},
        {"selector": "td", "props": "padding:5px 10px;border-bottom:1px solid #f1f5f9"},
        {"selector": "caption", "props": f"caption-side:top;font-weight:600;font-size:15px;color:{INK};padding:6px 0"},
        {"selector": "", "props": "border-collapse:collapse;font-family:Inter,Arial;font-size:13px"},
    ])
    if caption:
        sty = sty.set_caption(caption)
    if not index:
        sty = sty.hide(axis="index")
    report.append(sty.to_html())
    if IN_NOTEBOOK:
        display(sty)
    else:
        print((f"\n  {caption}\n" if caption else "") + df.to_string(index=index))


def pct(x):
    return f"{x:+.1%}" if pd.notna(x) else "—"


def max_drawdown(series):
    return (series / series.cummax() - 1).min()


def cagr(series):
    s = series.dropna()
    years = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan


def curve_stats(r):
    """Ключевые метрики для ряда дневных/месячных доходностей."""
    curve = (1 + r.fillna(0)).cumprod()
    per_year = TRADING_DAYS if len(r) / max((r.index[-1] - r.index[0]).days / 365.25, 1e-9) > 60 else 12
    vol = r.std() * np.sqrt(per_year)
    c = cagr(curve)
    return {"CAGR": c, "Волатильность": vol, "Шарп": (c - RISK_FREE) / vol, "Макс. просадка": max_drawdown(curve),
            "Итог $1000": 1000 * curve.iloc[-1]}


# =====================================================================
#  1. Загрузка
# =====================================================================
companies = pd.read_csv(os.path.join(DATA_DIR, "sp500_companies.csv"))
companies["Sector"] = companies["Sector"].map(SECTOR_RU).fillna(companies["Sector"])
index = pd.read_csv(os.path.join(DATA_DIR, "sp500_index.csv"), parse_dates=["Date"]).set_index("Date")["S&P500"]
stocks = pd.concat([pd.read_csv(f, parse_dates=["Date"]) for f in stock_files(DATA_DIR)]).dropna(subset=["Adj Close"])
stocks = stocks.drop_duplicates(["Date", "Symbol"])

prices = stocks.pivot(index="Date", columns="Symbol", values="Adj Close").sort_index()
volume = stocks.pivot(index="Date", columns="Symbol", values="Volume").sort_index()
returns = prices.pct_change(fill_method=None)
idx_ret = index.pct_change().dropna()

info = companies.set_index("Symbol")
sector_of = info["Sector"]
name_of = info["Shortname"]
missing = sorted(set(companies["Symbol"]) - set(prices.columns))

header("📦 Данные", "Что есть в выгрузке")
print(f"  Компаний в справочнике:      {len(companies)}")
print(f"  Акций с историей цен:        {prices.shape[1]}")
print(f"  Период цен акций:            {prices.index.min():%d.%m.%Y} — {prices.index.max():%d.%m.%Y}")
print(f"  Период индекса S&P 500:      {index.index.min():%d.%m.%Y} — {index.index.max():%d.%m.%Y}")
print(f"  ⚠️ Без истории цен: {len(missing)} компаний (например, {', '.join(missing[:8])})")

# =====================================================================
#  2. Дашборд ключевых показателей
# =====================================================================
idx_dd = index / index.cummax() - 1
ytd_base = index[index.index < pd.Timestamp(index.index[-1].year, 1, 1)].iloc[-1]
top10_weight = companies.nlargest(10, "Weight")["Weight"].sum()

fig = make_subplots(rows=2, cols=3, specs=[[{"type": "indicator"}] * 3] * 2, vertical_spacing=0.25)
cards = [
    dict(mode="number+delta", value=index.iloc[-1], number=dict(valueformat=",.0f"),
         delta=dict(reference=ytd_base, relative=True, valueformat=".1%"), title="S&P 500<br><sub>с начала года</sub>"),
    dict(mode="number", value=cagr(index), number=dict(valueformat=".1%"), title="Доходность в год<br><sub>CAGR за 10 лет</sub>"),
    dict(mode="number", value=idx_dd.min(), number=dict(valueformat=".1%", font=dict(color=DOWN)),
         title=f"Худшая просадка<br><sub>{idx_dd.idxmin():%d.%m.%Y}</sub>"),
    dict(mode="gauge+number", value=top10_weight, number=dict(valueformat=".0%"),
         gauge=dict(axis=dict(range=[0, 1], tickformat=".0%"), bar=dict(color=ACCENT)),
         title="Вес топ-10 компаний"),
    dict(mode="number", value=companies["Marketcap"].sum() / 1e12, number=dict(prefix="$", suffix=" трлн", valueformat=".1f"),
         title="Капитализация<br><sub>всех компаний</sub>"),
    dict(mode="gauge+number", value=(prices.iloc[-1] > prices.rolling(200).mean().iloc[-1]).mean(),
         number=dict(valueformat=".0%"), gauge=dict(axis=dict(range=[0, 1], tickformat=".0%"), bar=dict(color=UP)),
         title="Акций выше SMA-200<br><sub>ширина рынка</sub>"),
]
for i, c in enumerate(cards):
    fig.add_trace(go.Indicator(**c), row=i // 3 + 1, col=i % 3 + 1)
show(fig, "Панель S&P 500", f"Данные на {index.index[-1]:%d.%m.%Y}", height=480)

# =====================================================================
#  3. Индекс S&P 500
# =====================================================================
header("📈 Индекс S&P 500", "Рост, просадки, лучшие и худшие дни")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
fig.add_scatter(x=index.index, y=index, name="S&P 500", line=dict(color=ACCENT, width=2),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.06)", row=1, col=1)
fig.add_scatter(x=index.index, y=index.rolling(200).mean(), name="SMA 200", line=dict(color="#f59e0b", dash="dot"), row=1, col=1)
fig.add_scatter(x=idx_dd.index, y=idx_dd, name="Просадка", fill="tozeroy", line=dict(color=DOWN, width=1),
                fillcolor="rgba(220,38,38,0.25)", row=2, col=1)
for date, label in [("2018-12-24", "Распродажа 2018"), ("2020-03-23", "COVID-19"), ("2022-10-12", "Ставки ФРС")]:
    d = pd.Timestamp(date)
    if d in index.index:
        fig.add_annotation(x=d, y=index[d], text=label, showarrow=True, arrowhead=2, ay=-45,
                           bgcolor="white", bordercolor=GRID, row=1, col=1)
fig.update_yaxes(tickformat=".0%", row=2, col=1)
fig.update_xaxes(rangeselector=dict(buttons=[dict(count=1, label="1г", step="year", stepmode="backward"),
                                             dict(count=3, label="3г", step="year", stepmode="backward"),
                                             dict(count=5, label="5г", step="year", stepmode="backward"),
                                             dict(step="all", label="Всё")]), row=1, col=1)
show(fig, "S&P 500 и «подводная» диаграмма просадок", "Нижний график — насколько индекс ниже своего исторического максимума", height=620)

yearly = index.resample("YE").last().pct_change().dropna()
yearly.index = yearly.index.year
fig = go.Figure(go.Bar(x=yearly.index, y=yearly, marker_color=[UP if v > 0 else DOWN for v in yearly],
                       text=[f"{v:+.1%}" for v in yearly], textposition="outside"))
fig.add_hline(y=yearly.mean(), line_dash="dash", line_color=MUTED, annotation_text=f"среднее {yearly.mean():+.1%}")
fig.update_yaxes(tickformat=".0%")
show(fig, "Доходность по годам")

# Эпизоды просадок > 10%: сколько падали и сколько восстанавливались
episodes, peak, trough, in_dd = [], None, None, False
for d, v in idx_dd.items():
    if not in_dd and v < 0:
        in_dd, peak, trough = True, d, d
    if in_dd:
        if v < idx_dd[trough]:
            trough = d
        if v == 0:
            if idx_dd[trough] <= -0.10:
                episodes.append((peak, trough, d, idx_dd[trough]))
            in_dd = False
if in_dd and idx_dd[trough] <= -0.10:
    episodes.append((peak, trough, pd.NaT, idx_dd[trough]))
ep = pd.DataFrame(episodes, columns=["Пик", "Дно", "Восстановление", "Падение"])
ep["Дней падения"] = (ep["Дно"] - ep["Пик"]).dt.days
ep["Дней восстановления"] = (ep["Восстановление"] - ep["Дно"]).dt.days
for c in ["Пик", "Дно", "Восстановление"]:
    ep[c] = ep[c].dt.strftime("%d.%m.%Y")
table(ep, pct=["Падение"], grad=["Падение"], caption="Коррекции индекса больше 10%", index=False)

best_worst = pd.concat([idx_ret.nlargest(7).rename("Лучшие дни"), idx_ret.nsmallest(7).rename("Худшие дни")], axis=1)
miss = {n: (1 + idx_ret.drop(idx_ret.nlargest(n).index)).prod() - 1 for n in [0, 5, 10, 20, 30]}
fig = go.Figure(go.Bar(x=[f"Без {n} лучших" if n else "Всё время в рынке" for n in miss], y=list(miss.values()),
                       marker_color=[ACCENT] + ["#93c5fd"] * (len(miss) - 1), text=[f"{v:+.0%}" for v in miss.values()],
                       textposition="outside"))
fig.update_yaxes(tickformat=".0%")
show(fig, "Цена пропуска лучших дней", "Итоговый рост индекса, если пропустить N самых сильных дней")
miss10 = miss[10] + 1
full = miss[0] + 1

# Распределение доходностей: «толстые хвосты», VaR, CVaR
mu_d, sd_d = idx_ret.mean(), idx_ret.std()
var95, var99 = np.percentile(idx_ret, 5), np.percentile(idx_ret, 1)
cvar95 = idx_ret[idx_ret <= var95].mean()
xs = np.linspace(idx_ret.min(), idx_ret.max(), 400)
fig = go.Figure()
fig.add_histogram(x=idx_ret, nbinsx=150, histnorm="probability density", name="Факт", marker_color="#93c5fd")
fig.add_scatter(x=xs, y=sps.norm.pdf(xs, mu_d, sd_d), name="Нормальное распределение", line=dict(color=INK, width=2))
for v, lab in [(var95, "VaR 95%"), (var99, "VaR 99%")]:
    fig.add_vline(x=v, line_dash="dash", line_color=DOWN, annotation_text=f"{lab}: {v:.2%}")
fig.update_xaxes(tickformat=".0%", range=[-0.08, 0.08])
show(fig, "Распределение дневных доходностей", f"Эксцесс {sps.kurtosis(idx_ret):.1f} — крупные движения бывают гораздо чаще, чем «по учебнику»")
print(f"  VaR 95%: в 1 день из 20 индекс падает сильнее {var95:.2%}; средняя потеря в такие дни (CVaR) {cvar95:.2%}")
sigma_days = (idx_ret.abs() > 4 * sd_d).sum()
print(f"  Дней с движением > 4σ: {sigma_days} (нормальное распределение предсказало бы ≈ {len(idx_ret) * 2 * sps.norm.sf(4):.2f})")

roll_vol = idx_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
fig = go.Figure(go.Scatter(x=roll_vol.index, y=roll_vol, line=dict(color=INK, width=1.3), name="Волатильность"))
for lo, hi, color, lab in [(0, 0.12, "rgba(22,163,74,0.10)", "спокойно"), (0.12, 0.25, "rgba(245,158,11,0.10)", "нервно"),
                           (0.25, 1.5, "rgba(220,38,38,0.10)", "паника")]:
    fig.add_hrect(y0=lo, y1=hi, fillcolor=color, line_width=0, annotation_text=lab, annotation_position="top left")
fig.update_yaxes(tickformat=".0%", range=[0, roll_vol.max() * 1.05])
show(fig, "Режимы волатильности", "Месячная скользящая волатильность в годовом выражении")

# =====================================================================
#  4. Сезонность
# =====================================================================
header("🗓 Сезонность", "Есть ли у рынка «любимые» месяцы и дни")
MONTHS = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
monthly_ret = index.resample("ME").last().pct_change().dropna()
season = pd.DataFrame({"year": monthly_ret.index.year, "month": monthly_ret.index.month, "ret": monthly_ret.values})
grid = season.pivot(index="year", columns="month", values="ret")
fig = px.imshow(grid, x=MONTHS[: grid.shape[1]], y=grid.index.astype(str), color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0, text_auto=".1%", aspect="auto")
fig.update_coloraxes(colorbar_tickformat=".0%")
show(fig, "Доходность по месяцам", height=520)

by_month = season.groupby("month")["ret"].agg(["mean", lambda s: (s > 0).mean()])
by_month.columns = ["Средняя", "Доля лет с ростом"]
by_month.index = MONTHS
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_bar(x=by_month.index, y=by_month["Средняя"], name="Средняя доходность",
            marker_color=[UP if v > 0 else DOWN for v in by_month["Средняя"]])
fig.add_scatter(x=by_month.index, y=by_month["Доля лет с ростом"], name="Доля лет с ростом", mode="lines+markers",
                line=dict(color=INK), secondary_y=True)
fig.update_yaxes(tickformat=".1%", secondary_y=False)
fig.update_yaxes(tickformat=".0%", range=[0, 1.05], secondary_y=True, showgrid=False)
show(fig, "Средний месяц года")

DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт"]
dow = idx_ret.groupby(idx_ret.index.dayofweek).agg(["mean", "std", lambda s: (s > 0).mean()])
dow.columns = ["Средняя", "Разброс", "Доля дней роста"]
dow.index = DAYS
table(dow, pct=["Средняя", "Разброс", "Доля дней роста"], bars=["Средняя"], caption="Дни недели")

# =====================================================================
#  5. Структура рынка
# =====================================================================
header("🏢 Структура S&P 500", "Секторы, концентрация, география")
companies["Marketcap_bn"] = companies["Marketcap"] / 1e9
companies["cap_per_employee_m"] = companies["Marketcap"] / companies["Fulltimeemployees"] / 1e6

fig = px.sunburst(companies, path=["Sector", "Industry"], values="Marketcap", color="Sector",
                  color_discrete_map=SECTOR_COLORS, maxdepth=2)
fig.update_traces(insidetextorientation="radial")
show(fig, "Из чего состоит индекс", "Сектор → отрасль, площадь = капитализация (кликните для приближения)", height=650)

# Карта рынка как у финансовых терминалов: размер = капитализация, цвет = доходность за год
last_year = prices[prices.index >= prices.index[-1] - pd.Timedelta(days=365)]
ret_1y = (last_year.ffill().iloc[-1] / last_year.bfill().iloc[0] - 1).rename("ret_1y")
heat = companies.merge(ret_1y, left_on="Symbol", right_index=True)
fig = px.treemap(heat, path=[px.Constant("S&P 500 (акции с ценами)"), "Sector", "Symbol"], values="Marketcap",
                 color=heat["ret_1y"].clip(-0.5, 0.5), color_continuous_scale=[[0, "#b91c1c"], [0.5, "#334155"], [1, "#15803d"]],
                 color_continuous_midpoint=0, custom_data=["Shortname", "ret_1y"])
fig.update_traces(texttemplate="<b>%{label}</b><br>%{customdata[1]:+.0%}",
                  hovertemplate="<b>%{customdata[0]}</b><br>За год: %{customdata[1]:+.1%}<extra></extra>",
                  textfont=dict(color="white"))
fig.update_coloraxes(colorbar_tickformat="+.0%", colorbar_title="За год")
show(fig, "Тепловая карта рынка за последний год", "Размер — капитализация, цвет — изменение цены", height=700)

top10 = companies.nlargest(10, "Weight")
fig = go.Figure(go.Bar(y=top10["Shortname"][::-1], x=top10["Weight"][::-1], orientation="h",
                       marker_color=[SECTOR_COLORS.get(s, MUTED) for s in top10["Sector"][::-1]],
                       text=[f"{w:.2%}  ·  ${c:,.0f} млрд" for w, c in zip(top10["Weight"][::-1], top10["Marketcap_bn"][::-1])],
                       textposition="outside"))
fig.update_xaxes(tickformat=".0%", range=[0, top10["Weight"].max() * 1.45])
show(fig, f"Топ-10 компаний = {top10_weight:.0%} индекса", "Цвет — сектор")

w = companies["Weight"].sort_values(ascending=False).cumsum().reset_index(drop=True)
fig = go.Figure(go.Scatter(x=np.arange(1, len(w) + 1), y=w, fill="tozeroy", line=dict(color=ACCENT)))
for share in [0.5, 0.8]:
    n = int((w < share).sum() + 1)
    fig.add_annotation(x=n, y=share, text=f"{share:.0%} индекса = {n} компаний", showarrow=True, arrowhead=2, ax=60, ay=30)
fig.update_yaxes(tickformat=".0%")
fig.update_xaxes(title="Число крупнейших компаний")
show(fig, "Кривая концентрации")

sectors = companies.groupby("Sector").agg(Компаний=("Symbol", "size"), Капитализация=("Marketcap", "sum"),
                                          Вес=("Weight", "sum"), Рост_выручки=("Revenuegrowth", "median"),
                                          Сотрудников=("Fulltimeemployees", "sum")).sort_values("Вес", ascending=False)
sectors["Капитализация"] = sectors["Капитализация"] / 1e9
table(sectors.rename(columns={"Капитализация": "Капитализация, $ млрд", "Рост_выручки": "Рост выручки (медиана)"}),
      pct=["Вес", "Рост выручки (медиана)"], num=["Капитализация, $ млрд"], bars=["Вес"],
      grad=["Рост выручки (медиана)"], caption="Секторы")

by_state = companies[companies["Country"] == "United States"].groupby("State").agg(
    компаний=("Symbol", "size"), cap=("Marketcap", "sum"), лидер=("Shortname", "first")).reset_index()
fig = px.choropleth(by_state, locations="State", locationmode="USA-states", color="компаний", scope="usa",
                    hover_data={"cap": ":.3s", "лидер": True}, color_continuous_scale="Blues")
fig.update_geos(bgcolor="white", lakecolor="white")
show(fig, "Где находятся штаб-квартиры", f"Топ городов: {', '.join(companies['City'].value_counts().head(5).index)}")

fig = px.scatter(companies.dropna(subset=["Fulltimeemployees"]), x="Fulltimeemployees", y="Marketcap", color="Sector",
                 color_discrete_map=SECTOR_COLORS, log_x=True, log_y=True, hover_name="Shortname", size="Weight",
                 size_max=55, opacity=0.8)
for _, r in companies.nlargest(6, "Marketcap").iterrows():
    fig.add_annotation(x=np.log10(r["Fulltimeemployees"]), y=np.log10(r["Marketcap"]), text=r["Symbol"], showarrow=False, yshift=14)
show(fig, "Сотрудники vs капитализация", "Чем выше и левее — тем дороже стоит каждый сотрудник", height=600)

# =====================================================================
#  6. Доходность и риск каждой акции
# =====================================================================
header("📊 Акции: доходность и риск", "Только акции с историей цен; в рейтингах — от 3 лет истории")
common = returns.index.intersection(idx_ret.index)
rows = []
for sym in prices.columns:
    p = prices[sym].dropna()
    if len(p) < TRADING_DAYS:
        continue
    r = returns[sym].dropna()
    vol = r.std() * np.sqrt(TRADING_DAYS)
    c = cagr(p)
    rr = returns.loc[common, sym].dropna()
    beta = rr.cov(idx_ret.loc[rr.index]) / idx_ret.loc[rr.index].var() if len(rr) > 100 else np.nan
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    rows.append({
        "Symbol": sym, "Компания": name_of.get(sym, sym), "Сектор": sector_of.get(sym, "?"),
        "Лет истории": len(p) / TRADING_DAYS, "Рост за всё время": p.iloc[-1] / p.iloc[0] - 1, "CAGR": c,
        "Волатильность": vol, "Шарп": (c - RISK_FREE) / vol, "Сортино": (c - RISK_FREE) / downside,
        "Макс. просадка": max_drawdown(p), "Бета": beta, "За год": ret_1y.get(sym, np.nan),
        "Доля дней роста": (r > 0).mean(),
    })
stats = pd.DataFrame(rows).set_index("Symbol")
ranked = stats[stats["Лет истории"] >= 3]
idx_cagr, idx_vol = cagr(index), idx_ret.std() * np.sqrt(TRADING_DAYS)

show_cols = ["Компания", "Сектор", "CAGR", "Волатильность", "Шарп", "Макс. просадка", "За год"]
for col, title, asc in [("CAGR", "🚀 Лучшие по среднегодовой доходности", False),
                        ("CAGR", "🐌 Худшие по среднегодовой доходности", True),
                        ("Шарп", "⚖️ Лучшее соотношение доходность/риск", False),
                        ("За год", "📅 Лидеры последнего года", False),
                        ("Макс. просадка", "🕳 Самые глубокие падения", True)]:
    table(ranked.sort_values(col, ascending=asc).head(8)[show_cols], pct=["CAGR", "Волатильность", "Макс. просадка", "За год"],
          num=["Шарп"], grad=["CAGR", "За год"], bars=["Шарп"], caption=title)

fig = px.scatter(ranked.reset_index(), x="Волатильность", y="CAGR", color="Сектор", color_discrete_map=SECTOR_COLORS,
                 size=ranked["Лет истории"].values, size_max=14, hover_name="Компания",
                 hover_data={"Symbol": True, "Шарп": ":.2f", "Макс. просадка": ":.0%"}, opacity=0.85)
for _, r in ranked.reset_index().nlargest(8, "CAGR").iterrows():
    fig.add_annotation(x=r["Волатильность"], y=r["CAGR"], text=r["Symbol"], showarrow=False, yshift=12, font=dict(size=11))
vmax = ranked["Волатильность"].max()
fig.add_scatter(x=[0, vmax], y=[RISK_FREE, RISK_FREE + (idx_cagr - RISK_FREE) / idx_vol * vmax], mode="lines",
                line=dict(color=MUTED, dash="dash"), name="Шарп индекса")
fig.add_scatter(x=[idx_vol], y=[idx_cagr], mode="markers+text", text=["  S&P 500"], textposition="middle right",
                marker=dict(size=20, color=INK, symbol="star"), name="Индекс")
fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
show(fig, "Карта риск/доходность", "Выше пунктира — лучше индекса с поправкой на риск", height=620)

fig = px.violin(ranked.reset_index(), x="Сектор", y="CAGR", color="Сектор", color_discrete_map=SECTOR_COLORS,
                box=True, points="all", hover_name="Компания")
fig.add_hline(y=idx_cagr, line_dash="dash", line_color=INK, annotation_text="S&P 500")
fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
show(fig, "Разброс доходностей внутри секторов", height=550)
print(f"  Акций, обогнавших индекс по CAGR: {(ranked['CAGR'] > idx_cagr).mean():.0%}")


def screener(min_cagr=None, max_vol=None, min_sharpe=None, max_drawdown=None, sector=None, n=20):
    """Фильтр акций. Пример: screener(min_cagr=0.2, max_vol=0.3, sector='Технологии')"""
    s = ranked.copy()
    if min_cagr is not None:
        s = s[s["CAGR"] >= min_cagr]
    if max_vol is not None:
        s = s[s["Волатильность"] <= max_vol]
    if min_sharpe is not None:
        s = s[s["Шарп"] >= min_sharpe]
    if max_drawdown is not None:
        s = s[s["Макс. просадка"] >= -abs(max_drawdown)]
    if sector:
        s = s[s["Сектор"] == sector]
    table(s.sort_values("Шарп", ascending=False).head(n)[show_cols], pct=["CAGR", "Волатильность", "Макс. просадка", "За год"],
          num=["Шарп"], grad=["Шарп"], caption=f"Скринер: найдено {len(s)}")
    return s


screener(min_cagr=0.15, max_vol=0.30, max_drawdown=0.40)


def radar(*symbols):
    """Профиль акций на «паутинке»: перцентили по 6 метрикам. Пример: radar('NVDA', 'MSFT', 'AMZN')"""
    metrics = {"Доходность": ranked["CAGR"].rank(pct=True), "Шарп": ranked["Шарп"].rank(pct=True),
               "Стабильность": (-ranked["Волатильность"]).rank(pct=True), "Устойчивость к падениям": ranked["Макс. просадка"].rank(pct=True),
               "Импульс (год)": ranked["За год"].rank(pct=True), "Независимость от рынка": (-ranked["Бета"]).rank(pct=True)}
    m = pd.DataFrame(metrics)
    fig = go.Figure()
    for s in symbols:
        s = s.upper()
        if s in m.index:
            vals = list(m.loc[s] * 100)
            fig.add_scatterpolar(r=vals + vals[:1], theta=list(m.columns) + [m.columns[0]], fill="toself", name=s, opacity=0.6)
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")))
    show(fig, "Профиль акций", "Перцентиль среди всех акций: 100% — лучше всех", height=550)


radar(*ranked.nlargest(3, "Шарп").index)

# =====================================================================
#  7. Гонка лидеров (анимация)
# =====================================================================
header("🏁 Гонка лидеров", "Во сколько раз выросла каждая акция с 2015 года — нажмите ▶")
start = pd.Timestamp("2015-01-02")
base = prices.loc[prices.index >= start]
base = base.loc[:, base.iloc[0].notna()]
growth = (base / base.iloc[0]).resample("QE").last()
frames, steps = [], []
for q, row in growth.iterrows():
    top = row.dropna().nlargest(10).iloc[::-1]
    label = f"{q.year} Q{q.quarter}"
    frames.append(go.Frame(name=label, data=[go.Bar(
        x=top.values, y=list(range(10)), orientation="h", text=[f"<b>{s}</b> ×{v:.1f}" for s, v in top.items()],
        textposition="outside", marker_color=[SECTOR_COLORS.get(sector_of.get(s), MUTED) for s in top.index],
        hovertext=[name_of.get(s, s) for s in top.index], hoverinfo="text")],
        layout=go.Layout(annotations=[dict(text=label, x=0.98, y=0.08, xref="paper", yref="paper", showarrow=False,
                                           font=dict(size=40, color="#cbd5e1"))])))
    steps.append(dict(method="animate", label=label, args=[[label], dict(mode="immediate", frame=dict(duration=300, redraw=True))]))
fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
fig.update_xaxes(type="log", range=[0, np.log10(growth.max().max() * 3)], title="Во сколько раз выросла (лог. шкала)",
                 tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500], ticktext=["×1", "×2", "×5", "×10", "×20", "×50", "×100", "×200", "×500"])
fig.update_yaxes(showticklabels=False, range=[-0.6, 9.6])
fig.update_layout(
    updatemenus=[dict(type="buttons", x=0, y=-0.12, xanchor="left", buttons=[
        dict(label="▶ Старт", method="animate", args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)]),
        dict(label="⏸", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
    sliders=[dict(steps=steps, x=0.12, y=-0.08, len=0.88, currentvalue=dict(visible=False))],
)
show(fig, "Топ-10 акций по росту с 2015 года", "Цвет — сектор", height=620)

# =====================================================================
#  8. Калькуляторы: разовая покупка и регулярные взносы
# =====================================================================
header("💵 Калькуляторы инвестора")


def _series(sym):
    return index if sym.upper() in ("S&P500", "SPX") else prices[sym.upper()]


def invest(symbols, start="2015-01-02", amount=1000, plot=True):
    """Сколько стоили бы $amount, вложенные в каждую бумагу в дату start. Пример: invest(['NVDA','S&P500'])"""
    start = pd.Timestamp(start)
    curves = {}
    for s in symbols:
        if s.upper() not in prices.columns and s.upper() not in ("S&P500", "SPX"):
            print(f"  Нет цен по {s}")
            continue
        p = _series(s)
        p = p[p.index >= start].dropna()
        if len(p):
            curves[s.upper()] = p / p.iloc[0] * amount
    curves = pd.DataFrame(curves)
    final = curves.ffill().iloc[-1].sort_values(ascending=False)
    if plot:
        fig = go.Figure()
        for s in final.index:
            fig.add_scatter(x=curves.index, y=curves[s], name=f"{s}: ${final[s]:,.0f}",
                            line=dict(width=3 if s == "S&P500" else 1.8, color=INK if s == "S&P500" else None))
        fig.update_yaxes(type="log", tickprefix="$")
        show(fig, f"${amount:,} с {start:%d.%m.%Y}", "Логарифмическая шкала: одинаковый наклон = одинаковый % роста")
    else:
        for s, v in final.items():
            print(f"  {s:<8} ${amount:,} → ${v:,.0f}")
    return curves


def dca(sym="S&P500", start="2015-01-02", monthly=100):
    """Регулярные покупки каждый месяц против одной покупки на ту же сумму. Пример: dca('NVDA', monthly=200)"""
    p = _series(sym)
    p = p[p.index >= pd.Timestamp(start)].dropna()
    buy_dates = p.resample("MS").first().index
    buy_px = p.reindex(p.index[p.index.searchsorted(buy_dates)])
    shares = (monthly / buy_px).cumsum().reindex(p.index, method="ffill")
    invested = pd.Series(monthly * np.arange(1, len(buy_px) + 1), index=buy_px.index).reindex(p.index, method="ffill")
    value = shares * p
    lump = p / p.iloc[0] * invested.iloc[-1]
    fig = go.Figure()
    fig.add_scatter(x=p.index, y=invested, name="Вложено", line=dict(color=MUTED, dash="dot"), fill="tozeroy",
                    fillcolor="rgba(100,116,139,0.08)")
    fig.add_scatter(x=p.index, y=value, name=f"Каждый месяц по ${monthly}", line=dict(color=ACCENT, width=2.5))
    fig.add_scatter(x=p.index, y=lump, name=f"Всё сразу (${invested.iloc[-1]:,.0f})", line=dict(color="#f59e0b"))
    fig.update_yaxes(tickprefix="$")
    show(fig, f"{sym.upper()}: регулярные взносы vs всё сразу",
         f"Вложено ${invested.iloc[-1]:,.0f} → стало ${value.iloc[-1]:,.0f} (×{value.iloc[-1] / invested.iloc[-1]:.2f}); "
         f"разовая покупка дала бы ${lump.iloc[-1]:,.0f}")


top_caps = [s for s in companies.nlargest(15, "Marketcap")["Symbol"] if s in prices.columns][:6]
invest(["S&P500"] + top_caps)
dca("S&P500", monthly=100)

# =====================================================================
#  9. Секторы: доходность, ротация, корреляции, COVID
# =====================================================================
header("🧩 Секторы", "Равновзвешенные портфели акций каждого сектора")
sector_ret = returns.T.groupby(sector_of.reindex(returns.columns)).mean().T.dropna(how="all")
sector_ret = sector_ret[sector_ret.index >= "2011-01-01"]
sector_curve = (1 + sector_ret.fillna(0)).cumprod()
fig = go.Figure()
for s in sector_curve.columns:
    fig.add_scatter(x=sector_curve.index, y=sector_curve[s], name=s, line=dict(color=SECTOR_COLORS.get(s), width=1.8))
fig.update_yaxes(type="log", title="Рост $1")
show(fig, "Рост $1 в каждом секторе с 2011 года", "Логарифмическая шкала", height=560)

sector_stats = pd.DataFrame({s: curve_stats(sector_ret[s].dropna()) for s in sector_ret}).T.sort_values("CAGR", ascending=False)
table(sector_stats, pct=["CAGR", "Волатильность", "Макс. просадка"], num=["Шарп"], money=["Итог $1000"],
      grad=["CAGR", "Шарп"], caption="Секторы: доходность и риск")

# «Таблица Менделеева» доходностей: какой сектор был лучшим в каждом году
sec_year = (1 + sector_ret).resample("YE").prod() - 1
sec_year.index = sec_year.index.year
names = list(sector_ret.columns)
rank_z, rank_text = [], []
for _, row in sec_year.iterrows():
    order = row.sort_values(ascending=False)
    rank_z.append([names.index(s) for s in order.index])
    rank_text.append([f"{s}<br><b>{v:+.0%}</b>" for s, v in order.items()])
n = len(names)
scale = []
for i, s in enumerate(names):
    c = SECTOR_COLORS.get(s, MUTED)
    scale += [[i / n, c], [(i + 1) / n, c]]
fig = go.Figure(go.Heatmap(z=np.array(rank_z).T, x=sec_year.index.astype(str), y=[f"#{i + 1}" for i in range(n)],
                           text=np.array(rank_text).T, texttemplate="%{text}", textfont=dict(size=10, color="white"),
                           colorscale=scale, zmin=0, zmax=n, showscale=False, xgap=3, ygap=3, hoverinfo="text"))
fig.update_yaxes(autorange="reversed")
show(fig, "Ротация секторов: рейтинг по годам", "Сверху — лучший сектор года. Лидеры постоянно меняются → диверсификация работает", height=720)

show(px.imshow(sector_ret.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
     "Корреляция секторов", "1 — двигаются одинаково, 0 — независимо", height=600)

# COVID: как секторы упали и как восстановились
crash = sector_curve.loc["2020-03-23"] / sector_curve.loc["2020-02-19"] - 1
rebound = sector_curve.loc["2020-12-31"] / sector_curve.loc["2020-03-23"] - 1
covid = pd.DataFrame({"Падение 19.02–23.03.2020": crash, "Отскок до конца 2020": rebound}).sort_values("Падение 19.02–23.03.2020")
fig = go.Figure()
fig.add_bar(y=covid.index, x=covid.iloc[:, 0], orientation="h", name=covid.columns[0], marker_color=DOWN)
fig.add_bar(y=covid.index, x=covid.iloc[:, 1], orientation="h", name=covid.columns[1], marker_color=UP)
fig.update_layout(barmode="relative", xaxis_tickformat=".0%")
show(fig, "Кризис COVID-19 по секторам", height=520)

# Ширина рынка: доля акций выше своей 200-дневной средней
sma200 = prices.rolling(200, min_periods=200).mean()
valid = sma200.notna()
breadth = ((prices > sma200) & valid).sum(axis=1) / valid.sum(axis=1)
breadth = breadth[valid.sum(axis=1) > 50]
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_scatter(x=breadth.index, y=breadth.rolling(5).mean(), name="Доля акций выше SMA-200", fill="tozeroy",
                line=dict(color="#0d9488", width=1), fillcolor="rgba(13,148,136,0.18)")
fig.add_scatter(x=index.index, y=index, name="S&P 500", line=dict(color=INK, width=1.5), secondary_y=True)
fig.update_yaxes(tickformat=".0%", range=[0, 1], secondary_y=False)
fig.update_yaxes(showgrid=False, secondary_y=True)
show(fig, "Ширина рынка", "Когда растёт индекс, а доля сильных акций падает — рост держится на немногих лидерах")

# Кластеризация акций по корреляции
recent = returns[returns.index >= returns.index[-1] - pd.Timedelta(days=3 * 365)]
liquid = recent.columns[recent.notna().mean() > 0.95]
corr = recent[liquid].corr()
order = leaves_list(linkage(squareform(1 - corr.values, checks=False), "average"))
ordered = corr.iloc[order, order]
show(px.imshow(ordered, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"),
     "Карта корреляций акций за 3 года", "Похожие акции сгруппированы рядом — видно «семейства» внутри рынка", height=800)
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().sort_values()
pair_tbl = pd.DataFrame(
    [(f"{a} & {b}", v, sector_of.get(a), sector_of.get(b)) for (a, b), v in list(pairs.tail(6).iloc[::-1].items()) + list(pairs.head(6).items())],
    columns=["Пара", "Корреляция", "Сектор 1", "Сектор 2"])
table(pair_tbl, num=["Корреляция"], grad=["Корреляция"], cmap="RdBu_r", index=False,
      caption="👯 Самые похожие пары и 🧲 лучшие пары для диверсификации")

# =====================================================================
#  10. Стратегии: проверка на истории
# =====================================================================
header("🧪 Бэктест стратегий", "Как сработали бы простые правила с 2015 года")
signal = (index > index.rolling(200).mean()).shift(1).reindex(idx_ret.index).fillna(False)
trend_ret = idx_ret.where(signal, RISK_FREE / TRADING_DAYS)

monthly_px = prices.resample("ME").last()
momentum = monthly_px.shift(1) / monthly_px.shift(12) - 1  # импульс 12-1 мес.
next_ret = monthly_px.pct_change(fill_method=None).shift(-1)
mom_ret, ew_ret = {}, {}
for d in monthly_px.index:
    m = momentum.loc[d].dropna()
    nr = next_ret.loc[d]
    if len(m) >= 30 and nr.notna().sum() > 0:
        top = m.nlargest(10).index
        mom_ret[d + pd.offsets.MonthEnd(1)] = nr[top].mean()
        ew_ret[d + pd.offsets.MonthEnd(1)] = nr.mean()
mom_ret, ew_ret = pd.Series(mom_ret).dropna(), pd.Series(ew_ret).dropna()
idx_m = index.resample("ME").last().pct_change()
start_bt = max(mom_ret.index.min(), idx_m.dropna().index.min())
strategies = {
    "Купить и держать S&P 500": idx_m[idx_m.index >= start_bt],
    "Тренд: в рынке, только если выше SMA-200": (1 + trend_ret).resample("ME").prod().sub(1).loc[lambda s: s.index >= start_bt],
    "Импульс: 10 самых растущих акций": mom_ret[mom_ret.index >= start_bt],
    "Все акции поровну": ew_ret[ew_ret.index >= start_bt],
}
fig = go.Figure()
for i, (name, r) in enumerate(strategies.items()):
    c = 1000 * (1 + r.fillna(0)).cumprod()
    fig.add_scatter(x=c.index, y=c, name=name, line=dict(width=3 if i == 0 else 2, color=[INK, "#f59e0b", "#db2777", "#0d9488"][i]))
fig.update_yaxes(type="log", tickprefix="$")
show(fig, "Рост $1000 по разным стратегиям", "Помесячная ребалансировка, без комиссий и налогов", height=560)
table(pd.DataFrame({k: curve_stats(v.dropna()) for k, v in strategies.items()}).T,
      pct=["CAGR", "Волатильность", "Макс. просадка"], num=["Шарп"], money=["Итог $1000"], grad=["CAGR", "Шарп"],
      caption="Итоги стратегий")
print("  ⚠️ В выгрузке только нынешние участники индекса (ошибка выжившего) — результаты стратегий по акциям завышены.")

# =====================================================================
#  11. Оптимальный портфель (Марковиц)
# =====================================================================
header("🎯 Оптимальный портфель", "8000 случайных портфелей из крупнейших акций за 5 лет")
universe = [s for s in companies.sort_values("Marketcap", ascending=False)["Symbol"] if s in liquid][:15]
r5 = returns.loc[returns.index >= returns.index[-1] - pd.Timedelta(days=5 * 365), universe].dropna()
mu, cov = r5.mean() * TRADING_DAYS, r5.cov() * TRADING_DAYS
rng = np.random.default_rng(42)
W = rng.dirichlet(np.ones(len(universe)) * 0.5, 8000)
port_ret = W @ mu.values
port_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov.values, W))
sharpe = (port_ret - RISK_FREE) / port_vol
best, safest = sharpe.argmax(), port_vol.argmin()
fig = go.Figure(go.Scattergl(x=port_vol, y=port_ret, mode="markers", marker=dict(color=sharpe, colorscale="Viridis", size=4,
                                                                                    opacity=0.6, colorbar=dict(title="Шарп")),
                             name="Портфели", hoverinfo="skip"))
fig.add_scatter(x=np.sqrt(np.diag(cov)), y=mu, mode="markers+text", text=universe, textposition="top center",
                marker=dict(color=INK, size=7), name="Отдельные акции")
fig.add_scatter(x=[port_vol[best]], y=[port_ret[best]], mode="markers", marker=dict(size=22, color=DOWN, symbol="star"), name="Макс. Шарп")
fig.add_scatter(x=[port_vol[safest]], y=[port_ret[safest]], mode="markers", marker=dict(size=18, color=ACCENT, symbol="diamond"), name="Мин. риск")
fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="Риск", yaxis_title="Доходность")
show(fig, "Граница эффективности", height=600)

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}] * 2])
for i, (k, label) in enumerate([(best, "Макс.<br>Шарп"), (safest, "Мин.<br>риск")]):
    wts = pd.Series(W[k], index=universe)
    wts = pd.concat([wts[wts >= 0.03], pd.Series({"Прочие": wts[wts < 0.03].sum()})])
    fig.add_pie(labels=wts.index, values=wts.values, hole=0.55, row=1, col=i + 1, textinfo="label+percent",
                textposition="inside", insidetextorientation="horizontal", sort=False,
                title=dict(text=f"<b>{label}</b>", font=dict(size=18)))
show(fig, "Состав оптимальных портфелей",
     f"Макс. Шарп: {port_ret[best]:.0%} в год при риске {port_vol[best]:.0%} · Мин. риск: {port_ret[safest]:.0%} при {port_vol[safest]:.0%}",
     height=480)
print("  ⚠️ Оптимизация по прошлым данным — будущие доходности могут сильно отличаться.")

# =====================================================================
#  12. Монте-Карло
# =====================================================================
header("🎲 Сценарии S&P 500 на год вперёд", "5000 симуляций из реальных прошлых дней (бутстрэп)")
sims = rng.choice(idx_ret.values, size=(5000, TRADING_DAYS), replace=True)
paths = index.iloc[-1] * np.cumprod(1 + sims, axis=1)
future = pd.bdate_range(index.index[-1] + pd.Timedelta(days=1), periods=TRADING_DAYS)
q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
fig = go.Figure()
fig.add_scatter(x=index.index[-500:], y=index.iloc[-500:], name="История", line=dict(color=INK))
for k in rng.choice(len(paths), 30, replace=False):
    fig.add_scatter(x=future, y=paths[k], line=dict(width=0.5, color="rgba(37,99,235,0.25)"), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=future, y=q[4], line=dict(width=0), showlegend=False)
fig.add_scatter(x=future, y=q[0], fill="tonexty", fillcolor="rgba(37,99,235,0.12)", line=dict(width=0), name="5–95%")
fig.add_scatter(x=future, y=q[3], line=dict(width=0), showlegend=False)
fig.add_scatter(x=future, y=q[1], fill="tonexty", fillcolor="rgba(37,99,235,0.30)", line=dict(width=0), name="25–75%")
fig.add_scatter(x=future, y=q[2], name="Медиана", line=dict(color=ACCENT, dash="dash", width=2.5))
final = paths[:, -1] / index.iloc[-1] - 1
show(fig, "Веер сценариев", f"Медиана {np.median(final):+.0%} · вероятность роста {(final > 0).mean():.0%} · "
     f"5% худших: {np.percentile(final, 5):+.0%} · 5% лучших: {np.percentile(final, 95):+.0%}", height=560)
fig = px.histogram(x=final, nbins=80, color=np.where(final > 0, "рост", "падение"),
                   color_discrete_map={"рост": UP, "падение": DOWN})
fig.update_xaxes(tickformat=".0%", title="Изменение за год")
show(fig, "Распределение итогов через год")

# =====================================================================
#  13. Технический анализ
# =====================================================================
def rsi(p, n=14):
    d = p.diff()
    up, down = d.clip(lower=0).ewm(alpha=1 / n).mean(), -d.clip(upper=0).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / down)


def analyze_ticker(sym, days=730, to_report=True):
    """Свечи, SMA 50/200, Боллинджер, MACD, объём и RSI. Пример: analyze_ticker('NFLX')"""
    sym = sym.upper()
    d = stocks[stocks["Symbol"] == sym].set_index("Date").sort_index()
    if d.empty:
        print(f"  Нет цен по {sym}. Доступные тикеры: {', '.join(prices.columns[:30])}… (всего {prices.shape[1]})")
        return
    d["SMA50"], d["SMA200"] = d["Close"].rolling(50).mean(), d["Close"].rolling(200).mean()
    mid, sd = d["Close"].rolling(20).mean(), d["Close"].rolling(20).std()
    d["BB_up"], d["BB_low"] = mid + 2 * sd, mid - 2 * sd
    d["RSI"] = rsi(d["Close"])
    d["MACD"] = d["Close"].ewm(span=12).mean() - d["Close"].ewm(span=26).mean()
    d["Signal"] = d["MACD"].ewm(span=9).mean()
    cross = np.sign(d["SMA50"] - d["SMA200"]).diff()
    d = d[d.index >= d.index[-1] - pd.Timedelta(days=days)]
    cross = cross.loc[d.index]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.52, 0.14, 0.17, 0.17], vertical_spacing=0.025)
    fig.add_scatter(x=d.index, y=d["BB_up"], line=dict(width=0), showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=d.index, y=d["BB_low"], fill="tonexty", fillcolor="rgba(148,163,184,0.15)", line=dict(width=0),
                    name="Полосы Боллинджера", row=1, col=1)
    fig.add_candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=sym,
                        increasing_line_color=UP, decreasing_line_color=DOWN, row=1, col=1)
    for col, color in [("SMA50", "#f59e0b"), ("SMA200", "#7c3aed")]:
        fig.add_scatter(x=d.index, y=d[col], name=col, line=dict(color=color, width=1.6), row=1, col=1)
    for dt in cross[cross > 0].index:
        fig.add_annotation(x=dt, y=d.loc[dt, "SMA50"], text="✚ золотой крест", font=dict(color=UP), arrowcolor=UP, row=1, col=1)
    for dt in cross[cross < 0].index:
        fig.add_annotation(x=dt, y=d.loc[dt, "SMA50"], text="✖ крест смерти", font=dict(color=DOWN), arrowcolor=DOWN, ay=40, row=1, col=1)
    fig.add_bar(x=d.index, y=d["Volume"], name="Объём", row=2, col=1,
                marker_color=np.where(d["Close"] >= d["Open"], "rgba(22,163,74,0.5)", "rgba(220,38,38,0.5)"))
    hist = d["MACD"] - d["Signal"]
    fig.add_bar(x=d.index, y=hist, name="MACD гист.", marker_color=np.where(hist >= 0, UP, DOWN), row=3, col=1)
    fig.add_scatter(x=d.index, y=d["MACD"], name="MACD", line=dict(color=ACCENT, width=1.2), row=3, col=1)
    fig.add_scatter(x=d.index, y=d["Signal"], name="Сигнал", line=dict(color="#f59e0b", width=1.2), row=3, col=1)
    fig.add_scatter(x=d.index, y=d["RSI"], name="RSI", line=dict(color="#0f766e"), row=4, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220,38,38,0.08)", line_width=0, row=4, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(22,163,74,0.08)", line_width=0, row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11)), margin=dict(t=130))
    last = d.iloc[-1]
    trend = "восходящий 📈" if last["SMA50"] > last["SMA200"] else "нисходящий 📉"
    state = "перекуплена" if last["RSI"] > 70 else "перепродана" if last["RSI"] < 30 else "нейтральна"
    show(fig, f"{sym} — {name_of.get(sym, '')}",
         f"${last['Close']:.2f} · тренд {trend} · RSI {last['RSI']:.0f} ({state}) · {sector_of.get(sym, '')}",
         height=850, to_report=to_report)
    if sym in stats.index:
        s = stats.loc[sym]
        print(f"  {sym}: CAGR {s['CAGR']:.1%}, волатильность {s['Волатильность']:.1%}, бета {s['Бета']:.2f}, "
              f"макс. просадка {s['Макс. просадка']:.1%}")


def compare(*symbols, start="2020-01-01"):
    """Сравнение нескольких акций с индексом. Пример: compare('MSFT', 'NVDA', 'GOOG')"""
    return invest(["S&P500", *symbols], start=start)


header("🔍 Технический анализ", "Пример для акции с лучшим соотношением доходность/риск")
analyze_ticker(ranked["Шарп"].idxmax())

# =====================================================================
#  14. Всплески торговой активности
# =====================================================================
header("📢 Всплески торговой активности", "Дни, когда объём торгов был в десятки раз выше обычного")
vol_ratio = volume / volume.rolling(60).mean().shift(1)
spikes = vol_ratio.stack().nlargest(12)
spk = pd.DataFrame([(d.strftime("%d.%m.%Y"), s, name_of.get(s, s), v, returns.loc[d, s]) for (d, s), v in spikes.items()],
                   columns=["Дата", "Тикер", "Компания", "Объём к норме, ×", "Изменение цены"])
table(spk, num=["Объём к норме, ×"], pct=["Изменение цены"], grad=["Изменение цены"], index=False, caption="Топ всплесков")

# =====================================================================
#  15. Интерактивная панель (только в Colab/Jupyter)
# =====================================================================
if IN_NOTEBOOK:
    try:
        import ipywidgets as widgets
        header("🕹 Интерактивная панель", "Выберите тикер и период — график перестроится")
        widgets.interact(lambda Тикер, Лет: analyze_ticker(Тикер, days=int(365 * Лет), to_report=False),
                         Тикер=widgets.Dropdown(options=sorted(prices.columns), value=ranked["CAGR"].idxmax()),
                         Лет=widgets.FloatSlider(value=2, min=0.5, max=10, step=0.5))
    except ImportError:
        pass

# =====================================================================
#  16. Главные выводы
# =====================================================================
header("💡 Главные выводы")
best_sector = sector_stats["CAGR"].idxmax()
champion = ranked["CAGR"].idxmax()
insights = [
    f"S&P 500 вырос на {full - 1:.0%} за {len(yearly)} лет — это {idx_cagr:.1%} в год.",
    f"Худшая просадка индекса — {idx_dd.min():.0%} ({MONTHS[idx_dd.idxmin().month - 1]} {idx_dd.idxmin().year}); "
    f"коррекций больше 10% было {len(ep)}.",
    f"Пропустив всего 10 лучших дней, инвестор получил бы {miss10 - 1:+.0%} вместо {full - 1:+.0%}.",
    f"Крупные дневные движения (>4σ) случались {sigma_days} раз — рынок рискованнее, чем говорит «нормальное» распределение.",
    f"Топ-10 компаний — {top10_weight:.0%} индекса; лидер — {top10.iloc[0]['Shortname']} ({top10.iloc[0]['Weight']:.1%}).",
    f"Лучший сектор — {best_sector} ({sector_stats.loc[best_sector, 'CAGR']:.1%} в год), но лидеры секторов меняются каждый год.",
    f"Акция-чемпион — {champion} ({name_of.get(champion, '')}): {ranked.loc[champion, 'CAGR']:.0%} в год.",
    f"Индекс по доходности обогнали {(ranked['CAGR'] > idx_cagr).mean():.0%} акций с историей от 3 лет.",
    f"Исторически лучший месяц — {by_month['Средняя'].idxmax()}, худший — {by_month['Средняя'].idxmin()}.",
]
for s in insights:
    print("  •", s)
if IN_NOTEBOOK:
    display(HTML("<div style='background:#f8fafc;border-left:4px solid #2563eb;padding:12px 18px;border-radius:8px;"
                 "font:14px/1.7 Inter,Arial'>" + "<br>".join(f"• {s}" for s in insights) + "</div>"))

# =====================================================================
#  17. Экспорт: Excel + оформленный HTML-отчёт
# =====================================================================
with pd.ExcelWriter(os.path.join(OUT_DIR, "sp500_report.xlsx")) as xw:
    stats.sort_values("CAGR", ascending=False).to_excel(xw, sheet_name="Акции")
    sector_stats.to_excel(xw, sheet_name="Секторы (цены)")
    sectors.to_excel(xw, sheet_name="Секторы (компании)")
    sec_year.to_excel(xw, sheet_name="Секторы по годам")
    grid.to_excel(xw, sheet_name="Сезонность")
    ep.to_excel(xw, sheet_name="Коррекции", index=False)
    yearly.to_frame("доходность").to_excel(xw, sheet_name="По годам")
    companies.drop(columns="Longbusinesssummary").to_excel(xw, sheet_name="Компании", index=False)

CSS = f"""
body{{font-family:Inter,'Segoe UI',Roboto,Arial;color:{INK};background:#f8fafc;margin:0}}
main{{max-width:1200px;margin:auto;padding:24px}}
.hero{{background:linear-gradient(120deg,#1e3a8a,#2563eb);color:white;padding:36px 32px;border-radius:18px}}
.hero h1{{margin:0 0 6px;font-size:34px}} .hero p{{margin:0;opacity:.85}}
.insights{{background:white;border-radius:14px;padding:18px 26px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
h2{{margin:36px 0 4px;font-size:24px}} .note{{color:{MUTED};margin:0 0 10px}}
.plotly-graph-div,table{{background:white;border-radius:14px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin:14px 0}}
table{{padding:8px}} footer{{color:{MUTED};font-size:12px;margin:40px 0}}
"""
with open(os.path.join(OUT_DIR, "sp500_report.html"), "w", encoding="utf-8") as f:
    f.write(f"<html><head><meta charset='utf-8'><title>S&P 500 — аналитика</title>"
            f"<script src='https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js'></script><style>{CSS}</style></head><body><main>"
            f"<div class='hero'><h1>📈 Аналитика S&P 500</h1><p>{prices.shape[1]} акций · {len(companies)} компаний · "
            f"{index.index.min():%d.%m.%Y} — {index.index.max():%d.%m.%Y}</p></div>"
            f"<div class='insights'><h3>💡 Главное</h3><ul>{''.join(f'<li>{s}</li>' for s in insights)}</ul></div>"
            + "\n".join(report)
            + "<footer>Учебный анализ исторических данных, не инвестиционная рекомендация.</footer></main></body></html>")

print(f"\n✅ Готово! Файлы в {OUT_DIR}: sp500_report.xlsx, sp500_report.html")
print("   Попробуйте: analyze_ticker('NFLX') · compare('MSFT', 'NVDA', 'GOOG') · radar('NVDA', 'AMZN', 'TSLA')")
print("              dca('NVDA', monthly=200) · screener(min_cagr=0.2, sector='Технологии') · invest(['TSLA'], start='2019-01-02')")
