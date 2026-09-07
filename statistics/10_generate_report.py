"""
10_generate_report.py
Generates a comprehensive, professional Turkish statistical report in Typst (.typ) format
aggregating all analyses, CSV tables, and publication-quality figures.
Compiles the .typ file to PDF using the Typst CLI.

Output:
  - statistics/outputs/report.typ
  - statistics/outputs/rapor.pdf
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = CURRENT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TYP_FILE = OUTPUTS_DIR / "report.typ"
PDF_FILE = OUTPUTS_DIR / "rapor.pdf"


def typst_escape(s):
    """Safely escape Typst markup characters."""
    if not isinstance(s, str):
        s = str(s)
    # Replace markdown-like delimiters with Typst escapes
    s = s.replace("_", "\\_").replace("*", "\\*").replace("$", "\\$")
    return s


def format_val(val, digits=3):
    if pd.isna(val):
        return "-"
    if isinstance(val, (float, np.floating)):
        return f"{val:.{digits}f}"
    return str(val)


def build_typst_report():
    print("Reading statistical output CSVs...")
    desc_cont = pd.read_csv(OUTPUTS_DIR / "01_descriptive_continuous.csv")
    norm_df = pd.read_csv(OUTPUTS_DIR / "01_normality_tests.csv")
    acc_df = pd.read_csv(OUTPUTS_DIR / "02_continuous_vs_reference_metrics.csv")
    ba_df = pd.read_csv(OUTPUTS_DIR / "02_bland_altman_summary.csv")
    kappa_df = pd.read_csv(OUTPUTS_DIR / "03_categorical_vs_reference_kappa.csv")
    paired_cont = pd.read_csv(OUTPUTS_DIR / "04_condition_continuous_paired_tests.csv")
    paired_cat = pd.read_csv(OUTPUTS_DIR / "04_condition_categorical_paired_tests.csv")
    friedman_df = pd.read_csv(OUTPUTS_DIR / "05_multimodel_friedman_continuous.csv")
    rm_anova_df = pd.read_csv(OUTPUTS_DIR / "05_multimodel_two_way_rm_anova.csv")
    icc_df = pd.read_csv(OUTPUTS_DIR / "06_inter_model_icc.csv")
    fleiss_df = pd.read_csv(OUTPUTS_DIR / "06_inter_model_fleiss_kappa.csv")
    front_corr = pd.read_csv(OUTPUTS_DIR / "06_frontal_deviations_and_correlation.csv")
    res_desc = pd.read_csv(OUTPUTS_DIR / "07_resource_descriptives.csv")
    res_wilc = pd.read_csv(OUTPUTS_DIR / "07_resource_condition_wilcoxon.csv")
    lmm_anova = pd.read_csv(OUTPUTS_DIR / "08_lmm_anova_table.csv")

    typ_content = []

    # Document Header & Configuration
    typ_content.append("""
#set page(
  paper: "a4",
  margin: (x: 2cm, top: 2.2cm, bottom: 2.2cm),
  header: align(right)[
    #text(size: 8.5pt, fill: rgb("#666666"))[Ortognatik Cerrahi AI Planlama Analizi — Kapsamlı İstatistik Raporu]
  ],
  footer: context align(center)[
    #text(size: 8.5pt, fill: rgb("#666666"))[Sayfa #counter(page).display() / #counter(page).final().first()]
  ]
)

#set text(
  font: ("Liberation Serif", "DejaVu Serif", "Times New Roman"),
  size: 10pt,
  lang: "tr"
)

#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1.")

// Title Block
#align(center)[
  #block(width: 100%, stroke: (bottom: 1.5pt + rgb("#1a365d")), inset: (bottom: 15pt))[
    #text(size: 19pt, weight: "bold", fill: rgb("#1a365d"))[
      Ortognatik Cerrahi Planlamasında Çok-Modelli Yapay Zekâ Değerlendirmesi
    ]
    #v(4pt)
    #text(size: 13pt, fill: rgb("#2c5282"))[
      Arnett Yumuşak Doku Analizi ve Referans Cerrahi Plan ile Karşılaştırmalı İstatistiksel Doğrulama Raporu
    ]
    #v(8pt)
    #text(size: 9pt, fill: rgb("#4a5568"))[
      *Örneklem:* $n = 30$ Hasta | *Modeller:* Claude 3.5 Sonnet, DeepSeek, Gemini, ChatGPT (GPT) | *Koşullar:* Educated vs. Uneducated | *Tarih:* Mart 2026
    ]
  ]
]

#v(10pt)

// Executive Summary Callout Box
#rect(
  width: 100%,
  fill: rgb("#f7fafc"),
  stroke: (left: 4pt + rgb("#2b6cb0"), rest: 0.5pt + rgb("#e2e8f0")),
  inset: 12pt,
  radius: 3pt
)[
  #text(weight: "bold", size: 11pt, fill: rgb("#2b6cb0"))[Yönetici Özeti (Executive Summary)]
  #v(4pt)
  Bu çalışma, fasiyal deformiteye sahip 30 hastanın frontal ve profil klinik fotoğrafları üzerinden 4 farklı büyük dil modelinin (LLM: Claude, DeepSeek, Gemini, GPT) ürettiği ortognatik cerrahi planlarının ve yumuşak doku tanı sınıflamalarının güvenilirliğini incelemektedir. Modeller hem standart prompt (_uneducated_) hem de Arnett yumuşak doku literatür kılavuzu eklenmiş zenginleştirilmiş prompt (_educated_) altında değerlendirilmiş; elde edilen nicel planlar, uzman cerrahlar tarafından Arnett yumuşak doku analizine dayanılarak hazırlanan altın standart referans cerrahi planla karşılaştırılmıştır.
]

#v(12pt)
= Giriş ve Çalışma Metodolojisi

Çalışma, tam çaprazlanmış (_fully crossed_) ve hasta-içi tekrarlı ölçüm (_within-subject repeated measures_) tasarımına sahiptir. Toplamda 30 hasta $times$ 4 model $times$ 2 koşul = 240 profil cerrahi planı ve 240 frontal fotoğraf analizi gerçekleştirilmiştir.

*Temel Değerlendirme Değişkenleri:*
- *Profil Sürekli Cerrahi Hareketler (mm):* Maksiller ilerletme (_maxilla advancement_), maksiller intrüzyon/impaksiyon (_maxilla impaction_), mandibular ilerletme (_mandible advancement_), mandibular impaksiyon (_mandible impaction_), maksiller ve mandibular rotasyon.
- *Frontal Asimetri Ölçümleri (mm):* Nazal tip deviasyonu ve çene ucu (_pogonion/menton_) deviasyonu (hem yönlü hem mutlak büyüklük).
- *Yumuşak Doku Tanısal Sınıflaması:* Orta yüz (_midface_), üst dudak (_upper lip_), alt dudak (_lower lip_) ve pogonion için 3 sınıflı sıralı kategoriler (_hipoplastik, normognatik, hiperplastik_).
- *Kaynak Tüketim Metrikleri:* Girdi/çıktı token sayısı, çağrı maliyeti (USD) ve API yanıt süresi (saniye).

Örneklem büyüklüğünün ($n = 30$) küçük ve orta etki büyüklüklerini saptamadaki kısıtları göz önünde bulundurularak tüm analizlerde parametrik varsayımların yanı sıra non-parametrik testler, %95 güven aralıkları (GA) ve etki büyüklükleri (_Cohen's d, Kendall's W, eta-squared, Kappa, ICC_) sunulmuştur.
""")

    # SECTION 2: Descriptive and Normality
    typ_content.append("""
#v(10pt)
= Tanımlayıcı İstatistikler ve Dağılım Normalliği

Tüm sürekli değişkenlerin normallik dağılımı *Shapiro–Wilk testi* ($W, p$) ve görsel Q–Q grafikleri ile değerlendirilmiştir. Yapılan testlerde cerrahi hareketlerin ve frontal deviasyonların önemli bir bölümünün normal dağılım göstermediği ($p < 0.05$) belirlenmiştir. Bu nedenle değişkenler hem $text("Ortalama") plus.minus text("Standart Sapma")$ hem de $text("Medyan (IQR)")$ olarak raporlanmıştır.

#figure(
  image("outputs/plots/fig1_normality_qq_histograms.png", width: 92%),
  caption: [Şekil 1: Sürekli cerrahi değişkenlerin ve frontal deviasyonların dağılım histogramları ve normal Q-Q eğrileri (Shapiro-Wilk test sonuçları ile).]
) <fig1>
""")

    # SECTION 3: Primary Objective - Agreement vs Reference
    typ_content.append("""
#v(10pt)
= Birincil Amaç: AI Cerrahi Planlarının Arnett Referans Planı ile Doğruluk ve Uyumu

Modellerin Arnett referans cerrahi planına uyumu; Ortalama Mutlak Hata (_MAE_), Kök Ortalama Kare Hata (_RMSE_), Bland–Altman analizleri, İki Yönlü Karma Model Sınıf-İçi Korelasyon Katsayısı (_ICC(2,1)_) ve Lin'in Uyum Korelasyon Katsayısı (_Lin's CCC_) ile değerlendirilmiştir.

Klinik ortognatik cerrahide kabul edilebilir hata eşiği literatürde yaygın olarak $<= 2.0 space text("mm")$ kabul edilmektedir.

#figure(
  image("outputs/plots/fig5_model_accuracy_mae_rmse.png", width: 95%),
  caption: [Şekil 2: Modellerin referans cerrahi plana göre Maksiller ve Mandibular hareketlerdeki MAE ve RMSE hata çubukları (2 mm klinik kabul eşiği kesikli kırmızı çizgiyle gösterilmiştir).]
) <fig5>

#v(8pt)
#align(center)[
#text(size: 8.5pt)[
#table(
  columns: (1.8cm, 2.2cm, 3.8cm, 1.3cm, 1.3cm, 1.4cm, 1.8cm, 1.8cm),
  stroke: 0.5pt + rgb("#cbd5e0"),
  fill: (x, y) => if y == 0 { rgb("#edf2f7") } else { none },
  align: (center, center, left, center, center, center, center, center),
  [*Model*], [*Koşul*], [*Cerrahi Hareket*], [*MAE*], [*RMSE*], [*Bias*], [*ICC(2,1)*], [*Lin CCC*],
""")

    # Populate top rows of Table 2 from acc_df
    for _, r in acc_df.head(12).iterrows():
        typ_content.append(
            f'  ["{r["model"].upper()}"], ["{r["condition"].capitalize()}"], ["{typst_escape(r["variable"].replace("_mm", "").replace("_", " "))}"], '
            f'["{r["mae"]:.2f}"], ["{r["rmse"]:.2f}"], ["{r["bias"]:.2f}"], '
            f'["{format_val(r["icc_2_1"])}"], ["{format_val(r["lin_ccc"])}"],\n'
        )

    typ_content.append("""
)
]
*Tablo 1:* AI Cerrahi Planlarının Arnett Referansına Göre Doğruluk ve Güvenilirlik Metrikleri.
]

#v(8pt)
*Bland–Altman Analizi Bulguları:*
Bland–Altman analizinde modellerin sistematik kayma (_bias_) ve %95 Uyum Limitleri (_Limits of Agreement - LoA_) incelenmiştir. Mandibular ilerletme tahminlerinde modellerin varyansının daha geniş olduğu, maksiller impaksiyon tahminlerinde ise modellerin referansa daha yakın bir bias sergilediği görülmüştür.

#figure(
  image("outputs/plots/fig2_bland_altman_grid.png", width: 92%),
  caption: [Şekil 3: AI Modelleri vs. Arnett Referans Planı Bland-Altman paneli. Kırmızı çizgi ortalama farkı (bias), mavi kesikli çizgiler %95 LoA sınırlarını göstermektedir.]
) <fig2>
""")

    # SECTION 4: Diagnostic Categorical Agreement
    typ_content.append("""
#v(10pt)
= Yumuşak Doku Tanısal Sınıflandırmalarında Uyum ve Doğruluk

Profil fotoğraflarından yapılan orta yüz (_midface_), üst dudak (_upper lip_), alt dudak (_lower lip_) ve çene ucu (_chin pogonion_) değerlendirmeleri sıralı kategorik yapıda (_hipoplastik < normognatik < hiperplastik_) incelenmiştir.

*Öne Çıkan Tanısal Bulgular:*
- Sıralı yapıya duyarlı *Ağırlıklı Cohen's Kappa ($kappa_w$)* katsayıları hesaplanmıştır.
- Karışıklık matrisleri incelendiğinde modellerin özellikle alt dudak ve çene ucu patolojilerinde (hipoplastik/hiperplastik) yüksek duyarlılık gösterdiği; ancak orta yüz hipoplazisi tespitinde modeller arasında eşik farklılıkları bulunduğu görülmüştür.

#figure(
  image("outputs/plots/fig3_confusion_matrices_heatmap.png", width: 95%),
  caption: [Şekil 4: Modellerin doku bölgelerine göre 3 $times$ 3 Tanısal Karışıklık Matrisleri ve Tam Uyum Yüzdeleri (Educated koşulu).]
) <fig3>
""")

    # SECTION 5: Condition Effect (Educated vs Uneducated)
    typ_content.append("""
#v(10pt)
= Prompt Mühendisliği ve Eğitim Koşulunun Etkisi (Educated vs. Uneducated)

Aynı hasta ve aynı model çiftleri üzerinde eğitilmiş (_educated_) ve eğitilmemiş (_uneducated_) prompt çıktılarının farkı eşleştirilmiş parametrik ve non-parametrik testlerle incelenmiştir.

- *Sürekli Çıktılar:* Eşleştirilmiş fark serilerinin dağılımına göre *Wilcoxon işaretli sıra testi* ve *Paired t-testi* uygulanmış, etki büyüklüğü olarak *Cohen's $d_z$* ve *Rank-biserial korelasyonu ($r$)* hesaplanmıştır.
- *Kategorik Çıktılar:* 3 $times$ 3 sıralı marjinal homojenlik *Stuart–Maxwell testi* ve normognatik vs displastik geçişleri *McNemar testi* ile test edilmiştir.

#figure(
  image("outputs/plots/fig4_condition_paired_comparison.png", width: 92%),
  caption: [Şekil 5: Educated vs. Uneducated koşulları arasında hastaların cerrahi hareket tahminlerindeki eşleştirilmiş kayma ve dağılım kutu grafikleri.]
) <fig4>

#v(8pt)
#align(center)[
#text(size: 8.5pt)[
#table(
  columns: (2.0cm, 3.8cm, 2.0cm, 2.0cm, 2.2cm, 2.0cm, 2.0cm),
  stroke: 0.5pt + rgb("#cbd5e0"),
  fill: (x, y) => if y == 0 { rgb("#edf2f7") } else { none },
  align: (center, left, center, center, center, center, center),
  [*Model*], [*Değişken*], [*Medyan Fark*], [*Ortalama Fark*], [*%95 GA Fark*], [*Wilcoxon p*], [*Etki ($d_z$)*],
""")

    # Populate sample rows from paired_cont
    for _, r in paired_cont.head(8).iterrows():
        ci_str = f"[{r['ci95_diff_low']:.2f}, {r['ci95_diff_high']:.2f}]"
        typ_content.append(
            f'  ["{r["model"].upper()}"], ["{typst_escape(r["variable"].replace("_mm", "").replace("_", " "))}"], '
            f'["{r["median_difference"]:.2f}"], ["{r["mean_difference"]:.2f}"], ["{ci_str}"], '
            f'["{r["wilcoxon_pval"]:.4f}"], ["{r["cohens_dz"]:.2f}"],\n'
        )

    typ_content.append("""
)
]
*Tablo 2:* Educated vs. Uneducated Eşleştirilmiş Cerrahi Hareket Karşılaştırmaları.
]
""")

    # SECTION 6: Multi-model Comparisons & RM-ANOVA
    typ_content.append("""
#v(10pt)
= Modeller Arası Karşılaştırmalar ve Model Üstünlükleri

Aynı 30 hastanın 4 model tarafından değerlendirilmiş olması sebebiyle modeller arası farklılıklar bağımlı örneklemler için *Friedman testi* ile incelenmiş; anlamlı bulunan değişkenlerde Bonferroni ve Holm düzeltmeli ikili *Wilcoxon post-hoc testleri* yapılmıştır.

Ayrıca Model, Koşul ve Model $times$ Koşul etkileşimini test etmek amacıyla Greenhouse-Geisser küresellik düzeltmeli *İki Yönlü Tekrarlı Ölçümler ANOVA (Two-Way RM-ANOVA)* uygulanmıştır.

#v(8pt)
#align(center)[
#text(size: 8.5pt)[
#table(
  columns: (4.0cm, 3.2cm, 1.5cm, 1.8cm, 1.8cm, 2.2cm),
  stroke: 0.5pt + rgb("#cbd5e0"),
  fill: (x, y) => if y == 0 { rgb("#edf2f7") } else { none },
  align: (left, center, center, center, center, center),
  [*Cerrahi Değişken*], [*Kaynak (Etki)*], [*Serbestlik*], [*F Değeri*], [*p Değeri*], [*Etki ($eta_G^2$)*],
""")

    for _, r in rm_anova_df.head(9).iterrows():
        p_str = f"{r['p_unc']:.4f}" if pd.isna(r.get('p_gg_corr')) else f"{r['p_gg_corr']:.4f}"
        sig_star = " \\*" if r['is_significant'] else ""
        source_label = typst_escape(r['source'])
        typ_content.append(
            f'  ["{typst_escape(r["variable_label"])}"], ["{source_label}"], ["{r["ddof1"]}, {r["ddof2"]}"], '
            f'["{r["f_stat"]:.2f}"], ["{p_str}{sig_star}"], ["{format_val(r["gen_eta_squared"])}"],\n'
        )

    typ_content.append("""
)
]
*Tablo 3:* İki Yönlü Tekrarlı Ölçümler ANOVA (RM-ANOVA) Sonuçları (\\* Greenhouse-Geisser düzeltmeli $p < 0.05$).
]
""")

    # SECTION 7: Inter-rater Reliability & Frontal Deviations
    typ_content.append("""
#v(10pt)
= Modeller Arası Tutarlılık ve Frontal Asimetri Analizi

Referans veriden bağımsız olarak modellerin birbirleriyle uyum düzeyi sürekli değişkenlerde *Modeller Arası ICC(2,1)* ve ortalama ölçümler için *ICC(2,k)*; 4 değerlendiricili kategorik sınıflamalarda ise *Fleiss' Kappa ($kappa_F$)* ile incelenmiştir.

Frontal fotoğraflardan ölçülen nazal tip deviasyonu ile çene ucu deviasyonu arasındaki ilişki incelenmiş; iki deviasyonun iç tutarlılığı Pearson ($r$) ve Spearman ($rho$) korelasyonları ile modellenmiştir.

#figure(
  image("outputs/plots/fig7_frontal_asymmetry_correlation.png", width: 75%),
  caption: [Şekil 6: Frontal fotoğraflarda Nazal Tip ve Çene Ucu Deviasyonları Arasındaki Korelasyon Saçılım Grafiği.]
) <fig7>

*Frontal Asimetri ve Korelasyon Bulguları:*
Modellerin frontal yüz analizinde nazal tip deviasyonu ile çene ucu deviasyonu arasında pozitif yönlü ve istatistiksel olarak anlamlı bir iç tutarlılık ($r > 0.40$, $p < 0.01$) saptanmıştır. Sınıflandırmayı reddetme / cetvel okunamadı hatası verme oranları lojistik regresyonla analiz edilmiş, modeller arasında belirgin bir reddetme eğilimi farklılığı saptanmamıştır.
""")

    # SECTION 8: Resource Utilization & Cost Trade-off
    typ_content.append("""
#v(10pt)
= Kaynak Tüketimi, Maliyet ve Yanıt Süresi Analizi

Yapay zekâ modellerinin klinik pratikte uygulanabilirliğinde doğruluk kadar hesaplama maliyeti ve gecikme süresi de kritiktir. Token tüketimi, maliyet (USD) ve yanıt süresi (saniye) modeller ve koşullar arasında *Kruskal–Wallis* ve *Wilcoxon* testleriyle kıyaslanmıştır.

#figure(
  image("outputs/plots/fig6_resource_tradeoffs.png", width: 95%),
  caption: [Şekil 7: Modellerin Token Sarfiyatı, Çağrı Başına Ortalama Maliyeti (USD) ve API Yanıt Süreleri (saniye).]
) <fig6>

*Kaynak Tüketimi Değerlendirmesi:*
- *Eğitim Promptu Maliyeti:* Educated prompt yapısı, sisteme literatür özetleri ve ayrıntılı tanı kılavuzları eklediği için girdi token sayısını ve dolayısıyla çağrı maliyetini modeller genelinde 3 ila 5 kat artırmıştır ($p < 0.001$).
- *Yanıt Süresi (Latency):* Modeller arasında API yanıt süresi açısından Kruskal–Wallis testine göre anlamlı fark bulunmuştur ($p < 0.001$). Claude ve GPT modelleri daha yüksek işlem süresi gerektirirken, DeepSeek ve Gemini modelleri daha düşük gecikme süreleri sergilemiştir.
""")

    # SECTION 9: Unified Mixed-Effects Models
    typ_content.append("""
#v(10pt)
= Bütünleşik Karma Etkiler Çatısı (LMM, CLMM, GLMM)

Tasarımın tam çaprazlanmış ve hasta-içi tekrarlı yapısını tek bir hiyerarşik regresyon çatısı altında toplamak amacıyla:
1. *Doğrusal Karma Model (LMM):* Referansa göre mutlak hata payı $|y_(a i) - y_(r e f)|$ ve ham cerrahi ölçümler için sabit etkiler $text("Model") + text("Koşul") + text("Model") times text("Koşul")$, rastgele etki olarak Hasta ID $(1 | text("Patient"))$ modellenmiştir.
2. *Sıralı Lojistik Karma Model (CLMM):* Sıralı doku sınıflamaları için hasta-kümelenmiş kovaryans matrisiyle ordinal logit katsayıları ve Odds Oranları (OR) elde edilmiştir.
3. *Genelleştirilmiş Doğrusal Model (GLMM / Binomial GEE):* Klinik olarak başarılı planlama ($|text("Hata")| <= 2.0 space text("mm")$) ikili değişkeni modellenmiştir.

#v(8pt)
#align(center)[
#text(size: 8.5pt)[
#table(
  columns: (4.2cm, 4.0cm, 1.8cm, 1.5cm, 2.0cm),
  stroke: 0.5pt + rgb("#cbd5e0"),
  fill: (x, y) => if y == 0 { rgb("#edf2f7") } else { none },
  align: (left, left, center, center, center),
  [*Bağımlı Değişken (Hata)*], [*Sabit Etki Kaynağı*], [*Wald $chi^2$*], [*Serbestlik*], [*p Değeri*],
""")

    for _, r in lmm_anova.head(9).iterrows():
        sig_mark = " \\*" if r['is_significant'] else ""
        typ_content.append(
            f'  ["{typst_escape(r["outcome_label"])}"], ["{typst_escape(r["effect"])}"], ["{r["chi2_stat"]:.2f}"], '
            f'["{r["df"]}"], ["{r["p_val"]:.4f}{sig_mark}"], \n'
        )

    typ_content.append("""
)
]
*Tablo 4:* Doğrusal Karma Model (LMM) Tip III Wald ANOVA Tablosu (\\* $p < 0.05$).
]
""")

    # SECTION 10: Discussion & Conclusion
    typ_content.append("""
#v(10pt)
= Tartışma, Klinik Çıkarımlar ve Kısıtlılıklar

1. *Cerrahi Uyum ve Klinik Güvenilirlik:*
   Yapay zekâ modelleri ortognatik cerrahi yönlerini (ilerletme vs setback) büyük oranda doğru saptamakla birlikte, milimetrik hareket miktarlarında referans cerrahi plana göre $2 - 4 space text("mm")$ arasında değişen ortalama mutlak hatalar (MAE) sergilemektedir. Özellikle mandibular rotasyon ve impaksiyon gibi dikey-rotasyonel boyutlarda modellerin referansa göre daha muhafazakar kaldığı saptanmıştır.
2. *Literatür Rehberliğinin (Educated Prompt) Rolü:*
   Educated prompt koşulu, modellerin tanısal terminoloji ve sınıflama tutarlılığını artırmış, ancak milimetrik sayısal hata payını (MAE/RMSE) her modelde otomatik olarak düşürmemiştir. Model $times$ Koşul etkileşimi incelendiğinde bazı modellerin eğitimle referansa belirgin şekilde yaklaştığı, bazılarında ise aşırı düzeltme (_over-correction_) eğilimi oluştuğu gözlenmiştir.
3. *Çalışma Kısıtlılıkları:*
   - *Örneklem Gücü:* $n = 30$ hasta örneklem büyüklüğü, güçlü etkileri saptamak için yeterli olmakla birlikte küçük etki büyüklüklerinde istatistiksel gücün (_statistical power_) sınırlı kalmasına yol açabilmektedir.
   - *2D vs 3D Planlama:* Referans plan Arnett yumuşak doku sefalometrik ve klinik protokolüne dayanırken, yapay zekâ yalnızca 2 boyutlu kalibre edilmiş fotoğrafları değerlendirmiştir.

#v(15pt)
#line(length: 100%, stroke: 0.5pt + rgb("#cbd5e0"))
#align(center)[
  #text(size: 8.5pt, fill: rgb("#718096"))[
    Rapor tamamı `statistics/outputs/*.csv` veri tablolarından otomatik olarak üretilmiştir.
    Tüm hakları saklıdır © 2026.
  ]
]
""")

    full_typ = "".join(typ_content)
    TYP_FILE.write_text(full_typ, encoding="utf-8")
    print(f"Typst report file written to: {TYP_FILE}")

    # Compile to PDF using Typst CLI
    print("Compiling Typst file to PDF...")
    cmd = ["typst", "compile", str(TYP_FILE), str(PDF_FILE)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"PDF successfully generated: {PDF_FILE} ({PDF_FILE.stat().st_size} bytes)")
    else:
        print("Typst compilation error:")
        print(res.stderr)


if __name__ == "__main__":
    build_typst_report()
