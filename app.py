from flask import Flask, render_template, request
from sympy import symbols
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
import sys

# Burada kendi nonlinear_solver modülünüzdeki fonksiyonları import edin:
from nonlinear_solver import compute_Wmax, compute_WFPs, solve_gamma_range, generate_plots

app = Flask(__name__)

# --- Yardımcı Fonksiyon: Unicode/Özel Karakterleri Temizleme ve "Add objekt is not callable" Sorununu Önleme ---
def convert_to_sympy(raw_text):
    """
    raw_text: Kullanıcının 'γ₁ = ½ + (W / c₁)·[ ... ]' gibi denklemi olduğu haliyle yapıştırdığı metin.
    İstenen dönüşümler:
      1) Unicode→ASCII: γ₁→gamma1, c₁→c1, ½→1/2, ·→*, θ→theta, vs.
      2) Köşeli parantezleri normal paranteze dönüştürme: [ → (, ] → )
      3) İki parantezin yan yana gelmesi durumunda “)(” → “)*("
    4) Her satır “LHS = RHS” formatından “fX = (LHS) - (RHS)” formatına dönüştürme.
    """
    lines = raw_text.strip().splitlines()
    sympy_lines = []

    # 1) En başa sembol tanımı (projenizde kullanılan tüm semboller)
    symbols_line = (
        "gamma1, gamma2, gamma3, W, c1, c2, c3, theta = "
        "symbols('gamma1 gamma2 gamma3 W c1 c2 c3 theta')"
    )
    sympy_lines.append(symbols_line)

    # 2) Unicode/özel karakter eşleştirmeleri
    replacements = {
        'γ₁': 'gamma1',
        'γ₂': 'gamma2',
        'γ₃': 'gamma3',
        'c₁': 'c1',
        'c₂': 'c2',
        'c₃': 'c3',
        'θ':  'theta',

        '½':  '1/2',
        '–':  '-',   # kısa tire (en dash)
        '—':  '-',   # uzun tire (em dash)
        '−':  '-',   # gerçek eksi işareti (minus sign)
        '·':  '*',   # nokta → çarpı

        '¹':  '1',
        '²':  '2',
        '³':  '3',
        '⁴':  '4',
        '⁵':  '5',
        '⁶':  '6',
        '⁷':  '7',
        '⁸':  '8',
        '⁹':  '9',

        '[': '(',
        ']': ')',
        '{': '(',
        '}': ')'
    }

    # Sympy parse işlemi için gerekli dönüşümler
    transformations = (standard_transformations + (implicit_multiplication_application,))

    idx_f = 1
    idx_g = 1

    for i, raw_line in enumerate(lines):
        satir = raw_line.strip()
        if not satir or satir.startswith('#'):
            # Boş satır veya yorum satırıysa atla
            continue

        # 3) Her satırdaki özel karakterleri ASCII karşılığına dönüştür
        for uni, asc in replacements.items():
            satir = satir.replace(uni, asc)

        # 4) İki parantezin yan yana geldiği durumları çarpma işaretli hale getir
        #    Bu satırı mutlaka replacements'ten sonra koymalısınız:
        satir = satir.replace(')(', ')*(')

        # 5) Satırda mutlaka '=' karakteri olmalı
        if '=' not in satir:
            raise ValueError(f"Satır {i+1}: '=' karakteri bulunamadı: '{raw_line}'")

        lhs_text, rhs_text = satir.split('=', 1)
        lhs_text = lhs_text.strip()
        rhs_text = rhs_text.strip()

        # 6) Sympy parse kontrolü (hatalı bir ifade varsa buradan yakalayalım)
        try:
            parse_expr(lhs_text, transformations=transformations)
            parse_expr(rhs_text, transformations=transformations)
        except Exception as e:
            raise ValueError(f"Satır {i+1}: parse hatası ({e}) → '{raw_line}'")

        # 7) Denklem adı ataması: ilk üç satır f1,f2,f3; sonraki üç satır g1,g2,g3
        if idx_f <= 3:
            eq_name = f"f{idx_f}"
            idx_f += 1
        else:
            eq_name = f"g{idx_g}"
            idx_g += 1

        # 8) “fX = (LHS) - (RHS)” biçimine dönüştür
        sympy_line = f"{eq_name} = ({lhs_text}) - ({rhs_text})"
        sympy_lines.append(sympy_line)

    return "\n".join(sympy_lines)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1) Kullanıcının “Normal Denklem Sistemi” alanına girdiği metin
        raw_eq_text = request.form.get('raw_equations', '').strip()
        # 2) (Opsiyonel) Kullanıcının direkt Sympy formatında ASCII girmesi
        eq_text_direct = request.form.get('equations', '').strip()

        # 3) Hangi alandan geldiğine göre tercih et (raw_equations öncelikli)
        if raw_eq_text:
            try:
                eq_text = convert_to_sympy(raw_eq_text)
            except Exception as e:
                return f"Denklemler Sympy formatına çevrilirken hata: {e}", 400
        else:
            eq_text = eq_text_direct

        if not eq_text:
            return "Lütfen denklem sistemini girin.", 400

        # 4) Parametreleri alıp sayıya çevir
        c1_str = request.form.get('c1', '').strip()
        c2_str = request.form.get('c2', '').strip()
        c3_str = request.form.get('c3', '').strip()
        theta_str = request.form.get('theta', '').strip()
        try:
            c1_val = float(c1_str)
            c2_val = float(c2_str)
            c3_val = float(c3_str)
            theta_val = float(theta_str)
        except ValueError:
            return "c₁, c₂, c₃ ve θ sayısal olmalı.", 400

        if not (c1_val < c2_val < c3_val):
            return "Lütfen c₁ < c₂ < c₃ olacak şekilde değer girin.", 400
        if not (0.5 < theta_val < 1.0):
            return "Lütfen 0.5 < θ < 1 aralığında bir değer girin.", 400

        # 5) Sympy sembollerini tanımla
        gamma1, gamma2, gamma3, W, c1, c2, c3, theta = symbols(
            'gamma1 gamma2 gamma3 W c1 c2 c3 theta'
        )

        # 6) exec ile denklemleri değerlendirme
        locals_dict = {
            'symbols': symbols,
            'gamma1': gamma1, 'gamma2': gamma2, 'gamma3': gamma3,
            'W': W, 'c1': c1, 'c2': c2, 'c3': c3, 'theta': theta
        }
        try:
            exec(eq_text, {}, locals_dict)
        except Exception as e:
            return f"Denklemler parse edilirken hata: {e}", 400

        # 7) f1,f2,f3,g1,g2,g3 var mı kontrol et
        required_keys = ['f1', 'f2', 'f3', 'g1', 'g2', 'g3']
        missing = [k for k in required_keys if k not in locals_dict]
        if missing:
            return f"Denkliklerde eksik isimlendirme var: {missing}", 400

        # 8) referansları al
        f1 = locals_dict['f1']
        f2 = locals_dict['f2']
        f3 = locals_dict['f3']
        g1 = locals_dict['g1']
        g2 = locals_dict['g2']
        g3 = locals_dict['g3']

        # 9) Hesaplamalar: Wmax ve gamma_at_Wmax
        Wmax_val, gamma_at_Wmax = compute_Wmax(
            c1_val, c2_val, c3_val, theta_val,
            f1, f2, f3, initial_guess=(0.5, 0.5, 0.5)
        )

        # 10) W_FP, W_FP1 ve W_FP_min
        WFP_val, WFP1_val, WFPmin_val = compute_WFPs(
            c1_val, c2_val, c3_val, theta_val,
            Wmax_val, gamma_at_Wmax,
            g1, g2, g3, initial_guess=None
        )

        # 11) γ eğrilerini hesapla
        W_vals1, gamma_vals1 = solve_gamma_range(
            0, Wmax_val, 100,
            c1_val, c2_val, c3_val, theta_val,
            (f1, f2, f3), gamma_at_Wmax
        )
        if WFPmin_val is None:
            WFPmin_val = Wmax_val + 1.0
        W_vals2, gamma_vals2 = solve_gamma_range(
            Wmax_val, WFPmin_val, 100,
            c1_val, c2_val, c3_val, theta_val,
            (g1, g2, g3), gamma_at_Wmax
        )

        # 12) Grafik oluşturma
        plot1_b64, plot2_b64 = generate_plots(
            W_vals1, gamma_vals1, W_vals2, gamma_vals2,
            Wmax_val, WFPmin_val
        )

        # 13) Sonuç sayfasını render et
        return render_template(
            'results.html',
            c1=c1_val, c2=c2_val, c3=c3_val, theta=theta_val,
            Wmax=Wmax_val, WFP=WFP_val, WFP1=WFP1_val, WFPmin=WFPmin_val,
            plot1_data=plot1_b64, plot2_data=plot2_b64
        )

    # GET isteği: formu göster
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
