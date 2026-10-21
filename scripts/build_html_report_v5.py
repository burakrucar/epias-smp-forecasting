"""
HTML Rapor Derleyici (V5)
Markdown raporunu MathJax ve profesyonel akademik CSS temasıyla HTML'e çevirir.
"""

import os
import markdown

def build_html():
    md_path = "reports/SMF_Tez_Deney_ve_Tahmin_Sonuclari.md"
    html_out_desktop = "/Users/burakmac/Desktop/SMF_Tez_Deney_ve_Tahmin_Sonuclari.html"
    html_out_local = "reports/SMF_Tez_Deney_ve_Tahmin_Sonuclari.html"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Markdown to HTML conversion
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "nl2br"]
    )

    template = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMF ve Dengeleme Yönü Tahmini - Deney Sonuçları Raporu (Revizyon V5)</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <!-- MathJax 3 -->
    <script>
    window.MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true
        }},
        svg: {{
            fontCache: 'global'
        }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root {{
            --bg-main: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-muted: #475569;
            --primary: #1e3a8a;
            --primary-light: #eff6ff;
            --primary-border: #bfdbfe;
            --accent: #0284c7;
            --success: #059669;
            --success-bg: #ecfdf5;
            --success-border: #a7f3d0;
            --warning: #d97706;
            --warning-bg: #fffbeb;
            --warning-border: #fde68a;
            --border-color: #e2e8f0;
            --border-dark: #cbd5e1;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            line-height: 1.65;
            padding: 40px 20px;
            -webkit-font-smoothing: antialiased;
        }}

        .container {{
            max-width: 1280px;
            margin: 0 auto;
            background: var(--bg-card);
            padding: 56px 64px;
            border-radius: 16px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
        }}

        /* Header & Titles */
        h1 {{
            color: var(--primary);
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 8px;
            border-bottom: 3px solid var(--primary);
            padding-bottom: 16px;
        }}

        h2 {{
            color: #1e293b;
            font-size: 1.45rem;
            font-weight: 600;
            margin-top: 40px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1.5px solid var(--border-color);
            display: flex;
            align-items: center;
        }}

        h3 {{
            color: #334155;
            font-size: 1.15rem;
            font-weight: 600;
            margin-top: 24px;
            margin-bottom: 12px;
        }}

        p {{
            margin-bottom: 14px;
            color: var(--text-main);
            font-size: 0.96rem;
        }}

        hr {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 32px 0;
        }}

        ul, ol {{
            margin-left: 24px;
            margin-bottom: 16px;
            font-size: 0.95rem;
        }}

        li {{
            margin-bottom: 6px;
        }}

        /* Code & Monospace */
        code {{
            font-family: 'JetBrains Mono', monospace;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.88em;
            border: 1px solid #e2e8f0;
        }}

        pre {{
            background-color: #0f172a;
            color: #f8fafc;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 16px 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
        }}

        pre code {{
            background: none;
            color: inherit;
            padding: 0;
            border: none;
        }}

        /* Tables Styling */
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 20px 0 28px 0;
            font-size: 0.86rem;
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}

        th {{
            background-color: #f8fafc;
            color: #334155;
            font-weight: 600;
            text-align: left;
            padding: 11px 14px;
            border-bottom: 2px solid var(--border-color);
            white-space: nowrap;
        }}

        td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-color);
            color: #1e293b;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:nth-child(even) {{
            background-color: #fbfcfe;
        }}

        tr:hover {{
            background-color: #f1f5f9;
            transition: background-color 0.15s ease;
        }}

        /* Highlight classes */
        td strong, th strong {{
            color: var(--primary);
        }}

        /* Blockquotes / Callouts */
        blockquote {{
            background-color: var(--primary-light);
            border-left: 4px solid var(--primary);
            color: #1e3a8a;
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            margin: 20px 0;
            font-size: 0.94rem;
        }}

        blockquote p {{
            color: inherit;
            margin-bottom: 8px;
        }}

        blockquote p:last-child {{
            margin-bottom: 0;
        }}

        /* Badges */
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            font-size: 0.78rem;
            font-weight: 600;
            border-radius: 12px;
            margin-left: 6px;
        }}
        .badge-success {{
            background: var(--success-bg);
            color: var(--success);
            border: 1px solid var(--success-border);
        }}
        .badge-info {{
            background: var(--primary-light);
            color: var(--primary);
            border: 1px solid var(--primary-border);
        }}

        /* Print friendly */
        @media print {{
            body {{
                background: #ffffff;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                border: none;
                padding: 0;
                max-width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>
"""

    with open(html_out_desktop, "w", encoding="utf-8") as f:
        f.write(template)
    with open(html_out_local, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"HTML Raporu Başarıyla Oluşturuldu:")
    print(f"  -> Masaüstü: {html_out_desktop}")
    print(f"  -> Proje:    {html_out_local}")

if __name__ == "__main__":
    build_html()
