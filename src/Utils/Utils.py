import markdown
from weasyprint import HTML, CSS
from datetime import datetime
import os


def convert_to_pdf_advanced(markdown_text, output_path, mode="study"):
    """
    mode options:
    - study
    - resume
    - corporate
    """

    html_body = markdown.markdown(
        markdown_text,
        extensions=[
            "markdown.extensions.tables",
            "fenced_code",
            "toc",
            "attr_list"
        ]
    )

    # Mode-based themes
    theme_styles = {
        "study": """
            body { font-family: "Segoe UI", sans-serif; }
            h1, h2 { color: #1a237e; }
        """,
        "resume": """
            body { font-family: Georgia, serif; }
            h1 { font-size: 28px; }
            h2 { border-bottom: 1px solid #ddd; }
        """,
        "corporate": """
            body { font-family: Arial, sans-serif; }
            h1, h2 { color: #0d47a1; }
            table th { background: #e3f2fd; }
        """
    }

    selected_theme = theme_styles.get(mode, theme_styles["study"])

    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
    </head>
    <body>

        <!-- COVER -->
        <div class="cover">
            <h1>Interview Preparation Report</h1>
            <p>Generated on {datetime.now().strftime("%d %B %Y")}</p>
        </div>

        <div class="page-break"></div>

        <!-- TABLE OF CONTENTS -->
        <div class="toc">
            <h2>Table of Contents</h2>
            <div class="toc-content"></div>
        </div>

        <div class="page-break"></div>

        <!-- MAIN CONTENT -->
        {html_body}

    </body>
    </html>
    """

    css = CSS(string=f"""
        @page {{
            size: A4;
            margin: 2cm;

            @bottom-center {{
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10px;
                color: #555;
            }}
        }}

        body {{
            line-height: 1.6;
            color: #2c3e50;
        }}

        .cover {{
            text-align: center;
            margin-top: 200px;
        }}

        .page-break {{
            page-break-after: always;
        }}

        /* Auto TOC */
        .toc-content::before {{
            content: target-counter(attr(href), page);
        }}

        a {{
            color: #1565c0;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        pre {{
            background: #1e1e1e;
            color: #f8f8f2;
            padding: 12px;
            border-radius: 6px;
        }}

        code {{
            background: #f4f4f4;
            padding: 3px 6px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}

        {selected_theme}
    """)

    HTML(string=full_html).write_pdf(output_path, stylesheets=[css])
