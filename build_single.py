"""Rebuilds 'single code.html': one fully self-contained file (CSS, JS, Chart.js and fonts inlined).
Run after editing index.html / styles.css / app.js:   python build_single.py
"""
import base64, re
from pathlib import Path

root = Path(__file__).parent
html = (root / "index.html").read_text(encoding="utf-8")
css = (root / "styles.css").read_text(encoding="utf-8")
js = (root / "app.js").read_text(encoding="utf-8")
data = (root / "data" / "icu-data.js").read_text(encoding="utf-8")
chart = (root / "vendor" / "chart.umd.js").read_text(encoding="utf-8")

def font_data_uri(match):
    data = (root / "fonts" / match.group(1)).read_bytes()
    return "url(data:font/woff2;base64," + base64.b64encode(data).decode() + ")"

css = re.sub(r"url\(fonts/([^)]+)\)", font_data_uri, css)
html = html.replace('<link rel="stylesheet" href="styles.css">', "<style>\n" + css + "\n</style>")
html = html.replace('<script src="vendor/chart.umd.js"></script>', "<script>\n" + chart.replace("</script>", "<\\/script>") + "\n</script>")
html = html.replace('<script src="data/icu-data.js"></script>', "<script>\n" + data + "\n</script>")
html = html.replace('<script src="app.js"></script>', "<script>\n" + js.replace("</script>", "<\\/script>") + "\n</script>")
(root / "single code.html").write_text(html, encoding="utf-8")
print("Wrote 'single code.html'", round(len(html) / 1024), "KB")
