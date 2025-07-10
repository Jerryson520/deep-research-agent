import subprocess

# 调用 pandoc 转 PDF
subprocess.run(["pandoc", "report.md", "-o", "report.pdf"], check=True)
print("✅ Generated report.pdf")