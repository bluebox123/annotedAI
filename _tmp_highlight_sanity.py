import os
import fitz
import time
from simple_highlighter import SimpleHighlighter


def make_test_pdf(path: str):
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "This is a highlighting sanity test.\n"
        "If you can read this text through the highlight, it works.\n"
        "The quick brown fox jumps over the lazy dog.\n"
        "Highlighting should be visible but not obscure content."
    )
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()


def main():
    in_pdf = os.path.abspath("_tmp_input.pdf")
    make_test_pdf(in_pdf)

    targets = [(0, "Highlighting sanity test")]
    out = SimpleHighlighter().highlight_multiple_simple(in_pdf, targets, "", out_stub="sanity", max_highlights=5)
    print("input:", in_pdf)
    print("output:", out)


if __name__ == "__main__":
    main()
