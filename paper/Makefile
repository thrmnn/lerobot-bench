# Build the lerobot-bench arxiv writeup.
# Standard incantation: pdflatex -> bibtex -> pdflatex -> pdflatex.
# The double-final-pass resolves cross-references (table/figure
# numbers, bibliography back-references) that the first post-bibtex
# run cannot place yet.

LATEX ?= pdflatex
BIBTEX ?= bibtex
LATEXOPTS ?= -interaction=nonstopmode -halt-on-error

PAPER = main

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(LATEXOPTS) $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(LATEX) $(LATEXOPTS) $(PAPER).tex
	$(LATEX) $(LATEXOPTS) $(PAPER).tex

view: $(PAPER).pdf
	@command -v xdg-open >/dev/null && xdg-open $(PAPER).pdf || echo "open $(PAPER).pdf manually"

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).pdf $(PAPER).toc $(PAPER).fls \
	      $(PAPER).fdb_latexmk
