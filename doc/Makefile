MAKEFLAGS += \
	--no-builtin-rules \
	--warn-undefined-variables

REV := v2025.02

# Inkscape
INKSCAPE ?= inkscape
INKSCAPE_FLAGS ?=
INKSCAPE_FLAGS += --export-text-to-path
INKSCAPE_VERSION_1 := $(shell inkscape --version | grep -o -E 'Inkscape 1\.[0-9]+')

all_figs :=

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = secbench-core
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help
# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: serve
serve: build-html
	@cd $(BUILDDIR) \
	&& cd html \
	&& echo http://127.0.0.1:8000/ \
	&& python3 -m http.server

.PHONY: figs
figs: $(all_figs)

# Inkscape Rules
%.pdf: %.svg
ifeq ($(INKSCAPE_VERSION_1),) # Inkscape < 1.0
	$(INKSCAPE) $(INKSCAPE_FLAGS) -D -A $@ $<
else
	$(INKSCAPE) $(INKSCAPE_FLAGS) -D -o $@ $<
endif

%.png: %.svg
ifeq ($(INKSCAPE_VERSION_1),) # Inkscape < 1.0
	$(INKSCAPE) $(INKSCAPE_FLAGS) -D -d 100 -e $@ $<
else
	$(INKSCAPE) $(INKSCAPE_FLAGS) -D -d 100 -o $@ $<
endif

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
build-%: Makefile figs
	@$(SPHINXBUILD) -M $* "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

deploy: build-html
	touch _build/html/.nojekyll
	ghp-import _build/html
