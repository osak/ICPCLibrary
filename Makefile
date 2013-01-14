.PHONY: clean

pdf: document.tex
	platex -interaction=nonstopmode -jobname=ICPCLibrary document.tex
	dvipdfmx -p a4 -l -o ICPCLibrary.pdf ICPCLibrary.dvi

clean:
	rm -f ICPCLibrary.*
