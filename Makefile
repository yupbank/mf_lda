sgd: sgd.cpp util.cpp util.h sgd_component.h sgd_component.cpp
	g++ -std=c++11 sgd.cpp sgd_component.cpp util.cpp -o sgd

lda: lda.cpp util.cpp lda_component.cpp lda_component.h util.h
	g++ -std=c++11 lda.cpp util.cpp lda_component.cpp -o lda

hft: hft.cpp lda_component.cpp util.cpp lda_component.h util.h sgd_component.h sgd_component.cpp hft.h
	g++ -std=c++11 hft.cpp lda_component.cpp sgd_component.cpp util.cpp -o hft
