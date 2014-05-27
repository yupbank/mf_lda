sgd: sgd.cpp util.cpp
	g++ -std=c++11 sgd.cpp util.cpp -o sgd

lda:
	g++ -std=c++11 main.cpp -o lda
