#ifndef XXX
#define XXX
#include "util.h"
const int num_topic = 5;
const double alpha = 50.0/num_topic;
const double beta = 0.01;
int steps = 50;
int twords = 50;
int num_term;
int num_doc;

double p[num_topic];

typedef std::map<std::string, int> mapword2id;
typedef std::map<int, std::string> mapid2word;

typedef std::vector< int > array;
typedef std::vector< array > matrix;

mapword2id word2id;
mapid2word id2word;


int get_word_id(std::string word);

std::string get_word(int word_id);

void lda(const matrix & docs, matrix & z, int ** &doc_topic, int ** &topic_term, int * &doc_sum, int * &topic_sum, const int num_topic, const int num_doc, const int num_term, double ** & theta, double ** &phi);

int get_topic(int K);

int get_topic(int m, int word, int num_word, int ** & doc_topic, int ** & topic_term, int * & doc_sum, int * & topic_sum );

void quicksort(std::vector<std::pair<int, double> > & vect, int left, int right) ;


void save_model(int num_term, double ** &phi);

#endif
