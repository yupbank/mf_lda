#ifndef LDA_INI 
#define LDA_INI
#include "util.h"
namespace lda
{
     static int num_topic = 5;
     static double alpha = 50.0/num_topic;
     static double beta = 0.01;
     static int steps = 5;
     static int twords = 50;



    std::string get_word(int word_id);

    int a_main(int, const char *[]);

    int get_topic();

    int get_topic(int m, int word, int num_word, int ** & doc_topic, int ** & topic_term, int * & doc_sum, int * & topic_sum );

    void save_model(double ** &phi, util::mapi &);


    void load_data(util::matrix & docs, util::matrix & z_docs, util::maps &, util::mapi &, util::maps &, util::mapi &, std::string file_name="item_corpus");

    
    void lda_model(const util::matrix & docs, util::matrix & z, int ** &doc_topic, int ** &topic_term, int * &doc_sum, int * &topic_sum, const int num_topic, const int num_doc, const int num_term, double ** & theta, double ** &phi);
    
    struct sort_pred {
        bool operator()(const std::pair<int,double> &left, const std::pair<int,double> &right) {
            return left.second > right.second;
        }
};
}
#endif
