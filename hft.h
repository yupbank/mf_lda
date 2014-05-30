#ifndef HFT_INI
#define HFT_INI 
#include "util.h"
#include "sgd_component.h"
using namespace util;


void init_z(matrix & docs, matrix & z, int num_doc, int ** & topic_term, int * & topic_sum);
int get_topic(int item, int word, int ** & topic_term, int * & topic_sum, double ** & doc_topic_distribution);
void lda_part(sgd::instance_type &, matrix &, matrix &, int ** &, int * &, int num_doc, int topic_num, int num_term, double ** & doc_topic_probrability, double ** & topic_term_probrability, mapi & id2word, int steps=10);
void nmf_part(sgd::instance_type & instances, matrix & docs, matrix & z, double ** & user_feature, double ** & item_feature, double * & user_bias, double * item_bias, double & glob_bias, int user_num, int item_num,double ** & doc_topic_probrability, double ** & topic_term_probrability, int steps=10);
double loss(double p, double y, my_array & doc, my_array & doc_z, int item_id, double ** & doc_topic_probrability, double ** & topic_term_probrability);
void hiddenfeature2probablity(double ** & feature, double ** & doc_topic_probrability, int num_item, int topic_num);
double likelihood(my_array & doc, my_array & doc_z, int item_id, double ** & doc_topic_probrability, double ** & topic_term_probrability);
#endif
