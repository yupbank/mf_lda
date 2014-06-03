#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
#include <stdlib.h>  
#include <algorithm>
#include <cmath>
#include "sgd_component.h"
#include "lda_component.h"
#include "hft.h"
#include "util.h"

using namespace std;
using namespace util;

const int topic_num = 5;
const double feature_scaler = 1.0;
const double learning_rate = 0.01;
const double reg_0 = 0;
const double reg_1 = 0.05;
const double alpha = 50.0/topic_num;
const double beta = 0.01; // for lda
const int total_steps = 10;
const double reg_likelihood = 0.5;

inline double squareloss(double p, double y)
{
   return 0.5 * (p - y) * (p - y); 
}


void test(sgd::instance_type &, mapi &, mapi & test_id2item, maps & user2id, maps & item2id,  double ** & user_feature, double ** & item_feature, double * & user_bias, double * & item_bias, double glob_bias);
void load_test_data(sgd::instance_type &, mapi &, mapi &);

int main()
{
    matrix docs;
    matrix z;

    maps user2id;
    mapi id2user;

    maps item2id;
    mapi id2item;

    mapi id2word;
    maps word2id;

    sgd::instance_type instances;

    sgd::load_data(docs, instances, word2id, id2word, user2id, id2item, item2id, id2user, "train_data");


    int user_num = user2id.size();
    int item_num = item2id.size();

    int term_num = word2id.size();
    int doc_num = docs.size();

    cout<<"num doc "<<doc_num
        <<" num term "<<term_num
        <<" user length, "<<user_num 
        <<" item length, "<<item_num<<endl; 

    double ** user_feature = create_double_matrix(user_num, topic_num);
    double ** item_feature = create_double_matrix(item_num, topic_num);

    sgd::init_features(user_feature, item_feature, topic_num, user_num, item_num);

    double glob_bias = 0.0;
    double * user_bias = new double[user_num]();
    double * item_bias = new double[item_num]();

    //int ** doc_topic = create_int_matrix(num_doc, topic_num);
    int ** topic_term = create_int_matrix(topic_num, term_num);
    int *topic_sum = new int[topic_num]();

    //int *doc_sum = new int[num_doc]();
    double **doc_topic_probrability = create_double_matrix(item_num, topic_num);
    double **topic_term_probrability = create_double_matrix(topic_num, term_num);

    sgd::instance_type test_instances;
    mapi test_id2user;
    mapi test_id2item;
    load_test_data(test_instances, test_id2user, test_id2item);
    
    for (int m=0; m<doc_num; m++)
    {
        my_array inner_z;
        for (auto &word :docs[m])
        {
            int new_topic = lda::get_topic();
            topic_term[new_topic][word] += 1;
            topic_sum[new_topic] += 1;
            inner_z.push_back(new_topic);
        }
        z.push_back(inner_z);
    }
    init_z(docs, z, doc_num, topic_term, topic_sum);
    for(int i =0; i< total_steps; i++)
    {
        hiddenfeature2probablity(item_feature, doc_topic_probrability, item_num, topic_num);

        lda_part(instances, docs, z, topic_term, topic_sum, doc_num, topic_num, term_num, doc_topic_probrability, topic_term_probrability, id2word);

        nmf_part(instances, docs, z, user_feature, item_feature, user_bias, item_bias, glob_bias, user_num, item_num, doc_topic_probrability, topic_term_probrability);

        test(test_instances, test_id2user, test_id2item, user2id, item2id,  user_feature, item_feature, user_bias, item_bias, glob_bias);
    }
    return 0;
}

void load_test_data(sgd::instance_type & test_instances, mapi & test_id2user, mapi & test_id2item)
{
    matrix test_docs;

    maps test_user2id;

    maps test_item2id;

    mapi test_id2word;
    maps test_word2id;

    sgd::load_data(test_docs, test_instances, test_word2id, test_id2word, test_user2id, test_id2item, test_item2id, test_id2user, "test_data");
}

void test(sgd::instance_type & test_instances, mapi & test_id2user, mapi & test_id2item, maps & user2id, maps & item2id,  double ** & user_feature, double ** & item_feature, double * & user_bias, double * & item_bias, double glob_bias)
{
    double sumloss = 0.0;
    int count = 0;
    for (auto instance : test_instances)
    {
        int user = get<0>(instance);
        int item = get<1>(instance);
        int rating = get<2>(instance);
        string user_name = get_word(user, test_id2user);
        int user_id = get_id(user_name, user2id);
        string item_name = get_word(item, test_id2item);
        int item_id = get_id(item_name, item2id);
        if ( user_id >= 0 && item_id >= 0)
        {
            double pred = sgd::predict(user_feature, item_feature, user_id, item_id, user_bias, item_bias, glob_bias);
            sumloss += squareloss(pred, rating);
            count ++;
        }
        /*else
        {
            cout<<user_name<<" "<<item_name<<endl; 
        }*/

    }
    cout<<endl;
    printf("test rmse (%f) cout (%d) \n", sqrt(sumloss/count), count);
}

void init_z(matrix & docs, matrix & z, int doc_num, int ** & topic_term, int * & topic_sum)
{
    srandom(time(0));
    for (int m=0; m<doc_num; m++)
    {
        my_array inner_z;
        for (auto &word :docs[m])
        {
            int new_topic = lda::get_topic();
            topic_term[new_topic][word] += 1;
            topic_sum[new_topic] += 1;
            inner_z.push_back(new_topic);
        }
        z.push_back(inner_z);
    }
}

void lda_part(sgd::instance_type & instances, matrix & docs, matrix & z, int ** & topic_term, int * & topic_sum, int num_doc, int topic_num, int num_term, double ** & doc_topic_probrability, double ** & topic_term_probrability, mapi & id2word, int steps)
{
    cout<<"gibbs sampling..."<<endl;
    for(int step=0; step<steps; step++)
    {
        cout<<" steps: "<<step;
        for (int m=0; m<num_doc; m++)
        {
            int n = 0;
            int item = get<1>(instances[m]);

            for (auto & word : docs[m])
            {
                int current_topic = z[m][n];
                topic_term[current_topic][word] -= 1;
                topic_sum[current_topic] -= 1;

                int new_topic = get_topic(item, word, topic_term, topic_sum, doc_topic_probrability);
                topic_term[new_topic][word] += 1;
                topic_sum[new_topic] += 1;
                z[m][n] = new_topic;
                n++;
            }
        }
    }
    for (int k = 0; k < topic_num; k++) {
        for (int w = 0; w < id2word.size(); w++) {
            topic_term_probrability[k][w] = double (topic_term[k][w] + beta) / (topic_sum[k] + num_term * beta);
        }
    }
    lda::save_model(topic_term_probrability, id2word);
}

void nmf_part(sgd::instance_type & instances, matrix & docs, matrix & z, double ** & user_feature, double ** & item_feature, double * & user_bias, double * item_bias, double & glob_bias, int user_num, int item_num,double ** & doc_topic_probrability, double ** & topic_term_probrability, int steps)
{
    cout<<endl;
    for(int step=0; step<steps; step++)
    {
        cout<<" step"<<step;
        double sumloss = 0.0;
        double totalloss = 0.0;
        int count = 0;
        for (auto instance:instances)
        {
            my_array doc = docs[count];
            my_array doc_z = z[count];
            int user = get<0>(instance);
            int item = get<1>(instance);
            int rating = get<2>(instance);
            double pred = sgd::predict(user_feature, item_feature, user, item, user_bias, item_bias, glob_bias);
            sumloss += squareloss(pred, rating);
            totalloss += loss(pred, rating, doc, doc_z, item, doc_topic_probrability, topic_term_probrability);
            count ++;
            
            double mult = 2*(pred - rating);
            glob_bias -= learning_rate * (mult + 2 * reg_0 * glob_bias);
            user_bias[user] -= learning_rate *(mult + 2*reg_0*user_bias[user]);
            item_bias[item] -= learning_rate *(mult + 2*reg_0*item_bias[item]);

            for (int i=0; i< topic_num; i++)
            {
                user_feature[user][i] -= learning_rate*(mult*item_feature[item][i] + reg_1*user_feature[user][i]);
                //item_feature[item][i] -= learning_rate*(mult*user_feature[user][i] + reg_1*item_feature[item][i]);
            }
        }
        if (step == steps-1){
        printf("sumloss rmse (%f)  ", sqrt(sumloss/count));
        printf("total rmse (%f) \n", sqrt(totalloss/count));
        }
    }

}


double loss(double p, double y, my_array & doc, my_array & doc_z, int item_id, double ** & doc_topic_probrability, double ** & topic_term_probrability)
{
    double first_part = squareloss(p, y);
    double second_part = likelihood(doc, doc_z, item_id, doc_topic_probrability, topic_term_probrability);
    return first_part - reg_likelihood*second_part;
}

void hiddenfeature2probablity(double ** & item_feature, double ** & doc_topic_probrability, int num_item, int topic_num)
{
    for (int i=0; i< num_item; i++)
    {
        double sum = 0;
        for (int j=0; j< topic_num; j++)
            sum += exp(feature_scaler*item_feature[i][j]);

        for (int j=0; j< topic_num; j++)
        {
            doc_topic_probrability[i][j] = exp(feature_scaler*item_feature[i][j])  / sum;
        }
    }
}

double likelihood(my_array & doc, my_array & doc_z, int item_id, double ** & doc_topic_probrability, double ** & topic_term_probrability)
{
   double res = 1.0;
   int i = 0;
   for ( auto word : doc)
   {
       int k = doc_z[i];
       double topic_prob = doc_topic_probrability[item_id][k];
       double term_prob = topic_term_probrability[k][word];
       res +=  log(topic_prob*term_prob);
       i++; 
   }
   return res;
}


int get_topic(int item, int word, int ** & topic_term, int * & topic_sum, double ** & doc_topic_probrability)
{
    double p[topic_num];
    for( int i=0; i < topic_num; i++)
    {
        p[i] =  doc_topic_probrability[item][i]* (topic_term[i][word] + alpha)/((topic_sum[i] + topic_num*alpha)-1 );
    }
    for (int k = 1; k < topic_num; k++) {
        p[k] += p[k - 1];
    }
    double u = ((double)random() / RAND_MAX) * p[topic_num - 1];

    for (int topic = 0; topic < topic_num; topic++) {
        if (p[topic] >=  u) {
            return topic;
        }
    }
    
}

