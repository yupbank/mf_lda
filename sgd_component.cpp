#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <stdlib.h>  
#include <map>
#include <cmath>
#include "sgd_component.h"
#include "util.h"

using namespace std;
using namespace sgd;
using namespace util;

inline double squareloss(double p, double y)
{
   return 0.5 * (p - y) * (p - y); 
}


void sgd::a_main()
{
    int steps = 20;
    double glob_bias = 0.0;

    matrix docs;
    maps user2id;
    mapi id2user;

    maps item2id;
    mapi id2item;

    maps word2id;
    mapi id2word;

    instance_type instances;
    load_data(docs, instances, word2id, id2word, user2id, id2item, item2id, id2user);
    int num_user = user2id.size();
    int num_item = item2id.size();
    cout<<"user length, "<<num_user<<endl; 
    cout<<"item length, "<<num_item<<endl; 
    cout<<"term length, "<<word2id.size()<<endl; 

    double ** user_feature = create_double_matrix(num_user, num_factor);
    double ** item_feature = create_double_matrix(num_item, num_factor);
    double * user_bias = new double[num_user]();
    double * item_bias = new double[num_item]();
    cout<<"start tranning.."<<endl; 

    nmf(instances, user_feature, item_feature, user2id, item2id, user_bias, item_bias, num_user, num_item, steps, glob_bias);


    instances.clear();
    user2id.clear();
    id2user.clear();
    item2id.clear();
    id2item.clear();

    word2id.clear();
    id2word.clear();

    clear_matrix(user_feature, num_user);
    clear_matrix(item_feature, num_item);
    clear_array(user_bias);
    clear_array(item_bias);
    clear_mvector(docs, docs.size());
}

void sgd::load_data(matrix & docs, instance_type & instances, maps & word2id, mapi & id2word, maps & user2id, mapi & id2item, maps & item2id, mapi & id2user, string file_name)
{
    fstream in_stream;
    in_stream.open(file_name);
    string line;
    while(getline(in_stream, line))
    {
        istringstream iss(line);
        string user;
        string item;
        string rating;
        getline(iss, user, ' ');
        getline(iss, item, ' ');
        getline(iss, rating, ' ');
        int user_id = get_id(user, user2id, id2user);
        int item_id = get_id(item, item2id, id2item);
        instances.push_back(tuple<int,int,int>(user_id, item_id, stoi(rating)));
        my_array words;
        string word;
        while(getline(iss, word, ' '))
        {
            int word_id = get_id(word, word2id, id2word);
            words.push_back(word_id);
        }
        docs.push_back(words);
        my_array z(words.size());
    }
    cout<<"load data finished..."<<endl; 
}

void sgd::init_features(double ** & user_feature, double ** & item_feature, int num_factor, int num_user, int num_item)
{
    cout<<"init fatures..."<<endl;
    for(int i=0; i< num_user; i++)
    {
        for(int j=0; j<num_factor; j++)
        {
            user_feature[i][j] = drand48();
        }
    }

    for(int i=0; i< num_item; i++)
    {
        for(int j=0; j<num_factor; j++)
        {
            item_feature[i][j] = drand48();
        }
    }

}

void sgd::nmf(instance_type & instances, double ** & user_feature, double ** & item_feature, maps & user2id, maps & item2id, double * & user_bias, double * & item_bias, int & num_user, int & num_item, int & steps, double & glob_bias)
{
    init_features(user_feature, item_feature, num_factor, num_user, num_item);

    instance_type test_instances;
    mapi test_id2user;
    mapi test_id2item;
    load_test_data(test_instances, test_id2user, test_id2item);

    for(int step=0; step<steps; step++)
    {
        double sumloss = 0.0;
        int count = 0;
        for (auto instance:instances)
        {
            int user = get<0>(instance);
            int item = get<1>(instance);
            int rating = get<2>(instance);
            double pred = predict(user_feature, item_feature, user, item, user_bias, item_bias, glob_bias);
            sumloss += squareloss(pred, rating);
            count ++;
            
            double mult = 2*(pred - rating);
            glob_bias -= learning_rate * (mult + 2 * reg_0 * glob_bias);
            user_bias[user] -= learning_rate *(mult + 2*reg_0*user_bias[user]);
            item_bias[item] -= learning_rate *(mult + 2*reg_0*item_bias[item]);

            for (int i=0; i< num_factor; i++)
            {
                user_feature[user][i] -= learning_rate*(mult*item_feature[item][i] + reg_1*user_feature[user][i]);
                item_feature[item][i] -= learning_rate*(mult*user_feature[user][i] + reg_1*item_feature[item][i]);
            }
        }
        cout<<"\n step: "<<step<<endl;
        printf("tranning rmse (%f)  couts(%d) \n", sqrt(sumloss/count), count);
        test(test_instances, test_id2user, test_id2item, user2id, item2id,  user_feature, item_feature, user_bias, item_bias, glob_bias);
    }
}

double sgd::predict(double ** & user_feature, double **& item_feature, int user_id, int item_id, double * & user_bias, double * & item_bias, double & glob_bias)
{
    double result = 0;
    for (int f=0; f<num_factor; f++)
    {
        result = result + user_feature[user_id][f]*item_feature[item_id][f];
    }
    result = result + user_bias[user_id] + item_bias[item_id];
    result += glob_bias;
    return result;
}

void sgd::load_test_data(instance_type & test_instances, mapi & test_id2user, mapi & test_id2item)
{
    matrix test_docs;
    maps test_user2id;
    maps test_item2id;
    mapi test_id2word;
    maps test_word2id;
    load_data(test_docs, test_instances, test_word2id, test_id2word, test_user2id, test_id2item, test_item2id, test_id2user, "test_data");
    test_user2id.clear();
    test_item2id.clear();
    test_id2word.clear();
    test_word2id.clear();
    clear_mvector(test_docs, test_docs.size());
}
void sgd::test(instance_type & test_instances, mapi & test_id2user, mapi & test_id2item, maps & user2id, maps & item2id,  double ** & user_feature, double ** & item_feature, double * & user_bias, double * & item_bias, double glob_bias)
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
            double pred = predict(user_feature, item_feature, user_id, item_id, user_bias, item_bias, glob_bias);
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


