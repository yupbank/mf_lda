#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
#include "util.h"

using namespace std;

typedef vector< tuple<int,int,int> > instance_type;
typedef map<int, int> mapi;

mapi user2id;
mapi id2user;

mapi item2id;
mapi id2item;

int get_id(int user_num, mapi &dict, mapi& dict1);
int get_word(int word_id, mapi &dict);
void nmf(instance_type &, double ** &, double ** &, double * &, double * &);

int num_factor = 5;
int num_user;
int num_item;


int main()
{
    fstream in_stream;
    in_stream.open("test");
    string line;
    instance_type instances;
    while(getline(in_stream, line))
    {
        istringstream iss(line);
        string user;
        string item;
        string rating;
        getline(iss, user, ' ');
        getline(iss, item, ' ');
        getline(iss, rating, ' ');
        int user_id = get_id(stoi(user), user2id, id2user);
        int item_id = get_id(stoi(item), item2id, id2item);
        instances.push_back(tuple<int,int,int>(user_id, item_id, stoi(rating)));
    }
    num_user = user2id.size();
    num_item = item2id.size();

    double ** user_feature = create_double_matrix(num_user, num_factor);
    double ** item_feature = create_double_matrix(num_item, num_factor);
    double * user_bias = new double[num_user]();
    double * item_bias = new double[num_item]();
    nmf(instances, user_feature, item_feature, user_bias, item_bias);
    return 0;
}
void nmf(instance_type & instances, double ** & user_feature, double ** & item_feature, double * & user_bias, double * & item_bias)
{
    for (auto instance:instances)
    {
        int user = get<0>(instance);
        int item = get<1>(instance);
        int rating = get<2>(instance);
        cout<<user<<" "<<item<<" "<<rating<<endl;

    }
}

int get_id(int user_num, mapi &dict, mapi& dict1){
    auto it = dict.begin();
    it = dict.find(user_num);
    if (it == dict.end()) {
        dict.insert(pair<int, int>(user_num, dict.size()));
        dict1.insert(pair<int, int>(dict.size(), user_num));
        return dict.size();
    } else{
        return it->second;
    }
}

int get_word(int word_id, mapi &dict){
    auto it = dict.begin();
    it = dict.find(word_id);
    if (it == dict.end()) {
        return -1;
    } else{
        return it->second;
    }
}
