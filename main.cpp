//
//  main.cpp
//  lda
//
//  Created by Yu Peng on 5/24/14.
//  Copyright (c) 2014 yupeng. All rights reserved.
//

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include "util.h"
#include "main.h"


using namespace std;


int main(int argc, const char * argv[])
{
    //load text..
    fstream in_stream;
    in_stream.open("new_data");
    string line;
    matrix docs;
    matrix z_docs;
    while(getline(in_stream, line))
    {
        istringstream iss(line);
        array words;
        string word;
        while(getline(iss, word, ' '))
        {
            int word_id = get_word_id(word);
            words.push_back(word_id);
        }
        docs.push_back(words);
        array z(words.size());
        z_docs.push_back(z);
    }
    in_stream.close();
    cout<<"load text finished..."<<endl;

    num_term = word2id.size();
    num_doc = docs.size();


    int ** doc_topic = create_int_matrix(num_doc, num_topic);
    int **topic_term = create_int_matrix(num_topic, num_term);

    int *doc_sum = new int[num_doc]();
    double **theta = create_double_matrix(num_doc, num_topic);
    double **phi = create_double_matrix(num_topic, num_term);
    int *topic_sum = new int[num_topic]();
    lda(docs, z_docs, doc_topic, topic_term, doc_sum, topic_sum, num_topic, num_doc, num_term, theta, phi);
    save_model(num_term, phi);
    clear_mvector(docs, num_doc);
    clear_mvector(z_docs, num_doc);
    clear_matrix(doc_topic, num_doc);
    clear_matrix(topic_term, num_topic);
    clear_array(topic_sum);
    clear_array(doc_sum);
    clear_matrix(phi, num_topic);
    clear_matrix(theta, num_doc);

    return 0;
}

void save_model(int V, double ** &phi){
    FILE * fout = fopen("twords.txt", "w");
    mapid2word::iterator it;
    for (int k = 0; k < num_topic; k++) {
        vector<pair<int, double> > words_probs;
        pair<int, double> word_prob;
        for (int w = 0; w < V; w++) {
            word_prob.first = w;
            word_prob.second = phi[k][w];
            words_probs.push_back(word_prob);
        }

        // quick sort to sort word-topic probability
        quicksort(words_probs, 0, words_probs.size() - 1);
        //sort(words_probs.begin(), words_probs.end(), sort_pred());
        fprintf(fout, "Topic %dth:\n", k);
        cout<<"Topic: "<<k<<"\n";
        for (int i = 0; i < twords; i++) {
            it = id2word.find(words_probs[i].first);
            if (it != id2word.end()) {
                fprintf(fout, "\t%s   %f\n", (it->second).c_str(), words_probs[i].second);
            }
        }
    }

    fclose(fout);
}



int get_word_id(string word){
    mapword2id::iterator it = word2id.begin();
    it = word2id.find(word);
    if (it == word2id.end()) {
    word2id.insert(pair<string, int>(word, word2id.size()));
    id2word.insert(pair<int, string>(word2id.size(), word));
    return word2id.size();
    } else{
    return it->second;
    }
}

string get_word(int word_id){
    mapid2word::iterator it = id2word.begin();
    it = id2word.find(word_id);
    if (it == id2word.end()) {
    return "error";
    } else{
    return it->second;
    }
}


void lda(const matrix & docs, matrix & z, int ** &doc_topic, int ** &topic_term, int * &doc_sum, int * &topic_sum, const int num_topic, const int num_doc, const int num_term, double ** & theta, double ** &phi)
{
    // initilize....
    
    cout<<"initilize...."<<endl;
    srandom(time(0));
    for (int m=0; m<num_doc; m++)
    {
        int n = 0;
        for (auto &word :docs[m])
        {
            int new_topic = get_topic(num_topic);
            doc_topic[m][new_topic] += 1;
            doc_sum[m] += 1;
            topic_term[new_topic][word] += 1;
            topic_sum[new_topic] += 1;
            z[m][n] = new_topic;
            n ++;
        }
    }
    //gibbs sampling...
    cout<<"gibbs sampling..."<<endl;
    for(int step=0; step<steps; step++)
    {
        cout<<"steps: "<<step<<endl;
        for (int m=0; m<num_doc; m++)
        {
            int n = 0 ;
            for (auto & word : docs[m])
            {
                int current_topic = z[m][n];
                doc_topic[m][current_topic] -= 1;
                doc_sum[m] -= 1;
                topic_term[current_topic][word] -= 1;
                topic_sum[current_topic] -= 1;

                int new_topic =  get_topic(m, word, num_term, doc_topic, topic_term, doc_sum, topic_sum);
                doc_topic[m][new_topic] += 1;
                doc_sum[m] += 1;
                topic_term[new_topic][word] += 1;
                topic_sum[new_topic] += 1;
                z[m][n] = new_topic;
            
                n++;
            }
        }
    }
    // calculate theta(doc_topic_distribution) and topic_term_distribution
    cout<<"calculating..."<<endl;

    for (int m = 0; m < num_doc; m++) {
        for (int k = 0; k < num_topic; k++) {
            theta[m][k] = (doc_topic[m][k] + alpha) / (doc_sum[m] + num_topic * alpha);
        }
    }
    for (int k = 0; k < num_topic; k++) {
        for (int w = 0; w < num_term; w++) {
            phi[k][w] = double (topic_term[k][w] + beta) / (topic_sum[k] + num_term * beta);
        }
    }
}

int get_topic(int K)
    { 
    using namespace std;
        int res = int (((double) random()/RAND_MAX) *K);
    return res ;
}

int get_topic(int m, int word, int num_word, int ** & doc_topic, int ** & topic_term, int * & doc_sum, int * & topic_sum )
{
    using namespace std;
    for( int i=0; i < num_topic; i++)
    {
        p[i] =  (doc_topic[m][i] + beta)/(doc_sum[m] + beta*num_word) *(topic_term[i][word] + alpha)/((topic_sum[i] + num_topic*alpha)-1);
    }
    for (int k = 1; k < num_topic; k++) {
        p[k] += p[k - 1];
    }
    double u = ((double)random() / RAND_MAX) * p[num_topic - 1];
    for (int topic = 0; topic < num_topic; topic++) {
        if (p[topic] >=  u) {
            return topic;
        }
    }
}

void quicksort(vector<pair<int, double> > & vect, int left, int right) {
    int l_hold, r_hold;
    pair<int, double> pivot;

    l_hold = left;
    r_hold = right;
    int pivotidx = left;
    pivot = vect[pivotidx];
    while (left < right) {
    while (vect[right].second <= pivot.second && left < right) {
        right--;
    }
    if (left != right) {
        vect[left] = vect[right];
        left++;
    }
    while (vect[left].second >= pivot.second && left < right) {
        left++;
    }
    if (left != right) {
        vect[right] = vect[left];
        right--;
    }
    }

    vect[left] = pivot;
    pivotidx = left;
    left = l_hold;
    right = r_hold;

    if (left < pivotidx) {
    quicksort(vect, left, pivotidx - 1);
    }
    if (right > pivotidx) {
    quicksort(vect, pivotidx + 1, right);
    }
}


