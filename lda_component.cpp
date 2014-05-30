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
#include "lda_component.h"
#include "util.h"

using namespace lda;
using namespace util;
using namespace std;



int lda::a_main(int argc, const char * argv[])
{
    //load text..
    matrix docs;
    matrix z_docs;
    maps word2id;
    mapi id2word;

    maps corpus2id;
    mapi id2corpus;

    load_data(docs, z_docs,  word2id, id2word, corpus2id, id2corpus);
    int num_term = word2id.size();
    int num_doc = docs.size();

    cout<<"num doc "<<num_doc<<endl;
    cout<<"num term "<<num_term<<endl;

    int ** doc_topic = create_int_matrix(num_doc, num_topic);
    int ** topic_term = create_int_matrix(num_topic, num_term);

    int *doc_sum = new int[num_doc]();
    double **theta = create_double_matrix(num_doc, num_topic);
    double **phi = create_double_matrix(num_topic, num_term);
    int *topic_sum = new int[num_topic]();

    lda_model(docs, z_docs, doc_topic, topic_term, doc_sum, topic_sum, num_topic, num_doc, num_term, theta, phi);

    save_model(phi, id2word);

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

void lda::save_model(double ** &phi, mapi & id2word){
    FILE * fout = fopen("twords.txt", "w");
    mapi::iterator it;
    for (int k = 0; k < num_topic; k++) {
        vector<pair<int, double> > words_probs;
        pair<int, double> word_prob;
        for (int w = 0; w < id2word.size(); w++) {
            word_prob.first = w;
            word_prob.second = phi[k][w];
            words_probs.push_back(word_prob);
        }

        // quick sort to sort word-topic probability
        //quicksort(words_probs, 0, words_probs.size() - 1); 
        sort(words_probs.begin(), words_probs.end(), sort_pred());
        fprintf(fout, "Topic %dth:\n", k);
        //cout<<"Topic: "<<k<<"\n";
        for (int i = 0; i < twords; i++) {
            it = id2word.find(words_probs[i].first);
            if (it != id2word.end()) {
                fprintf(fout, "\t%s   %f\n", (it->second).c_str(), words_probs[i].second);
            }
        }
    }

    fclose(fout);
}




void lda::lda_model(const matrix & docs, matrix & z, int ** &doc_topic, int ** &topic_term, int * &doc_sum, int * &topic_sum, const int num_topic, const int num_doc, const int num_term, double ** & theta, double ** &phi)
{
    // initilize....
    cout<<"initilize...."<<endl;
    srandom(time(0));
    for (int m=0; m<num_doc; m++)
    {
        int n = 0;
        for (auto &word :docs[m])
        {
            int new_topic = get_topic();
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

int lda::get_topic()
    { 
        int res = int (((double) random()/RAND_MAX) *num_topic);
    return res ;
}

int lda::get_topic(int m, int word, int num_word, int ** & doc_topic, int ** & topic_term, int * & doc_sum, int * & topic_sum )
{
    double p[num_topic];
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


void lda::load_data(matrix & docs, matrix & z_docs, maps & word2id, mapi & id2word, maps & corpus2id, mapi & id2corpus, string file_name)
{
    fstream in_stream;
    in_stream.open(file_name);
    string line;
    while(getline(in_stream, line))
    {
        istringstream iss(line);
        string corpus;
        getline(iss, corpus, ' ');
        int corpus_id = get_id(corpus, corpus2id, id2corpus);

        my_array words;
        string word;
        while(getline(iss, word, ' '))
        {
            if (word.compare(corpus))
            {
            int word_id = get_id(word, word2id, id2word);
            words.push_back(word_id);
            }
        }
        docs.push_back(words);
        my_array z(words.size());
        z_docs.push_back(z);
    }
    in_stream.close();
    cout<<"load text finished..."<<endl;
}

