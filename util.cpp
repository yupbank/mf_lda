#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include "util.h"

using namespace std;
using namespace util;

int ** util::create_int_matrix(int x, int y)
{
    int **me_doc_topic = new int* [x];
    for(int i=0; i<x; i++)
    {
        me_doc_topic[i] = new int[y](); 
    }
    return me_doc_topic;
    
}

double ** util::create_double_matrix(int x, int y)
{
    double **me_doc_topic = new double* [x];
    for(int i=0; i<x; i++)
    {
        me_doc_topic[i] = new double[y](); 
    }
    return me_doc_topic;
    
}

int util::get_id(string user_num, maps &dict, mapi& dict1){
    auto it = dict.begin();
    it = dict.find(user_num);
    if (it == dict.end()) {
        int num = dict.size();
        dict.insert(pair<string, int>(user_num, num));
        dict1.insert(pair<int, string>(num, user_num));
        return num;
    } else{
        return it->second;
    }
}

string util::get_word(int word_id, mapi &dict1){
    auto it = dict1.begin();
    it = dict1.find(word_id);
    if (it == dict1.end()) {
        return "error";
    } else{
        return it->second;
    }
}

//void quicksort(vector<pair<int, double> > & vect, int left, int right) 
//{
//    int l_hold, r_hold;
//    pair<int, double> pivot;
//
//    l_hold = left;
//    r_hold = right;
//    int pivotidx = left;
//    pivot = vect[pivotidx];
//    while (left < right) {
//        while (vect[right].second <= pivot.second && left < right) {
//            right--;
//        }
//        if (left != right) {
//            vect[left] = vect[right];
//            left++;
//        }
//        while (vect[left].second >= pivot.second && left < right) {
//            left++;
//        }
//        if (left != right) {
//            vect[right] = vect[left];
//            right--;
//        }
//    }
//
//    vect[left] = pivot;
//    pivotidx = left;
//    left = l_hold;
//    right = r_hold;
//
//    if (left < pivotidx) {
//        quicksort(vect, left, pivotidx - 1);
//    }
//    if (right > pivotidx) {
//        quicksort(vect, pivotidx + 1, right);
//    }
//}

