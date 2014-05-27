#include <vector>
#include "util.h"


int **create_int_matrix(int x, int y)
{
    int **me_doc_topic = new int* [x];
    for(int i=0; i<x; i++)
    {
        me_doc_topic[i] = new int[y](); 
    }
    return me_doc_topic;
    
}
template<typename T>
void clear_matrix(T x, int t)
{
    for(int i=0; i< t; i++)
    {
    delete [] x[i]; 
    }
    delete [] x;
}

template<typename T>
void clear_array(T x)
{
    delete [] x;
}

template<typename T>
void clear_mvector( T matrix, int x)
{
    for(int i=0; i< x; i++)
    {
        matrix[i].clear(); 
    }
    matrix.clear();
}

double **create_double_matrix(int x, int y)
{
    double **me_doc_topic = new double* [x];
    for(int i=0; i<x; i++)
    {
        me_doc_topic[i] = new double[y](); 
    }
    return me_doc_topic;
    
}
