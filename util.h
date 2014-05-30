#ifndef X
#define X
namespace util{
    int **create_int_matrix(int , int );

    typedef std::map<int, std::string> mapi;
    typedef std::map<std::string, int> maps;
    //void quicksort(std::vector<std::pair<int, double> > & vect, int left, int right); 

    int get_id(std::string, maps &, mapi&);
    int get_id(std::string , maps &);
    std::string get_word(int , mapi &);
    typedef std::vector< int > my_array;
    typedef std::vector< my_array > matrix;

    double **create_double_matrix(int , int );

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
}

#endif
