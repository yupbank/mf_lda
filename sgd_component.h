#ifndef SGD_INI
#define SGD_INI
#include "util.h"

namespace sgd{
    const int num_factor = 10;

    const double learning_rate = 0.01;
    const double reg_0 = 0;
    const double reg_1 = 0.05;

    typedef std::vector< std::tuple<int,int,int> > instance_type;

    void a_main();


    void load_data(util::matrix &, instance_type & instances, util::maps &, util::mapi &, util::maps &, util::mapi &, util::maps&, util::mapi &, std::string file_name="new_data");

    void nmf(instance_type &, double ** &, double ** &, double * &, double * &, int &, int &, int &, double &);

    double predict(double ** &, double **& , int , int , double * & , double * & , double & );

    void init_features(double ** & user_feature, double ** & item_feature, int num_factor, int num_user, int num_item);
}
#endif
