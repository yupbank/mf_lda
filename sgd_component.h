#ifndef SGD_INI
#define SGD_INI
#include "util.h"

namespace sgd{
     static int num_factor = 5;

     static double learning_rate = 0.01;
     static double reg_0 = 0;
     static double reg_1 = 0.05;

    typedef std::vector< std::tuple<int,int,int> > instance_type;

    void a_main();


    void load_data(util::matrix &, instance_type & instances, util::maps &, util::mapi &, util::maps &, util::mapi &, util::maps&, util::mapi &, std::string file_name="train_data");

    void nmf(instance_type &, double ** &, double ** &, util::maps &, util::maps &, double * &, double * &, int &, int &, int &, double &);

    double predict(double ** &, double **& , int , int , double * & , double * & , double & );

    void init_features(double ** & user_feature, double ** & item_feature, int num_factor, int num_user, int num_item);
    void test(instance_type & test_instances, util::mapi & test_id2user, util::mapi & test_id2item, util::maps & user2id, util::maps & item2id,  double ** & user_feature, double ** & item_feature, double * & user_bias, double * & item_bias, double glob_bias);
    void load_test_data(instance_type & test_instances, util::mapi & test_id2user, util::mapi & test_id2item);
}
#endif
