#include <gtest/gtest.h>
#include <armadillo>
#include <glog/logging.h>

TEST(test_first, demo1){
    LOG(INFO)<<"My First test!";
    arma::fmat in_1(32, 16, arma::fill::ones);
    ASSERT_EQ(in_1.n_cols, 16);
    ASSERT_EQ(in_1.n_rows, 32);
    ASSERT_EQ(in_1.size(), 32*16);
}

TEST(test_first, linear){
    arma::fmat A = "1,2,3;"
                   "4,5,6;"
                   "7,8,9;";

    arma::fmat X = "1,1,1;"
                   "1,1,1;"
                   "1,1,1;";   

    arma::fmat bias = "1,2,3;"
                      "1,2,3;"
                      "1,2,3;"; 

    arma::fmat output(3, 3);

    output = A*X+bias;
    // 7 8 9
    // 16 17 18
    // 25 26 27

    const uint32_t cols = 3;
    for(uint32_t c=0; c<cols; c++){
        float *col_ptr = output.colptr(c);
        ASSERT_EQ(*(col_ptr+0),7+c);
        ASSERT_EQ(*(col_ptr+1),16+c);
        ASSERT_EQ(*(col_ptr+2),25+c);
    }           
}