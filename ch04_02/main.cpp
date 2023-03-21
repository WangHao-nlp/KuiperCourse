#include <glog/logging.h>

int main() {

     google::InitGoogleLogging("MyKuiper");
	LOG(INFO) << "Kuiper Infer Course";
	return 0;
}