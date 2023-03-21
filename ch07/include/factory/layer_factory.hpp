//
// Created by fss on 22-12-21.
//

#ifndef KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#define KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#include "ops/op.hpp"
#include "layer/layer.hpp"

namespace kuiper_infer {
class LayerRegisterer {
 public:
  typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op);
  // 函数指针类型，我们将存放参数的Oprator类传入到该方法中，然后该方法根据Operator内的参数返回具体的Layer.

  typedef std::map<OpType, Creator> CreateRegistry;
  // value是用于创建该层的对应方法(Creator)

  static void RegisterCreator(OpType op_type, const Creator &creator);

  static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Operator> &op);

  static CreateRegistry &Registry();
};

class LayerRegistererWrapper {
 public:
  //op_type是算子的类型，作为Layer注册表的key使用，creator是创建具体层的工厂方法，作为Layer注册表的value
  LayerRegistererWrapper(OpType op_type, const LayerRegisterer::Creator &creator) {
    LayerRegisterer::RegisterCreator(op_type, creator);
  }
};

}
#endif //KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
