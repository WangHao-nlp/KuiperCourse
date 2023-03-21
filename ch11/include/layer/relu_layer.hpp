//
// Created by fss on 22-12-20.
//

#ifndef KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#include "layer.hpp"
#include "ops/relu_op.hpp"

namespace kuiper_infer {
class ReluLayer : public Layer {
 public:
  ~ReluLayer() override = default;

  // ͨ�������relu_op�е�thresh��֪��relu layer, ��Ϊ�����ʱ��Ҫ�õ�
  explicit ReluLayer(const std::shared_ptr<Operator> &op);

  // ִ��relu �����ľ��庯��Forwards
  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

 private:
  std::unique_ptr<ReluOperator> op_;
};
}
#endif //KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
