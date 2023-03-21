#include <glog/logging.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
	ReluLayer::ReluLayer(const std::shared_ptr<Operator>& op) :Layer("Relu") {
		// ��ʼ������
		CHECK(op->op_type_ == OpType::kOperatorRelu) << "Operator has a wrong type: " << int(op->op_type_);

		// dynamic_cast��ʲô��˼�� �����ж�һ��opָ���ǲ���ָ��һ��relu_op���ָ��
		// ��ߵ�op����ReluOperator���͵�ָ�룬�ͱ���
		// ��������ֻ����ReluOperator���͵�ָ��
		// ����ָ�����ָ������ReluOperator���͵�ָ��
		// Ϊʲô�������캯������Ϊconst std::shared_ptr<ReluOperator> &op��
		// Ϊ�˽ӿ�ͳһ�������½ڻ�˵��

		// ָ��ת��
		ReluOperator* relu_op = dynamic_cast<ReluOperator*>(op.get());
		// һ��opʵ����һ��layer һһ��Ӧ ����relu op��һ��relu layer
		// ��Ӧ��ϵ

		CHECK(relu_op != nullptr) << "Relu operator is empty";

		// ����һ����ָ��
		this->op_ = std::make_unique<ReluOperator>(relu_op->get_thresh());
	}

	void ReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
		std::vector<std::shared_ptr<Tensor<float>>>& outputs) {

		CHECK(this->op_ != nullptr);
		CHECK(this->op_->op_type_ == OpType::kOperatorRelu);

		// inputs��vector
		const uint32_t batch_size = inputs.size();
		for (int i = 0; i < batch_size; i++) {
			CHECK(!inputs.at(i)->empty());
			const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i); 

			input_data->data().transform([&](float value) {
				float thresh = op_->get_thresh();
				if (value >= thresh) {
					return value;
				}
				else {
					return 0.f;
				}
				});
			outputs.push_back(input_data);
		}
	}

	std::shared_ptr<Layer> ReluLayer::CreateInstance(const std::shared_ptr<Operator>& op) {
		std::shared_ptr<Layer> relu_layer = std::make_shared<ReluLayer>(op);
		return relu_layer;
	}

	LayerRegistererWrapper kReluLayer(OpType::kOperatorRelu, ReluLayer::CreateInstance);
}